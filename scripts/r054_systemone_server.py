"""R054 Phase 0: a TypeSafe-protocol `POST /v1/systemone` endpoint over one of two backends.

  llm: WorkflowEvals' own system_one_adapter LLM path (probability system prompt, JSON schema,
       one request per question) against an OpenAI-compatible vLLM server. Used for JevBench so it
       is asked exactly the way the WorkflowEvals `vllm:` provider asks.
       uv run --project <WorkflowEvals> python r054_systemone_server.py llm vllm:google/gemma-4-12B-it \
           --thinking default --base-url http://127.0.0.1:8000/v1 --port 8010
  d1:  LiquidAI/d1-3B's shipped native readout (`model.system_one`, trust_remote_code).
       python r054_systemone_server.py d1 LiquidAI/d1-3B --revision <sha> --port 8011
  clef: Cloudflare/clef-flash's shipped reference path (`joint_schema_model.systemone`, joint head, bf16).
       python r054_systemone_server.py clef Cloudflare/clef-flash --revision <sha> --port 8012

The response `model` is the served name, so WorkflowEvals' `typical:` provider rejects any mismatch.
"""
import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def llm_backend(args):
    sys.path.insert(0, str(Path(args.workflowevals).resolve()))
    from core.clients import QuestionClient
    from core.providers import configuration, resolve_model
    from core.question_types import Noul, Choice, Score, response_record
    from pydantic import TypeAdapter

    config = configuration(args.model, thinking=args.thinking, base_url=args.base_url,
                           question_mode="one", max_output_tokens=args.max_output_tokens, timeout_s=3600)
    model = resolve_model(config)
    parse = TypeAdapter(Noul | Choice | Score).validate_python

    def answer(state, questions):
        client = QuestionClient(config)  # ponytail: one client per request; the adapter is cheap to build
        try:
            response = client.system_one(model=model, state=state,
                                         questions={k: parse(q) for k, q in questions.items()})
        finally:
            client.close()
        return response_record(response)

    return config.identity.name, answer


def d1_backend(args):
    import torch
    from transformers import AutoModel

    net = AutoModel.from_pretrained(args.model, revision=args.revision, trust_remote_code=True,
                                    dtype=torch.bfloat16).to("cuda").eval()
    lock = threading.Lock()  # ponytail: one GPU forward at a time; batching only if d1 is the bottleneck

    def answer(state, questions):
        with lock, torch.inference_mode():
            return net.system_one(state, questions)

    return args.model.rsplit("/", 1)[-1], answer


def clef_backend(args):
    import torch
    from huggingface_hub import snapshot_download

    path = snapshot_download(args.model, revision=args.revision)
    sys.path.insert(0, path)  # the release's own code, as in its model card
    from joint_schema_model import load_release_model, systemone

    model, processor = load_release_model(path, device="cuda", dtype=torch.bfloat16)
    lock = threading.Lock()  # ponytail: one GPU forward at a time, as d1

    def answer(state, questions):
        with lock:
            return systemone(model, processor, {"model": args.model, "state": state, "questions": questions})

    return args.model.rsplit("/", 1)[-1], answer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backend", choices=("llm", "d1", "clef"))
    ap.add_argument("model")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--served-name", help="model id echoed in responses (default: backend identity)")
    ap.add_argument("--thinking", default="default")
    ap.add_argument("--base-url")
    ap.add_argument("--max-output-tokens", type=int, default=16384)
    ap.add_argument("--revision")
    ap.add_argument("--workflowevals", default="WorkflowEvals")
    args = ap.parse_args()
    name, answer = {"llm": llm_backend, "d1": d1_backend, "clef": clef_backend}[args.backend](args)
    name = args.served_name or name

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send(200, {"data": [{"id": name}]})

        def do_POST(self):
            if not self.path.startswith("/v1/systemone"):
                return self.send(404, {"error": "not found"})
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                out = answer(req.get("state"), req["questions"])
            except Exception as exc:  # surfaced to the harness as a failed call, never hidden
                print(f"error: {type(exc).__name__}: {str(exc)[:500]}", flush=True)
                return self.send(500, {"error": f"{type(exc).__name__}: {str(exc)[:500]}"})
            self.send(200, {**out, "model": name})

    print(f"serving {name} on :{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
