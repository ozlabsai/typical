#!/usr/bin/env python3
"""R056 teacher labeling without the HTTP shim: WorkflowEvals' own `vllm:` client (r054_systemone_server's
llm backend: probability system prompt + schema, one question per request) called in-process from a thread pool,
spread over several vLLM servers (least busy first). The stdlib shim stalled with ~180 queued connections while
its vLLM ran 3-9 requests (R055 saw the same), so this replaces `r055_build_mix.py label` for R056. Resumable:
(id, qid) pairs already answered in --out are skipped; errors are recorded and retried on the next run.

  cd WorkflowEvals && uv run --frozen python r056_label.py --data in.jsonl --out labels.jsonl \
      --model vllm:Qwen/Qwen3.8-27B-FP8 --base-url http://127.0.0.1:8021/v1 --base-url http://127.0.0.1:8022/v1
"""
import argparse
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r054_systemone_server import llm_backend  # noqa: E402
from r055_build_mix import wire_question  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", action="append", required=True)
    ap.add_argument("--workers", type=int, default=64, help="per server")
    ap.add_argument("--thinking", default="default")
    ap.add_argument("--workflowevals", default=".")
    args = ap.parse_args()
    backends = {url: llm_backend(SimpleNamespace(workflowevals=args.workflowevals, model=args.model,
                                                 thinking=args.thinking, base_url=url, max_output_tokens=16384))[1]
                for url in args.base_url}
    busy, lock = Counter({u: 0 for u in backends}), threading.Lock()
    done = set()
    if Path(args.out).exists():
        for line in open(args.out):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "answer" in r:
                done.add((r["id"], r["qid"]))
    rows = [json.loads(line) for line in open(args.data)]
    jobs = [(r, q) for r in rows for q in r["questions"] if (r["id"], q) not in done]
    print(f"{len(jobs)} questions to label ({len(done)} done)", flush=True)
    sink = open(args.out, "a")

    def one(job):
        row, qid = job
        with lock:
            url = min(busy, key=busy.__getitem__)
            busy[url] += 1
        try:
            out = backends[url](row["state"], {qid: wire_question(row["questions"][qid])})
            rec = {"id": row["id"], "qid": qid, "answer": out["answers"][qid]}
        except Exception as exc:  # recorded, never hidden; retried on the next run
            rec = {"id": row["id"], "qid": qid, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        finally:
            with lock:
                busy[url] -= 1
        with lock:
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
        return "ok" if "answer" in rec else "error"

    with ThreadPoolExecutor(args.workers * len(backends)) as pool:
        print(json.dumps(Counter(pool.map(one, jobs))), flush=True)


if __name__ == "__main__":
    main()
