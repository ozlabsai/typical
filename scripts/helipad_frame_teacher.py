"""Claude vision teacher for the frame questions: every option's probability per keyframe, cached.

uv run scripts/helipad_frame_teacher.py --frames <frames_dir>/frames.json --out <answers.jsonl> [--limit N --teacher <id>
    --runner openrouter|cli --workers 8 --cache <dir>]
OpenRouter (default) sends the image as an OpenAI-style image_url data URI; the key is OPENROUTER_API_KEY (environment, or
the repo .env, as scripts/r053_teacher_label.py reads it). Calls are cached by (image sha256, question-set version, teacher),
so reruns are free; a reply that fails validation is retried once, then recorded as {"failed": true} (and not cached).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from helipad.frames import FRAME_QUESTIONS, NULL_OPTION, QSET_VERSION  # noqa: E402

TEACHER = "anthropic/claude-fable-5.2"
MAX_USD = 40.0   # ponytail: a hard stop for the whole run, raise it for Plan 3 volumes
GUIDE = """\
How to read the questions:
- Heights and distances are of the single tallest object in view; distance is from the camera.
- "tallest_type" is "none" only when the view really holds no object above the ground (open field to the horizon).
- "ground" and "vegetation_belly" are about the foreground ground surface right in front of the camera.
- "dust" is how much dust a landing helicopter's downwash would raise from that foreground ground.
- Not seen never means absent. Answer "no" only when the image clearly shows the place where the thing would be and it is
  not there. When the frame does not show it (pointing at the sky or the ground only, blurred, dark, too far to tell),
  use the null option when the image does not show it."""


def prompt(questions: dict = FRAME_QUESTIONS) -> str:
    lines = ["You are labelling one video keyframe taken at a candidate helicopter landing site.",
             "Answer every question below with a probability for every one of its options, the null option included:",
             f'the null option is "{NULL_OPTION}". The probabilities of each question must sum to 1.', ""]
    for qid, (question, opts) in questions.items():
        lines.append(f"{qid}: {question} Options: " + ", ".join(f'"{o}"' for o in (*opts, NULL_OPTION)))
    lines += ["", GUIDE, "",
              'Reply with one JSON object and nothing else: {"<question id>": {"<option text>": <probability>, ...}, ...}']
    return "\n".join(lines)


def validate(reply: str, questions: dict = FRAME_QUESTIONS) -> dict | None:
    """The first JSON object in the reply -> {qid: {option: p}} over every option, or None. Every qid, only known options,
    no negative p; each question's total must be within 0.95-1.05 and is renormalised."""
    start = reply.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(reply[start:])
    except ValueError:
        return None
    if not isinstance(obj, dict) or set(obj) != set(questions):
        return None
    out = {}
    for qid, (_, opts) in questions.items():
        d, allowed = obj[qid], (*opts, NULL_OPTION)
        if not isinstance(d, dict) or set(d) - set(allowed):
            return None
        try:
            probs = {o: float(d.get(o, 0.0)) for o in allowed}
        except (TypeError, ValueError):
            return None
        total = sum(probs.values())
        if any(p < 0 for p in probs.values()) or not 0.95 <= total <= 1.05:
            return None
        out[qid] = {o: p / total for o, p in probs.items()}
    return out


def cache_key(sha256: str, qset: str, teacher: str) -> str:
    return hashlib.sha256(f"{sha256}|{qset}|{teacher}".encode()).hexdigest()


def label(frames: list[dict], cache_dir: Path, runner, root: Path = Path("."), teacher: str = TEACHER,
          workers: int = 1) -> list[dict]:
    """One record per frame, in order: the frame's fields + {teacher, qset, answers} or + {failed: true}."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    text = prompt()

    def one(f: dict) -> dict:
        base = {**f, "teacher": teacher, "qset": QSET_VERSION}
        hit = cache_dir / f"{cache_key(f['sha256'], QSET_VERSION, teacher)}.json"
        if hit.exists():
            return {**base, "answers": json.loads(hit.read_text())}
        for _ in range(2):
            answers = validate(runner(Path(root) / f["path"], text))
            if answers is not None:
                hit.write_text(json.dumps(answers))
                return {**base, "answers": answers}
        return {**base, "failed": True}

    with ThreadPoolExecutor(max(1, workers)) as pool:
        return list(pool.map(one, frames))


class OpenRouterRunner:
    """POST image + prompt; tokens and cost from the response usage are added to self.spent."""

    def __init__(self, model: str):
        from r053_teacher_label import OPENROUTER, _api_key
        self.model, self.url, self.key = model, OPENROUTER, _api_key()
        self.spent = {"calls": 0, "usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "call_s": 0.0}
        self.lock = threading.Lock()

    def __call__(self, image: Path, text: str) -> str:
        import httpx
        if self.spent["usd"] >= MAX_USD:
            raise RuntimeError(f"teacher budget ${MAX_USD} spent")
        uri = "data:image/jpeg;base64," + base64.b64encode(Path(image).read_bytes()).decode()
        body = {"model": self.model, "max_tokens": 6000, "reasoning": {"effort": "low"}, "usage": {"include": True},
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}},
                                                          {"type": "text", "text": text}]}]}
        t0 = time.time()
        for attempt in range(4):
            try:
                r = httpx.post(self.url, json=body, timeout=600, headers={"Authorization": f"Bearer {self.key}"})
                out = r.json()
                if r.status_code == 200 and out.get("choices"):
                    break
                err = f"HTTP {r.status_code}: {r.text[:200]}"
            except (httpx.HTTPError, ValueError) as e:
                err = f"{type(e).__name__}: {e}"
            if attempt == 3:
                raise RuntimeError(f"OpenRouter failed: {err}")
            time.sleep(5 * 2 ** attempt)
        u = out.get("usage") or {}
        with self.lock:
            self.spent["calls"] += 1
            self.spent["usd"] += float(u.get("cost") or 0)
            self.spent["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
            self.spent["completion_tokens"] += int(u.get("completion_tokens") or 0)
            self.spent["call_s"] += time.time() - t0
        return out["choices"][0]["message"].get("content") or ""


def cli_runner(model: str):
    """Fallback: `claude -p` reads the image file itself with the Read tool."""
    def run(image: Path, text: str) -> str:
        return subprocess.run(["claude", "-p", "--model", model, "--allowedTools", "Read", "--output-format", "text",
                               f"{text}\n\nRead the image at {Path(image).resolve()}"],
                              capture_output=True, text=True, timeout=600).stdout
    return run


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, help="default: <out dir>/cache")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--runner", choices=("openrouter", "cli"), default="openrouter")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    frames = json.loads(a.frames.read_text())
    frames = frames[:a.limit] if a.limit else frames
    runner = OpenRouterRunner(a.teacher) if a.runner == "openrouter" else cli_runner(a.teacher)
    t0 = time.time()
    recs = label(frames, a.cache or a.out.parent / "cache", runner, root=a.frames.parent, teacher=a.teacher,
                 workers=a.workers)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    failed = sum(bool(r.get("failed")) for r in recs)
    spent = getattr(runner, "spent", {})
    print(json.dumps({"frames": len(recs), "failed": failed, "wall_s": round(time.time() - t0, 1), **spent,
                      "usd_per_new_call": round(spent["usd"] / spent["calls"], 4) if spent.get("calls") else None}))


if __name__ == "__main__":
    main()
