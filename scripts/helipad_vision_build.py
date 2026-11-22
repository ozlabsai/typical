"""Vision training and test rows (helipad/frames.py grouped rows) from keyframes + teacher answers, split by video.

uv run scripts/helipad_vision_build.py --teacher <frames_v0.jsonl> --frames <frames_dir> --out <vision_v0> \
    [--videos $HELIPAD_PRIVATE/videos.json --seed 0]
train/dev: per frame a real row (teacher soft targets), a mismatch row and a gold-dropped row (R044 thirds).
test (test videos only): test_real, test_mismatch, test_shuffled (another test video's image, the original targets) and
test_noimage (no image field). --videos only stratifies the test split by round (one test video per round).

Mismatch rows show a heavily blurred frame from a *different video* with every target null. The frame questions are
generic ("are wires visible?"), so an unblurred frame from another video still answers them: it would be a second real
row with a null target, which teaches "null" on real images. A blurred frame supports no option, which is the
"can't tell" a drone or phone video really produces (motion blur, glare).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from helipad.frames import FRAME_QUESTIONS, NULL_OPTION, argmax, grouped_row  # noqa: E402

FILES = ("train", "dev", "test_real", "test_shuffled", "test_noimage", "test_mismatch")
N_TEST, N_DEV = 3, 2
BLUR = 0.06       # Gaussian radius as a share of the long side: wires, poles and ground texture are gone
DROP_MIN_P = 0.6  # gold-dropped only removes an answer the teacher is sure of
EPOCHS, BATCH = 3, 8


def split_by_video(frames: list[dict], seed: int, rounds: dict[str, str | None] | None = None) -> dict[str, list]:
    """Seeded split of whole videos: N_TEST test (one per known round first), N_DEV dev, the rest train."""
    rng = random.Random(seed)
    videos = sorted({f["video"] for f in frames})
    rng.shuffle(videos)
    test = []
    for r in sorted({x for x in (rounds or {}).values() if x}):
        test += [v for v in videos if (rounds or {}).get(v) == r and v not in test][:1]
    test = (test + [v for v in videos if v not in test])[:N_TEST]
    rest = [v for v in videos if v not in test]
    which = {**{v: "test" for v in test}, **{v: "dev" for v in rest[:N_DEV]}, **{v: "train" for v in rest[N_DEV:]}}
    return {s: [f for f in frames if which[f["video"]] == s] for s in ("train", "dev", "test")}


def _fid(rec: dict) -> str:
    return rec["path"].rsplit(".", 1)[0]


def _donor(rec: dict, pool: list[dict], rng: random.Random) -> dict:
    return rng.choice([f for f in pool if f["video"] != rec["video"]])


def build(records: list[dict], frames_dir: Path, out: Path, seed: int = 0,
          rounds: dict[str, str | None] | None = None) -> dict:
    rng = random.Random(seed)
    ok = [r for r in records if not r.get("failed")]
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)

    def image(rec: dict, blur: bool = False) -> str:
        rel = f"images/{'blur_' if blur else ''}{_fid(rec).replace('/', '_')}.jpg"
        if not (out / rel).exists():
            with Image.open(frames_dir / rec["path"]) as im:
                im = im.convert("RGB")
                im = im.filter(ImageFilter.GaussianBlur(BLUR * max(im.size))) if blur else im
                im.save(out / rel, quality=92)
        return rel

    def row(rec: dict, suffix: str, img: str | None, targets: dict, variant: str, src: dict | None = None, **meta):
        return grouped_row(_fid(rec) + suffix, img, targets, video=rec["video"], t=rec["t"], variant=variant,
                           **({"image_video": src["video"]} if src else {}), **meta)

    nulls = {q: {NULL_OPTION: 1.0} for q in FRAME_QUESTIONS}
    split = split_by_video(ok, seed, rounds)
    files: dict[str, list] = {n: [] for n in FILES}
    for s in ("train", "dev"):
        for rec in split[s]:
            files[s].append(row(rec, "", image(rec), rec["answers"], "real"))
            d = _donor(rec, split[s], rng)
            files[s].append(row(rec, "~mismatch", image(d, blur=True), nulls, "mismatch", d))
            sure = [q for q, dist in rec["answers"].items() if argmax(dist) != NULL_OPTION and max(dist.values()) >= DROP_MIN_P]
            if sure:
                q = rng.choice(sure)
                files[s].append(grouped_row(_fid(rec) + "~dropped", image(rec), rec["answers"], drop=q, video=rec["video"],
                                            t=rec["t"], variant="gold_dropped", dropped=q))
        rng.shuffle(files[s])   # the trainer evaluates the first --eval-samples dev examples
    test = split["test"]
    by_video = {v: [f for f in test if f["video"] == v] for v in dict.fromkeys(f["video"] for f in test)}
    vids = list(by_video)
    for rec in test:
        files["test_real"].append(row(rec, "", image(rec), rec["answers"], "real"))
        files["test_noimage"].append(row(rec, "", None, rec["answers"], "noimage"))
        other = by_video[vids[(vids.index(rec["video"]) + 1) % len(vids)]]
        sh = other[by_video[rec["video"]].index(rec) % len(other)]
        files["test_shuffled"].append(row(rec, "", image(sh), rec["answers"], "shuffled", sh))
        d = _donor(rec, test, rng)
        files["test_mismatch"].append(row(rec, "~mismatch", image(d, blur=True), nulls, "mismatch", d))
    for n, rows in files.items():
        (out / f"{n}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    examples = len(files["train"]) * len(FRAME_QUESTIONS)
    return {"skipped_failed": len(records) - len(ok), "rows": {n: len(r) for n, r in files.items()},
            "videos": {s: sorted({f["video"] for f in fs}) for s, fs in split.items()},
            "train_examples": examples, "steps_3_epochs": math.ceil(EPOCHS * examples / BATCH)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--videos", type=Path, help="private videos.json: stratify the test split by the site's round")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rounds = None
    if a.videos:
        from helipad.paths import SITES_JSON
        round_of = {s["public_id"]: s["round"] for s in json.loads(SITES_JSON.read_text())}
        rounds = {v: round_of.get(m["site"]) for v, m in json.loads(a.videos.read_text()).items()}
    recs = [json.loads(x) for x in a.teacher.read_text().splitlines() if x.strip()]
    stats = build(recs, a.frames, a.out, a.seed, rounds)
    (a.out / "build.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
