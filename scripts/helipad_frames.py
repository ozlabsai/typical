"""Keyframes from landing-site videos: ffmpeg decodes at --fps, then greedy farthest-point selection on colour histograms.

uv run scripts/helipad_frames.py --videos <dir of .mp4> --out <frames_dir> [--fps 2 --k 16]
Writes <frames_dir>/<video_stem>/fNN.jpg (long side 896 px) and <frames_dir>/frames.json = [{video, t, path, sha256}].
A video whose file sha256 equals an earlier one is skipped (and printed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

LONG_SIDE = 896


def _l1(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def select_keyframes(hists: list[list[float]], k: int) -> list[int]:
    """Greedy farthest-point on L1 histogram distance, from the frame nearest the elementwise median. Ties (a static or
    black video) go to the frame farthest in time from those picked, so the picks still spread over the video."""
    n = len(hists)
    if not n or k <= 0:
        return []
    median = [sorted(col)[n // 2] for col in zip(*hists)]
    picked = [min(range(n), key=lambda i: _l1(hists[i], median))]
    while len(picked) < min(k, n):
        rest = [i for i in range(n) if i not in picked]
        picked.append(max(rest, key=lambda i: (min(_l1(hists[i], hists[j]) for j in picked),
                                               min(abs(i - j) for j in picked))))
    return sorted(picked)


def histogram(img: Image.Image) -> list[float]:
    """8x8x8 RGB histogram, L1-normalised."""
    px = img.convert("RGB").resize((64, 64)).tobytes()
    h = [0.0] * 512
    for r, g, b in zip(px[0::3], px[1::3], px[2::3]):
        h[(r >> 5) * 64 + (g >> 5) * 8 + (b >> 5)] += 1
    total = sum(h)
    return [x / total for x in h]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(video: Path, out_dir: Path, fps: float, k: int) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps}", f"{tmp}/%05d.png"], check=True)
        decoded = sorted(Path(tmp).glob("*.png"))
        hists = []
        for p in decoded:
            with Image.open(p) as im:
                hists.append(histogram(im))
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for n, i in enumerate(select_keyframes(hists, k)):
            with Image.open(decoded[i]) as im:
                im = im.convert("RGB")
                im.thumbnail((LONG_SIDE, LONG_SIDE))
                dest = out_dir / f"f{n:02d}.jpg"
                im.save(dest, quality=92)
            rows.append({"video": video.stem, "t": round(i / fps, 2), "path": f"{video.stem}/{dest.name}",
                         "sha256": sha256(dest)})
        return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=2)
    ap.add_argument("--k", type=int, default=16)
    a = ap.parse_args(argv)
    seen: dict[str, str] = {}
    frames = []
    for video in sorted(a.videos.glob("*.mp4")):
        digest = sha256(video)
        if digest in seen:
            print(f"skip {video.stem}: identical to {seen[digest]}")
            continue
        seen[digest] = video.stem
        rows = extract(video, a.out / video.stem, a.fps, a.k)
        print(f"{video.stem}: {len(rows)} keyframes")
        frames += rows
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "frames.json").write_text(json.dumps(frames, indent=1))
    print(f"{len(frames)} keyframes from {len(seen)} videos -> {a.out / 'frames.json'}")


if __name__ == "__main__":
    main()
