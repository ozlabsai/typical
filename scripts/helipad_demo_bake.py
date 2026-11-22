"""Bake recorded demo results (helipad-demo/DATA.md) for the real sites into the private data dir.

uv run scripts/helipad_demo_bake.py [--scored .context/helipads/eval/run800_full.json] [--frames frames.json]
Writes $HELIPAD_PRIVATE/demo/{index.json, sites/<public_id>.json, img/}; point the app at it with
HELIPAD_DEMO_DATA=$HELIPAD_PRIVATE/demo. Private: real names, imagery and the expert's calls never go into git.
Satellite crops (Esri World Imagery export) are fetched once and cached.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from helipad.demo import bbox, frame_evidence, pick_frames, site_record  # noqa: E402
from helipad.from_lz import case_from_lz  # noqa: E402
from helipad.paths import PRIVATE, SITES_JSON  # noqa: E402
from scripts.helipad_sites import OUT as SITES_DIR, is_complete  # noqa: E402

ESRI = ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
        "?bbox={b}&bboxSR=4326&imageSR=4326&size={px},{px}&f=image&format=jpg")


def fetch_satellite(lat: float, lon: float, half_m: float, path: Path, px: int = 1024) -> None:
    if path.exists():
        return
    url = ESRI.format(b=",".join(f"{v:.6f}" for v in bbox(lat, lon, half_m)), px=px)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        path.write_bytes(r.read())


def pred_key(scored: dict, site: dict) -> str | None:
    """Scored files key real sites by public id (since the privacy scrub) or, in older runs, by name."""
    return next((k for k in (site["public_id"], site["name"]) if k in scored), None)


def _vision(root: Path, e2e: str = "e2e") -> dict:
    """Plan 2 outputs, or empty when the vision run is absent (every site is then map only). e2e: the folder whose
    score.json holds the decision model's answers on the map + video cases."""
    rows_f, score_f, videos_f = root / "results/eval/frames_tuned.jsonl", root / e2e / "score.json", PRIVATE / "videos.json"
    if not (rows_f.exists() and score_f.exists() and videos_f.exists()):
        return {"cases": {}, "scores": {}, "rows": {}, "root": root}
    from scripts.helipad_vision_eval import site_cases
    rows: dict[str, list] = {}
    for line in open(rows_f):
        r = json.loads(line)
        rows.setdefault(r["video"], []).append(r)
    cases = site_cases(json.loads(videos_f.read_text()), json.loads(SITES_JSON.read_text()), SITES_DIR,
                       {"student": {v: [{"t": r["t"], "answers": r["answers"]} for r in rs] for v, rs in rows.items()}})
    return {"cases": cases, "scores": json.loads(score_f.read_text())["e2e.jsonl"]["cases"], "rows": rows, "root": root}


def _site_frames(pid: str, videos: list[str], vision: dict, out: Path, k: int) -> dict | None:
    picked = []
    for v in videos:
        for r in pick_frames(vision["rows"].get(v, []), max(1, k // len(videos))):
            img = f"img/{pid}_{r['id'].replace('/', '_')}.jpg"
            shutil.copyfile(vision["root"] / "frames" / f"{r['id']}.jpg", out / img)
            picked.append((img, r))
    return frame_evidence(picked) if picked else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default=str(ROOT / ".context/helipads/eval/run800_full.json"))
    ap.add_argument("--vision", default=str(ROOT / ".context/helipads/vision"),
                    help="Plan 2 outputs: frames/, results/eval/frames_tuned.jsonl, e2e/score.json; sites with a matched video use the full system (map + video)")
    ap.add_argument("--e2e", default="e2e", help="folder under --vision with the decision model's map + video scores")
    ap.add_argument("--frames-per-site", type=int, default=6)
    ap.add_argument("--out", default=str(PRIVATE / "demo"))
    ap.add_argument("--half-width", type=float, default=400)
    ap.add_argument("--model-version", default="plan1-800 · vision v0")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "sites").mkdir(parents=True, exist_ok=True)
    (out / "img").mkdir(exist_ok=True)
    scored = json.loads(Path(a.scored).read_text())["helipad_real.jsonl"]["cases"]
    vision = _vision(Path(a.vision), a.e2e)
    index = []
    for s in json.loads(SITES_JSON.read_text()):
        f = SITES_DIR / s["slug"] / "lz.json"
        key = pred_key(scored, s)
        if not f.exists() or key is None:
            print(f"{s['public_id']}: skipped (no complete fetch or no scored prediction)")
            continue
        lz = json.loads(f.read_text())
        if not is_complete(lz):
            print(f"{s['public_id']}: skipped (OSM or DEM missing)")
            continue
        pid = s["public_id"]
        lat, lon = lz["verdict"]["center_used"]
        fetch_satellite(lat, lon, a.half_width, out / "img" / f"{pid}_sat.jpg")
        pred, case, frames = scored[key], case_from_lz(lz, s["name"]), None
        full = vision["cases"].get(pid, {}).get("student")
        if full is not None and f"{pid}+video(student)" in vision["scores"]:
            pred, case = vision["scores"][f"{pid}+video(student)"], full
            frames = _site_frames(pid, vision["cases"][pid]["videos"], vision, out, a.frames_per_site)
        expert = ({"call": s["expert"], "reason": {"en": s.get("expert_reason") or "", "he": s.get("expert_reason") or ""}}
                  if s.get("expert") else None)
        rec = site_record(pid, {"en": s.get("name_en") or pid, "he": s["name"]}, s["round"], expert,
                          case, pred["items"], pred["verdict"], f"img/{pid}_sat.jpg", a.half_width, frames)
        (out / "sites" / f"{pid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        index.append({"id": pid, "round": s["round"], "verdict": rec["verdict"]["status"]})
        print(f"{pid}: {rec['verdict']['status']}{' (map + video)' if frames else ' (map only)'}")
    if not index:
        sys.exit("no site could be baked; index.json left untouched")
    (out / "index.json").write_text(json.dumps({"model": {"name": "typical-medium · helipad", "version": a.model_version},
                                                "sites": index}, ensure_ascii=False, indent=1))
    print(f"{len(index)} sites -> {out}")


if __name__ == "__main__":
    main()
