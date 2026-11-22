"""Which registry site (public id) is each video? Private output, confirmed by the user, never auto-edited after a first run.

HELIPAD_PRIVATE=... uv run scripts/helipad_match_videos.py --frames <frames_dir>/frames.json --brief <BRIEF.md>
    [--deck <deck text>] [--raw <video dir>] [--out $HELIPAD_PRIVATE/videos.json]
For each video the teacher (scripts/helipad_frame_teacher.py's runner) sees a contact sheet of its keyframes plus the private
site descriptions, and names a site or null. Writes {video: {site: public_id | null, confidence: high|low|none, evidence}};
videos already in the file are kept as they are (edit it by hand to correct a match).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from helipad.paths import PRIVATE, SITES_JSON, require  # noqa: E402

CELL_W = 256
CONFIDENCE = ("high", "low", "none")


def contact_sheet(frames: list[Path], out: Path, cols: int = 6) -> Path:
    """Tiles in reading order; the cell aspect is the first frame's."""
    with Image.open(frames[0]) as im:
        ch = round(CELL_W * im.height / im.width)
    sheet = Image.new("RGB", (cols * CELL_W, -(-len(frames) // cols) * ch))
    for i, f in enumerate(frames):
        with Image.open(f) as im:
            sheet.paste(im.convert("RGB").resize((CELL_W, ch)), ((i % cols) * CELL_W, (i // cols) * ch))
    sheet.save(out)
    return out


def parse_mapping(reply: str, ids: list[str]) -> dict:
    start = reply.find("{")
    try:
        obj = json.JSONDecoder().raw_decode(reply[start:])[0] if start >= 0 else None
        if (isinstance(obj, dict) and (obj.get("site") is None or obj["site"] in ids)
                and obj.get("confidence") in CONFIDENCE):
            return {"site": obj["site"], "confidence": obj["confidence"], "evidence": str(obj.get("evidence", ""))}
    except ValueError:
        pass
    return {"site": None, "confidence": "none", "evidence": f"unparsed reply: {reply[:200]}"}


def match(videos: dict[str, list], out: Path, ids: list[str], ask) -> dict:
    """ask(video) -> teacher reply. Videos already in `out` are never asked again or changed."""
    mapping = json.loads(out.read_text()) if out.exists() else {}
    for v in videos:
        if v not in mapping:
            mapping[v] = parse_mapping(ask(v), ids)
            out.write_text(json.dumps(mapping, indent=1, ensure_ascii=False))
    return mapping


def site_prompt(sites: list[dict], brief: str, deck: str, video: str, seconds: float) -> str:
    lines = [f"The image is a contact sheet of {video}'s keyframes, in time order: a phone panorama filmed at a candidate "
             f"helicopter landing site ({seconds:.0f} s; files named v* came with the site rounds, wa_* are WhatsApp videos "
             "dated 2026-11-16 that may show none of the listed sites).",
             "Which of these registry sites is it, if any? Sites (public id, name, round, coordinate, expert reason):"]
    lines += [f"- {s['public_id']}: {s['name']} ({s['round']}) {s.get('lat')}, {s.get('lon')} {s.get('expert_reason') or ''}"
              for s in sites]
    lines += ["", "Brief:", brief, ""]
    if deck:
        lines += ["October deck text (Hebrew, extracted from a PDF; per-site findings say what each site's video shows):",
                  deck, ""]
    lines.append('Reply with one JSON object and nothing else: {"site": "<public id>" or null, "confidence": "high" | "low" '
                 '| "none", "evidence": "<what in the frames matches or rules out a site>"}')
    return "\n".join(lines)


def duration(video: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                         capture_output=True, text=True).stdout
    return float(out or 0)


def main(argv: list[str] | None = None) -> None:
    from helipad_frame_teacher import TEACHER, OpenRouterRunner
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--brief", type=Path, required=True)
    ap.add_argument("--deck", type=Path)
    ap.add_argument("--raw", type=Path, help="video dir, for durations")
    ap.add_argument("--out", type=Path, default=PRIVATE / "videos.json")
    a = ap.parse_args(argv)
    sites = json.loads(require(SITES_JSON).read_text())
    videos: dict[str, list[Path]] = {}
    for f in json.loads(a.frames.read_text()):
        videos.setdefault(f["video"], []).append(a.frames.parent / f["path"])
    sheets = a.out.parent / "contact_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    runner = OpenRouterRunner(TEACHER)
    brief, deck = a.brief.read_text(), a.deck.read_text() if a.deck else ""

    def ask(v: str) -> str:
        secs = duration(a.raw / f"{v}.mp4") if a.raw else 0.0
        return runner(contact_sheet(videos[v], sheets / f"{v}.jpg"), site_prompt(sites, brief, deck, v, secs))

    mapping = match(videos, a.out, [s["public_id"] for s in sites], ask)
    for v, m in mapping.items():
        print(f"{v:8} {str(m['site']):5} {m['confidence']:5} {m['evidence'][:160]}")
    print(json.dumps({"wrote": str(a.out), **runner.spent}))


if __name__ == "__main__":
    main()
