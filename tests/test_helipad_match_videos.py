"""scripts/helipad_match_videos.py: contact sheets and the private video -> site mapping. No network.
uv run pytest tests/test_helipad_match_videos.py -q"""
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import helipad_match_videos as mv  # noqa: E402

IDS = ["S1", "S2", "O1"]


def test_contact_sheet_tiles_all_frames(tmp_path):
    paths = []
    for i in range(7):
        paths.append(tmp_path / f"{i}.png")
        Image.new("RGB", (30, 40), (30 * i, 0, 0)).save(paths[-1])
    out = tmp_path / "sheet.png"
    mv.contact_sheet(paths, out, cols=6)
    with Image.open(out) as im:
        cw, ch = mv.CELL_W, round(mv.CELL_W * 40 / 30)
        assert im.size == (6 * cw, 2 * ch)
        assert im.getpixel((cw + cw // 2, ch // 2))[0] > 20   # the second tile is drawn
        assert im.getpixel((cw + cw // 2, ch + ch // 2)) == (0, 0, 0)   # 7 frames: the rest of row 2 is blank


def test_mapping_schema():
    got = mv.parse_mapping('Sure. {"site": "O1", "confidence": "high", "evidence": "lattice pole"}', IDS)
    assert got == {"site": "O1", "confidence": "high", "evidence": "lattice pole"}
    assert mv.parse_mapping('{"site": null, "confidence": "none", "evidence": "new site"}', IDS)["site"] is None
    for bad in ("no json", '{"site": "S9", "confidence": "high", "evidence": ""}',
                '{"site": "S1", "confidence": "sure", "evidence": ""}'):
        assert mv.parse_mapping(bad, IDS) == {"site": None, "confidence": "none", "evidence": f"unparsed reply: {bad[:200]}"}


def test_existing_mapping_is_never_overwritten(tmp_path):
    out = tmp_path / "videos.json"
    out.write_text(json.dumps({"v01": {"site": "S1", "confidence": "high", "evidence": "user confirmed"}}))
    calls = []
    mapping = mv.match({"v01": [], "v02": []}, out, IDS, lambda v: calls.append(v) or
                       '{"site": "S2", "confidence": "low", "evidence": "x"}')
    assert calls == ["v02"] and mapping["v01"]["evidence"] == "user confirmed" and mapping["v02"]["site"] == "S2"
    assert json.loads(out.read_text()) == mapping
