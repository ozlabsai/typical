"""scripts/helipad_frames.py: keyframe selection and extraction. uv run pytest tests/test_helipad_frames.py -q"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import helipad_frames as hf  # noqa: E402


def _hist(i: int) -> list[float]:
    h = [0.0] * 8
    h[i] = 1.0
    return h


def test_select_keyframes_spreads_over_distinct_scenes():
    hists = [_hist(0)] * 5 + [_hist(3)] * 5 + [_hist(6)] * 5
    picked = hf.select_keyframes(hists, 3)
    assert sorted({i // 5 for i in picked}) == [0, 1, 2] and picked == sorted(picked)


def test_select_keyframes_short_or_static_video():
    assert hf.select_keyframes([_hist(0), _hist(1)], 16) == [0, 1]
    picked = hf.select_keyframes([_hist(2)] * 10, 4)
    assert len(set(picked)) == 4 and picked == sorted(picked) and all(0 <= i < 10 for i in picked)
    assert hf.select_keyframes([], 4) == []


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_duplicate_videos_are_skipped(tmp_path, capsys):
    videos = tmp_path / "videos"
    videos.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=3:size=160x120:rate=10",
                    "-pix_fmt", "yuv420p", str(videos / "a.mp4")], check=True)
    shutil.copy(videos / "a.mp4", videos / "b.mp4")
    out = tmp_path / "frames"
    hf.main(["--videos", str(videos), "--out", str(out), "--fps", "2", "--k", "4"])
    frames = json.loads((out / "frames.json").read_text())
    assert {f["video"] for f in frames} == {"a"} and len(frames) == 4
    assert "b" in capsys.readouterr().out
    for f in frames:
        assert (out / f["path"]).is_file() and len(f["sha256"]) == 64 and f["t"] >= 0
