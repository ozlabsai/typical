"""scripts/helipad_vision_build.py on synthetic frames and teacher answers. uv run pytest tests/test_helipad_vision_build.py -q"""
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import helipad_vision_build as vb  # noqa: E402
from helipad import frames as fr  # noqa: E402

VIDEOS = [f"v{i:02d}" for i in range(1, 9)]


def answers(i: int) -> dict:
    """A confident teacher: option i % n gets .9, the rest share .1 with null."""
    out = {}
    for qid in fr.FRAME_QUESTIONS:
        opts = fr.options(qid)
        top = opts[i % (len(opts) - 1)]
        out[qid] = {o: 0.9 if o == top else 0.1 / (len(opts) - 1) for o in opts}
    return out


@pytest.fixture
def built(tmp_path):
    frames = tmp_path / "frames"
    recs = []
    for vi, v in enumerate(VIDEOS):
        (frames / v).mkdir(parents=True)
        for k in range(3):
            Image.new("RGB", (24, 32), (20 * vi, 10 * k, 90)).save(frames / v / f"f{k:02d}.jpg")
            recs.append({"video": v, "t": float(k), "path": f"{v}/f{k:02d}.jpg", "sha256": f"{vi}{k}".zfill(64),
                         "answers": answers(vi + k)})
    recs.append({"video": "v01", "t": 9.0, "path": "v01/f00.jpg", "sha256": "f" * 64, "failed": True})
    out = tmp_path / "vision_v0"
    stats = vb.build(recs, frames, out, seed=0, rounds={v: ("a" if i < 4 else "b") for i, v in enumerate(VIDEOS)})
    rows = {n: [json.loads(x) for x in (out / f"{n}.jsonl").read_text().splitlines()] for n in vb.FILES}
    return out, rows, stats


def test_no_video_in_two_splits(built):
    out, rows, stats = built
    split = vb.split_by_video([{"video": v} for v in VIDEOS for _ in range(2)], 0)
    vids = {k: {f["video"] for f in fs} for k, fs in split.items()}
    assert len(vids["test"]) == 3 and len(vids["dev"]) == 2 and len(vids["train"]) == 3
    assert not (vids["train"] & vids["dev"]) and not (vids["train"] & vids["test"]) and not (vids["dev"] & vids["test"])
    by_file = {n: {r["meta"]["video"] for r in rs} for n, rs in rows.items()}
    test = by_file["test_real"]
    assert not (by_file["train"] & test) and not (by_file["dev"] & test) and not (by_file["train"] & by_file["dev"])
    assert all(by_file[n] == test for n in ("test_shuffled", "test_noimage", "test_mismatch"))
    for n, rs in rows.items():   # the image a row shows comes from its own split too
        assert {r["meta"].get("image_video", r["meta"]["video"]) for r in rs} <= by_file[n]
    assert stats["skipped_failed"] == 1


def test_one_test_video_per_round_when_rounds_are_known():
    rounds = {v: ("a" if i < 4 else "b") for i, v in enumerate(VIDEOS)}
    for seed in range(5):
        test = {f["video"] for f in vb.split_by_video([{"video": v} for v in VIDEOS], seed, rounds)["test"]}
        assert {rounds[v] for v in test} == {"a", "b"}


def test_mismatch_rows_never_reuse_the_same_video(built):
    _, rows, _ = built
    mism = [r for n in ("train", "dev", "test_mismatch") for r in rows[n] if r["meta"]["variant"] == "mismatch"]
    assert mism and all(r["meta"]["image_video"] != r["meta"]["video"] for r in mism)
    assert all(t == {"label": "null"} for r in mism for t in r["targets"].values())
    shuffled = rows["test_shuffled"]
    assert all(r["meta"]["image_video"] != r["meta"]["video"] for r in shuffled)
    assert [r["targets"] for r in shuffled] == [r["targets"] for r in rows["test_real"]]


def test_gold_dropped_targets_null_only_for_the_dropped_question(built):
    _, rows, _ = built
    dropped = [r for r in rows["train"] if r["meta"]["variant"] == "gold_dropped"]
    real = {r["id"]: r for r in rows["train"] if r["meta"]["variant"] == "real"}
    assert dropped
    for r in dropped:
        q = r["meta"]["dropped"]
        assert r["targets"][q] == {"label": "null"} and len(r["questions"][q]["criteria"]) == len(fr.options(q)) - 1
        base = real[r["id"].split("~")[0]]
        assert all(r["targets"][o] == base["targets"][o] for o in r["targets"] if o != q)


def test_noimage_rows_have_no_image_field(built):
    out, rows, _ = built
    assert all("image" not in r for r in rows["test_noimage"])
    assert [r["id"] for r in rows["test_noimage"]] == [r["id"] for r in rows["test_real"]]
    for n in ("train", "dev", "test_real", "test_shuffled", "test_mismatch"):
        assert all((out / r["image"]).is_file() for r in rows[n])
