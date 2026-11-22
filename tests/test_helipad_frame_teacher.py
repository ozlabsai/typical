"""scripts/helipad_frame_teacher.py with a fake runner: no network. uv run pytest tests/test_helipad_frame_teacher.py -q"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import helipad_frame_teacher as ft  # noqa: E402
from helipad import frames as fr  # noqa: E402

Q = fr.FRAME_QUESTIONS


def answer(total: float = 1.0) -> dict:
    return {qid: {o: total / len(fr.options(qid)) for o in fr.options(qid)} for qid in Q}


def test_prompt_lists_every_option_and_null():
    p = ft.prompt(Q)
    for qid, (question, opts) in Q.items():
        assert qid in p and question in p and all(f'"{o}"' in p for o in opts)
    assert fr.NULL_OPTION in p and "use the null option when the image does not show it" in p


def test_teacher_reply_validation():
    assert ft.validate("not json at all", Q) is None
    a = answer()
    del a["dust"]
    assert ft.validate(json.dumps(a), Q) is None
    a = answer()
    a["wires"]["maybe"] = 0.0
    assert ft.validate(json.dumps(a), Q) is None
    got = ft.validate("Here you go:\n```json\n" + json.dumps(answer(0.97)) + "\n```", Q)
    assert got is not None and all(abs(sum(d.values()) - 1) < 1e-9 for d in got.values())
    assert set(got["wires"]) == set(fr.options("wires"))
    assert ft.validate(json.dumps(answer(0.5)), Q) is None
    a = answer()
    a["pole"] = {"yes": 1.2, "no": -0.2}
    assert ft.validate(json.dumps(a), Q) is None


def _frames(tmp_path, n=3):
    out = []
    for i in range(n):
        (tmp_path / f"f{i}.jpg").write_bytes(b"jpg%d" % i)
        out.append({"video": "v01", "t": float(i), "path": f"f{i}.jpg", "sha256": f"{i:064d}"})
    return out


class Runner:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, image, prompt):
        self.calls.append(image)
        return self.replies.pop(0) if self.replies else json.dumps(answer())


def test_cache_key_separates_image_qset_and_teacher():
    keys = {ft.cache_key("a" * 64, "v0", "t1"), ft.cache_key("b" * 64, "v0", "t1"),
            ft.cache_key("a" * 64, "v1", "t1"), ft.cache_key("a" * 64, "v0", "t2")}
    assert len(keys) == 4


def test_cache_makes_reruns_free(tmp_path):
    frames, cache = _frames(tmp_path), tmp_path / "cache"
    first = Runner([])
    recs = ft.label(frames, cache, first, root=tmp_path, teacher="t")
    assert len(first.calls) == 3 and all(not r.get("failed") for r in recs)
    again = Runner([])
    assert ft.label(frames, cache, again, root=tmp_path, teacher="t") == recs and again.calls == []


def test_failed_after_one_retry_is_recorded(tmp_path):
    frames = _frames(tmp_path, 1)
    r = Runner(["nope", "still nope"])
    rec = ft.label(frames, tmp_path / "cache", r, root=tmp_path, teacher="t")[0]
    assert len(r.calls) == 2 and rec["failed"] is True and "answers" not in rec
    ok = Runner(["nope"])   # failures are not cached: a rerun asks again, and one retry is enough here
    rec = ft.label(frames, tmp_path / "cache", ok, root=tmp_path, teacher="t")[0]
    assert len(ok.calls) == 2 and not rec.get("failed") and set(rec["answers"]) == set(Q)
