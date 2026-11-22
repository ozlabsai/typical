"""helipad/frames.py: the frame question set, grouped vision rows, and the frame-answers -> video-findings merge.
uv run pytest tests/test_helipad_frames_rows.py -q"""
import json

from PIL import Image

from helipad import frames as fr
from scripts.train_gemma_decision_lora import load_examples, validate_multimodal_examples

SPEC = {
    "wires": ("yes", "no"), "pole": ("yes", "no"), "tall_structure": ("yes", "no"),
    "tallest_height": ("below 1.2 m", "1.2–4.5 m", "4.5–24 m", "above 24 m"),
    "tallest_distance": ("below 10 m", "10–50 m", "50–300 m", "above 300 m"),
    "tallest_type": ("tree", "building", "wall or fence", "pole", "vehicle", "none"),
    "ground": ("paved", "hard earth", "stony", "ploughed", "vegetation", "powdery"),
    "vegetation_belly": ("yes", "no"), "dust": ("light", "medium", "heavy"),
}


def teacher() -> dict:
    """A teacher answer per question: most mass on the first option, a little on the rest and on null."""
    out = {}
    for qid, (_, opts) in fr.FRAME_QUESTIONS.items():
        rest = 0.2 / len(opts)
        out[qid] = {o: (0.8 if i == 0 else 0.0) + rest for i, o in enumerate((*opts, fr.NULL_OPTION))}
        out[qid] = {o: p / sum(out[qid].values()) for o, p in out[qid].items()}
    return out


def test_questions_match_spec_and_every_row_has_null():
    assert fr.QSET_VERSION == "v0" and fr.NULL_OPTION == "None of the options is supported by the image."
    assert {q: opts for q, (_, opts) in fr.FRAME_QUESTIONS.items()} == SPEC
    row = fr.grouped_row("v01/f00", "images/v01_f00.jpg", teacher())
    assert set(row["questions"]) == set(SPEC) and row["meta"]["qset"] == "v0" and row["image"] == "images/v01_f00.jpg"
    for qid, q in row["questions"].items():
        assert q["type"] == "choice" and q["instructions"] == fr.FRAME_QUESTIONS[qid][0]
        assert list(q["criteria"]) == [*(f"option_{i:02d}" for i in range(len(SPEC[qid]))), "null"]
        assert q["criteria"]["null"] == fr.NULL_OPTION and list(q["criteria"].values())[:-1] == list(SPEC[qid])


def test_grouped_row_targets_sum_to_one_and_use_option_keys():
    row = fr.grouped_row("v01/f00", "x.jpg", teacher())
    for qid, soft in row["soft_targets"].items():
        assert abs(sum(soft) - 1) < 1e-9 and len(soft) == len(row["questions"][qid]["criteria"])
        assert row["targets"][qid] == {"label": "option_00"}
    nulls = {q: {fr.NULL_OPTION: 1.0} for q in fr.FRAME_QUESTIONS}
    row = fr.grouped_row("v01/f00", "x.jpg", nulls)
    assert all(t == {"label": "null"} for t in row["targets"].values())
    assert all(s[-1] == 1.0 for s in row["soft_targets"].values())


def test_dropped_question_loses_its_argmax_option_and_targets_null():
    row = fr.grouped_row("v01/f00", "x.jpg", teacher(), drop="ground")
    crit = row["questions"]["ground"]["criteria"]
    assert "paved" not in crit.values() and list(crit)[-1] == "null" and len(crit) == len(SPEC["ground"])
    assert row["targets"]["ground"] == {"label": "null"} and row["soft_targets"]["ground"][-1] == 1.0
    assert row["targets"]["wires"] == {"label": "option_00"}


def test_grouped_row_loads_in_the_gemma_trainer(tmp_path):
    Image.new("RGB", (32, 32), (90, 120, 60)).save(tmp_path / "f.png")
    rows = [fr.grouped_row("v01/f00", "f.png", teacher()), fr.grouped_row("v01/f00~d", "f.png", teacher(), drop="dust")]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    examples = load_examples([tmp_path / "rows.jsonl"])
    validate_multimodal_examples(examples, 1)
    assert len(examples) == 2 * len(SPEC)
    assert all(e["image"] == str((tmp_path / "f.png").resolve()) for e in examples)
    assert all(abs(sum(e["probabilities"]) - 1) < 1e-9 for e in examples)


def frame(t: float, **top) -> dict:
    """A frame's answers: `top` = {qid: (option, p)}; the rest of each question's mass is spread over its other options."""
    out = {}
    for qid in fr.FRAME_QUESTIONS:
        opts = fr.options(qid)
        o, p = top.get(qid, (fr.NULL_OPTION, 0.9))
        out[qid] = {x: (p if x == o else (1 - p) / (len(opts) - 1)) for x in opts}
    return {"t": t, "answers": out}


def test_video_findings_have_no_invented_direction_or_point_height():
    tree = {"tallest_type": ("tree", 0.9), "tallest_height": ("4.5–24 m", 0.7), "tallest_distance": ("10–50 m", 0.8),
            "wires": ("no", 0.9)}
    obs, info = fr.video_findings([frame(0.0, **tree), frame(2.0, **tree)], "v01")
    assert len(obs) == 1
    o = obs[0]
    assert o.bearing is None and o.source == "video" and o.height_src == "estimated" and not o.in_pad
    assert "4.5–24 m" in o.label and "10–50 m" in o.label
    assert 4.5 <= o.height_m < 24 and 10 <= o.dist_center_m < 50
    assert info["video"] == "v01" and info["presence"]["wires"]["p"] < 0.5
    # an unreadable frame (null argmax on the tallest-object questions) adds nothing
    assert fr.video_findings([frame(0.0)], "v02")[0] == []


def test_wire_seen_in_video_becomes_a_wire_obstacle():
    from helipad.case import SiteCase
    from helipad.rules import judge

    def case(obs):
        return SiteCase("S9", 1, obs, [], 2.0, 6.0, None, None, None, True, ())
    seen = [frame(0.0, wires=("no", 0.9)), frame(4.0, wires=("yes", 0.8))]
    obs, info = fr.video_findings(seen, "v01")
    wire = [o for o in obs if o.kind == "wire"]
    assert len(wire) == 1 and wire[0].bearing is None and wire[0].source == "video" and "distance unknown" in wire[0].label
    assert info["presence"]["wires"] == {"p": 0.8, "t": [4.0]}
    assert judge(case(obs))["low_wires"] == "probably_fails"
    unseen, _ = fr.video_findings([frame(0.0, wires=("no", 0.9))], "v01")
    assert judge(case(unseen))["low_wires"] == "not_checked"


def test_video_ground_gives_dust_and_belly_only_when_readable():
    f = frame(0.0, dust=("heavy", 0.8), ground=("ploughed", 0.7), vegetation_belly=("no", 0.9))
    _, info = fr.video_findings([f], "v01")
    assert info["dust"] == "heavy" and info["belly"] == "obstacles"
    _, info = fr.video_findings([frame(0.0)], "v01")
    assert info["dust"] is None and info["belly"] is None
