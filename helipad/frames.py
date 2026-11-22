"""Frame questions a vision Typical answers about one keyframe of a landing-site video, as R044/R056 grouped image rows.

Every question also offers NULL_OPTION ("can't tell"): not seen never means absent. Spec: docs/superpowers/specs/
2026-11-21-helipad-lz-decision-model-design.md §6.
"""
from __future__ import annotations

QSET_VERSION = "v0"
NULL_OPTION = "None of the options is supported by the image."
STATE = "A ground photo taken at a candidate helicopter landing site."
YES_NO = ("yes", "no")
FRAME_QUESTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "wires": ("Are overhead wires or cables visible?", YES_NO),
    "pole": ("Is a power or lighting pole visible?", YES_NO),
    "tall_structure": ("Is a tall mast, tower or crane visible on the horizon?", YES_NO),
    "tallest_height": ("Height of the tallest object in view?", ("below 1.2 m", "1.2–4.5 m", "4.5–24 m", "above 24 m")),
    "tallest_distance": ("Distance from the camera to that object?", ("below 10 m", "10–50 m", "50–300 m", "above 300 m")),
    "tallest_type": ("What is the tallest object in view?", ("tree", "building", "wall or fence", "pole", "vehicle", "none")),
    "ground": ("Ground surface in the foreground?", ("paved", "hard earth", "stony", "ploughed", "vegetation", "powdery")),
    "vegetation_belly": ("Is foreground vegetation taller than 10 cm?", YES_NO),
    "dust": ("Likely dust if a helicopter lands here?", ("light", "medium", "heavy")),
}


def options(qid: str) -> tuple[str, ...]:
    """Every option of a question, null last."""
    return (*FRAME_QUESTIONS[qid][1], NULL_OPTION)


def argmax(dist: dict[str, float]) -> str:
    return max(dist, key=dist.__getitem__)


def grouped_row(frame_id: str, image_rel: str | None, targets: dict[str, dict[str, float]] | None,
                drop: str | None = None, **meta) -> dict:
    """One keyframe, every frame question. targets: {qid: {option text: p}} (teacher answers; keys from options()).
    drop: one qid whose target-argmax option is removed, so its target is null (R044's gold-dropped row).
    Soft targets are r056_build's field: soft_targets[qid] = probabilities in criteria order."""
    row = {"id": frame_id, "state": STATE, "questions": {}, "meta": {"qset": QSET_VERSION, **meta}}
    if targets is not None:
        row["targets"], row["soft_targets"] = {}, {}
    for qid, (question, _) in FRAME_QUESTIONS.items():
        opts = list(options(qid))
        if qid == drop:
            opts.remove(argmax(targets[qid]))
        keys = [f"option_{i:02d}" for i in range(len(opts) - 1)] + ["null"]
        row["questions"][qid] = {"type": "choice", "instructions": question, "criteria": dict(zip(keys, opts))}
        if targets is None:
            continue
        probs = [0.0] * (len(opts) - 1) + [1.0] if qid == drop else [float(targets[qid].get(o, 0.0)) for o in opts]
        total = sum(probs)
        probs = [p / total for p in probs]
        row["soft_targets"][qid] = probs
        row["targets"][qid] = {"label": keys[max(range(len(probs)), key=probs.__getitem__)]}
    if image_rel is not None:
        row["image"] = image_rel
    return row


# -- frame answers -> video findings (spec §6 merge rules)
HEIGHT_M = {"below 1.2 m": (0.0, 1.2), "1.2–4.5 m": (1.2, 4.5), "4.5–24 m": (4.5, 24.0), "above 24 m": (24.0, 76.0)}
DIST_M = {"below 10 m": (0.0, 10.0), "10–50 m": (10.0, 50.0), "50–300 m": (50.0, 300.0), "above 300 m": (300.0, 1000.0)}
KIND = {"tree": "tree", "building": "building", "wall or fence": "wall", "pole": "pole"}   # vehicles move: no finding
SEEN = 0.5   # presence and readability threshold
DUST = ("light", "medium", "heavy")


def _range(frames: list[dict], qid: str, spans: dict) -> tuple[str, float]:
    """The argmax class over the frames' summed answers, and the midpoint of the confidence-weighted class range, kept
    inside that class so the rules and the label agree."""
    mass = {c: sum(f["answers"][qid].get(c, 0.0) for f in frames) for c in spans}
    top = argmax(mass)
    w = {c: m / sum(mass.values()) for c, m in mass.items()}
    mid = sum(w[c] * (spans[c][0] + spans[c][1]) / 2 for c in spans)
    lo, hi = spans[top]
    return top, round(min(max(mid, lo), hi - 0.1), 1)


def video_findings(frame_preds: list[dict], video_stem: str) -> tuple[list, dict]:
    """frame_preds: [{"t": seconds, "answers": {qid: {option: p}}}] for one video -> (Obstacles, info).
    Obstacles carry source="video", bearing=None (no compass) and height_src="estimated"/"unknown"; the camera stands for
    the touchdown point. A class becomes its range in the label; the number the rules read is kept inside that range.
    Presence (wires, pole, tall structure) = the max P(yes) over frames, with the timestamps of frames at P >= SEEN."""
    from helipad.case import PAD_HALF, Obstacle

    def p_yes(f, q):
        return f["answers"][q].get("yes", 0.0)
    presence = {q: {"p": round(max((p_yes(f, q) for f in frame_preds), default=0.0), 4),
                    "t": [f["t"] for f in frame_preds if p_yes(f, q) >= SEEN]} for q in ("wires", "pole", "tall_structure")}
    obs = []
    if presence["wires"]["p"] >= SEEN:   # no frame question measures a wire: unknown height and distance, the worst ring
        obs.append(Obstacle("wire", "overhead wire (video; height and distance unknown, treated as within 50 m)",
                            8.0, "unknown", 25.0, False, None, "video", 25.0 + PAD_HALF))
    groups: dict[tuple, list] = {}
    for f in frame_preds:
        key = tuple(argmax(f["answers"][q]) for q in ("tallest_type", "tallest_height", "tallest_distance"))
        if key[0] in KIND and key[1] in HEIGHT_M and key[2] in DIST_M:
            groups.setdefault(key, []).append(f)
    for (kind, _, _), fs in groups.items():
        hc, h = _range(fs, "tallest_height", HEIGHT_M)
        dc, d = _range(fs, "tallest_distance", DIST_M)
        obs.append(Obstacle(KIND[kind], f"{kind} (video; height {hc}, {dc} from the camera)", h, "estimated",
                            round(max(0.0, d - PAD_HALF), 1), False, None, "video", d))
    n = max(1, len(frame_preds))
    mean = lambda q, opts: {c: sum(f["answers"][q].get(c, 0.0) for f in frame_preds) / n for c in opts}   # noqa: E731
    dust, ground = mean("dust", DUST), mean("ground", options("ground"))
    veg = mean("vegetation_belly", YES_NO)
    if max((f["answers"]["vegetation_belly"].get("yes", 0.0) for f in frame_preds), default=0.0) >= SEEN \
            or argmax(ground) == "ploughed":
        belly = "obstacles"
    elif veg["no"] >= SEEN:
        belly = "clear"
    else:
        belly = None
    info = {"video": video_stem, "frames": len(frame_preds), "presence": presence, "belly": belly,
            "dust": argmax(dust) if sum(dust.values()) >= SEEN else None, "dust_dist": dust, "ground_dist": ground}
    return obs, info
