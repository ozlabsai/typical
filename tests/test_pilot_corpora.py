import json
from pathlib import Path

import numpy as np
from PIL import Image

from pcdm import train
from scripts import pilot_corpora as corpora
from scripts.eval_language_pairs import paired_metrics


def test_pilot_corpora_write_visual_and_hebrew_buckets(tmp_path, monkeypatch):
    labels = [f"intent_{i}" for i in range(12)]

    def fake_dataset(name, *args, **kwargs):
        if name == "json":
            lang = "en" if "/en.json" in kwargs["data_files"] else "he"
            return [{"id": str(i), "text": f"request {i}" if lang == "en" else f"בקשה בעברית {i}",
                     "label_text": label} for i, label in enumerate(labels)]
        return [{"image": Image.new("RGB", (8, 8), "red"), "question": "Which color?",
                 "choices": ["red", "blue", "green"], "answer": 0, "hint": "", "task": "closed choice"}]

    monkeypatch.setattr(corpora, "load_dataset", fake_dataset)
    he = tmp_path / "he"
    counts = corpora.build_hebrew(he, limit=2)
    assert counts == {"train": 2, "validation": 2, "test": 2}
    first = json.loads((he / "train.jsonl").read_text().splitlines()[0])
    assert first["meta"]["fam_bucket"] == "H"
    assert any(word in first["query"] for word in ("מה", "איזו"))
    he_pair = train.load_jsonl(he / "eval" / "massive_he_test.jsonl")[0]
    en_pair = train.load_jsonl(he / "eval" / "massive_en_test_paired.jsonl")[0]
    assert he_pair["meta"]["pair"] == en_pair["meta"]["pair"]
    assert he_pair["candidates"] == en_pair["candidates"] and en_pair["meta"]["lang"] == "en"

    visual = tmp_path / "visual"
    counts = corpora.build_scienceqa(visual, null_fraction=1.0)
    assert counts == {"train": 2, "validation": 1, "test": 1}
    assert (visual / "images" / "train" / "0.png").exists()
    assert '"fam_bucket": "V"' in (visual / "train.jsonl").read_text()
    assert '"image"' not in (visual / "eval" / "scienceqa_test_noimage.jsonl").read_text()
    assert Path(train.load_jsonl(visual / "eval" / "scienceqa_test.jsonl")[0]["image"]).exists()
    controls = train.load_jsonl(visual / "eval" / "scienceqa_test_answerability.jsonl")
    assert [row["meta"]["variant"] for row in controls] == ["base", "image_mismatch", "remove_gold"]
    assert [row["p_null"] for row in controls] == [0.0, 1.0, 1.0]


def test_paired_language_metrics():
    left = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]
    right = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]
    examples = [{"candidates": ["a", "b"], "label": label} for label in (0, 1)]
    metrics = paired_metrics(left=np.array(left), right=np.array(right),
                             left_examples=examples, right_examples=examples)
    assert metrics["prediction_agreement"] == metrics["both_correct"] == 1.0
    assert 0 < metrics["mean_jsd"] < 0.1
