import json
from pathlib import Path

from PIL import Image

from pcdm import train
from scripts import pilot_corpora as corpora


def test_pilot_corpora_write_visual_and_hebrew_buckets(tmp_path, monkeypatch):
    labels = [f"intent_{i}" for i in range(12)]

    def fake_dataset(name, *args, **kwargs):
        if name == "json":
            return [{"text": f"בקשה בעברית {i}", "label_text": label} for i, label in enumerate(labels)]
        return [{"image": Image.new("RGB", (8, 8), "red"), "question": "Which color?",
                 "choices": ["red", "blue", "green"], "answer": 0, "hint": "", "task": "closed choice"}]

    monkeypatch.setattr(corpora, "load_dataset", fake_dataset)
    he = tmp_path / "he"
    counts = corpora.build_hebrew(he, limit=2)
    assert counts == {"train": 2, "validation": 2, "test": 2}
    first = json.loads((he / "train.jsonl").read_text().splitlines()[0])
    assert first["meta"]["fam_bucket"] == "H"
    assert any(word in first["query"] for word in ("מה", "איזו"))

    visual = tmp_path / "visual"
    counts = corpora.build_scienceqa(visual, null_fraction=1.0)
    assert counts == {"train": 2, "validation": 1, "test": 1}
    assert (visual / "images" / "train" / "0.png").exists()
    assert '"fam_bucket": "V"' in (visual / "train.jsonl").read_text()
    assert '"image"' not in (visual / "eval" / "scienceqa_test_noimage.jsonl").read_text()
    assert Path(train.load_jsonl(visual / "eval" / "scienceqa_test.jsonl")[0]["image"]).exists()
