import hashlib
import json

import pytest

from scripts.build_r045_compositional_transfer import _pair
from scripts.build_r047_clause_arms import build
from scripts.preflight_r047 import audit


def _fixture(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    rows = _pair("train", ("decimal", "policy"), 72, seed=45, case_index=0)
    rows.extend(_pair("train", ("fraction",), 12, seed=45, case_index=0))
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    source = source_dir / "arm-b.jsonl"
    source.write_bytes(payload)
    out = tmp_path / "r047"
    digest = hashlib.sha256(payload).hexdigest()
    build(source, out, expected_input_sha256=digest)
    return source, out, digest


def test_preflight_validates_hashes_gold_alignment_and_matched_token_budget(tmp_path):
    source, out, digest = _fixture(tmp_path)
    report = audit(
        source, out / "control.jsonl", out / "clause.jsonl", out / "m0-treatment.jsonl",
        out / "manifest.json",
        token_counter=lambda _: 100, tokenizer_name="stub-tokenizer",
        expected_input_sha256=digest)
    assert report["passed"]
    assert report["control"]["auxiliary_views"] == 8
    assert report["clause"]["auxiliary_views"] == 8
    assert report["token_budget"] == {
        "tokenizer": "stub-tokenizer",
        "control_tokens": 1600,
        "clause_tokens": 1600,
        "absolute_delta": 0,
        "relative_delta": 0,
        "limit": .005,
        "within_limit": True,
    }


def test_preflight_rejects_token_budget_mismatch(tmp_path):
    source, out, digest = _fixture(tmp_path)

    def imbalanced(row):
        instruction = next(iter(row["questions"].values())).get("instructions", "")
        return 1_000 if "whole-case" in instruction else 100

    with pytest.raises(ValueError, match="encoded token budget differs"):
        audit(source, out / "control.jsonl", out / "clause.jsonl",
              out / "m0-treatment.jsonl", out / "manifest.json",
              token_counter=imbalanced, tokenizer_name="biased-stub",
              expected_input_sha256=digest)


def test_preflight_fails_closed_on_output_or_manifest_tampering(tmp_path):
    source, out, digest = _fixture(tmp_path)
    rows = [json.loads(line) for line in (out / "clause.jsonl").read_text().splitlines()]
    auxiliary = rows[0]["auxiliary_views"][0]
    qid = next(iter(auxiliary["targets"]))
    auxiliary["targets"][qid]["label"] = (
        "no" if auxiliary["targets"][qid]["label"] == "yes" else "yes")
    (out / "clause.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    with pytest.raises(ValueError, match="hash mismatch"):
        audit(source, out / "control.jsonl", out / "clause.jsonl",
              out / "m0-treatment.jsonl", out / "manifest.json",
              expected_input_sha256=digest)


def test_preflight_rejects_forbidden_manifest_provenance(tmp_path):
    source, out, digest = _fixture(tmp_path)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["provenance"] = "derived from JevBench labels"
    bad = out / "bad-manifest.json"
    bad.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="prohibited benchmark source"):
        audit(source, out / "control.jsonl", out / "clause.jsonl",
              out / "m0-treatment.jsonl", bad,
              expected_input_sha256=digest)
