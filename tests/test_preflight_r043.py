import hashlib
import json

import pytest

import scripts.preflight_r043 as module


def test_r043_preflight_checks_trainer_visible_counts(monkeypatch, tmp_path):
    outputs = {}
    for name in module.EXPECTED:
        data = b"{}\n"
        (tmp_path / f"{name}.jsonl").write_bytes(data)
        outputs[name] = {"rows": 1, "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {"experiment": "r043_causal_pairs", "seed": 43,
                "logical_rows": {name: values[0] for name, values in module.EXPECTED.items()},
                "trainer_examples": {name: values[0] for name, values in module.EXPECTED.items()},
                "presentations": {name: values[1] for name, values in module.EXPECTED.items()},
                "pair_modes": {"arm-a": "counterfactual", "arm-b": "counterfactual",
                               "arm-c": "shuffled_counterfactual",
                               "arm-d": "shuffled_counterfactual", "eval": "counterfactual"},
                "outputs": outputs}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    def loaded(paths):
        name = paths[0].stem
        return ([{"paired_views": [{}, {}]}] * (256 if name == "eval" else 1200) +
                ([] if name == "eval" else [{}] * 1600))

    monkeypatch.setattr(module, "load_examples", loaded)
    assert module.preflight(tmp_path)["arm-a"]["presentations"] == 4000
    monkeypatch.setattr(module, "load_examples", lambda paths: loaded(paths)[:-1]
                        if paths[0].stem == "arm-a" else loaded(paths))
    with pytest.raises(ValueError, match="trainer sees"):
        module.preflight(tmp_path)
