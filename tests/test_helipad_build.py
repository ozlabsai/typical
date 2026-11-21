import json
import shutil

import pytest

from scripts import helipad_build as hb


def _read(p):
    return [json.loads(line) for line in open(p)]


def test_write_shuffles_skips_empty_and_accepts_outside_root(tmp_path, capsys):
    rows = [{"i": i} for i in range(50)]
    hb._write(tmp_path / "a.jsonl", rows, hb.random.Random(0))
    got = [r["i"] for r in _read(tmp_path / "a.jsonl")]
    assert sorted(got) == list(range(50)) and got != list(range(50))
    hb._write(tmp_path / "empty.jsonl", [], hb.random.Random(0))
    assert not (tmp_path / "empty.jsonl").exists() and "skipped" in capsys.readouterr().out




REGISTRY = [{"slug": "s1", "public_id": "S1", "name": "site one", "expert": "not_approved", "expert_items": {}},
            {"slug": "s2", "public_id": "S2", "name": "site two", "expert": "approved", "expert_items": {}}]


def _site_dirs(sites):
    for slug, osm in (("s1", True), ("s2", False)):   # s2: OSM failed
        (sites / slug).mkdir()
        lz = json.loads(open("tests/fixtures/helipad/lz_min.json").read())
        lz.update(osm_ok=osm, dem_ok=True)
        (sites / slug / "lz.json").write_text(json.dumps(lz))
    (sites / "not_a_site").mkdir()
    shutil.copy(sites / "s1" / "lz.json", sites / "not_a_site" / "lz.json")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out, sites = tmp_path_factory.mktemp("out"), tmp_path_factory.mktemp("sites")
    _site_dirs(sites)
    reg = tmp_path_factory.mktemp("reg") / "sites.json"
    reg.write_text(json.dumps(REGISTRY))
    hb.main(["--n-train", "60", "--n-val", "20", "--n-holdout", "40", "--sites", str(sites), "--out", str(out),
             "--sites-json", str(reg)])
    return out


def test_missing_registry_names_the_env_var(tmp_path):
    _site_dirs(tmp_path)
    with pytest.raises(FileNotFoundError, match="HELIPAD_PRIVATE"):
        hb.main(["--n-train", "4", "--n-val", "2", "--n-holdout", "4", "--sites", str(tmp_path), "--out", str(tmp_path / "o"),
                 "--sites-json", str(tmp_path / "missing.json")])


def test_splits_are_shuffled_and_deduped(built):
    train, val, hold = (_read(built / f) for f in ("train.jsonl", "val.jsonl", "eval/import_oneliner.jsonl"))
    body = lambda rs: {r["state"] for r in rs}
    assert not body(val) & body(train) and not body(hold) & (body(train) | body(val))
    assert [r["meta"]["case"] for r in train[:24]] != sorted(r["meta"]["case"] for r in train[:24])


def test_contrast_and_norule(built):
    contrast = _read(built / "eval/helipad_contrast.jsonl")
    assert any("~" in r["meta"]["case"] for r in contrast)
    norule = _read(built / "eval/helipad_norule.jsonl")
    assert norule and all("Rule:" not in r["query"] for r in norule)


def test_incomplete_sites_and_unknown_slugs_are_excluded(built):
    for f in ("helipad_real", "helipad_expert"):
        names = {r["meta"]["case"] for r in _read(built / f"eval/{f}.jsonl")}
        assert names == {"S1"}   # by public id; s2 (OSM failed) and not_a_site are out


def test_states_carry_no_site_name(built):
    assert not any(r["state"].startswith("Site:") for r in _read(built / "train.jsonl"))
