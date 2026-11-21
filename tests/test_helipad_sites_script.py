
import json
import shutil
import subprocess
from dataclasses import replace

import pytest

import scripts.helipad_sites as hs
from scripts.helipad_sites import agreement


def test_agreement_reads_lz_json(tmp_path):
    d = tmp_path / "s1"
    d.mkdir()
    shutil.copy("tests/fixtures/helipad/lz_min.json", d / "lz.json")
    rows = agreement(tmp_path)
    assert rows == [{"slug": "s1", "lz": "fails", "rules": "fails", "visible": "fails", "match": True,
                     "osm": True, "dem": True, "govmap": True, "corridors": False}]


def test_check_compares_the_visible_case(tmp_path, monkeypatch, capsys):   # I8: the rendered case is the real gold
    d = tmp_path / "s1"
    d.mkdir()
    shutil.copy("tests/fixtures/helipad/lz_min.json", d / "lz.json")
    monkeypatch.setattr(hs, "visible", lambda c: replace(c, obstacles=[], canopy=None, slope_deg=0.0, zones=[]))
    monkeypatch.setattr(hs, "OUT", tmp_path)
    monkeypatch.setattr("sys.argv", ["helipad_sites.py", "check"])
    row = agreement(tmp_path)[0]
    assert row["rules"] == "fails" != row["visible"] and not row["match"]
    assert hs.main() == 1 and "visible=" in capsys.readouterr().out


def test_incomplete_fetch_is_not_cached():
    from scripts.helipad_sites import is_complete
    assert is_complete({"osm_ok": True, "dem_ok": True})
    assert not is_complete({"osm_ok": False, "dem_ok": True})


def _sites(tmp_path, monkeypatch, *slugs):
    f = tmp_path / "sites.json"
    f.write_text(json.dumps([{"slug": s, "lat": 31.5, "lon": 34.8, "link": None} for s in slugs]))
    monkeypatch.setattr(hs, "SITES", f)
    monkeypatch.setattr(hs, "OUT", tmp_path / "out")
    (tmp_path / "lz_check.py").touch()
    monkeypatch.setattr(hs, "LZ", tmp_path / "lz_check.py")


def test_fetch_without_lz_check_names_the_env_var(tmp_path, monkeypatch):
    _sites(tmp_path, monkeypatch, "a")
    monkeypatch.setattr(hs, "LZ", tmp_path / "missing" / "lz_check.py")
    with pytest.raises(FileNotFoundError, match="HELIPAD_PRIVATE"):
        hs.fetch()


def test_fetch_times_out_per_site_and_continues(tmp_path, monkeypatch, capsys):
    _sites(tmp_path, monkeypatch, "a", "b")
    timeouts = []

    def run(cmd, **kw):
        timeouts.append(kw["timeout"])
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(subprocess, "run", run)
    hs.fetch()
    out = capsys.readouterr().out
    assert timeouts == [900, 900] and "a: FAILED" in out and "b: FAILED" in out


def test_fetch_runs_lz_full_after_success(tmp_path, monkeypatch):
    _sites(tmp_path, monkeypatch, "a")
    v = json.loads(open("tests/fixtures/helipad/lz_min.json").read())["verdict"]
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        if str(hs.LZ) in cmd:
            d = tmp_path / "out" / "a"
            d.mkdir(parents=True, exist_ok=True)
            (d / "verdict.json").write_text(json.dumps(v))
            (d / "data.json").write_text(json.dumps({"osm": {"elements": []}}))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", run)
    hs.fetch()
    assert [c[-2:] for c in calls[1:]] == [[str(hs.FULL), str(tmp_path / "out" / "a")]]


def test_refresh_runs_lz_full_on_dirs_with_lz_json_only(tmp_path, monkeypatch):
    (tmp_path / "done").mkdir()
    (tmp_path / "done" / "lz.json").write_text("{}")
    (tmp_path / "partial").mkdir()
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    hs.refresh(tmp_path)
    assert [c[-1] for c in calls] == [str(tmp_path / "done")]
