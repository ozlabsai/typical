import json, re

import pytest

from helipad.paths import SITES_JSON

pytestmark = pytest.mark.skipif(not SITES_JSON.exists(), reason=f"private site registry not found at {SITES_JSON}; set HELIPAD_PRIVATE")


def test_sites_registry():
    sites = json.loads(SITES_JSON.read_text())
    assert len(sites) == 16
    assert len({s["slug"] for s in sites}) == 16
    assert all(re.fullmatch(r"[a-z0-9_]+", s["slug"]) for s in sites)
    assert [s["public_id"] for s in sites] == [f"S{i}" for i in range(1, 7)] + [f"O{i}" for i in range(1, 11)]
    sept = [s for s in sites if s["round"] == "2026-09"]
    assert sorted(s["expert"] for s in sept) == ["approved"] * 3 + ["not_approved"] * 3
    assert all(s["expert"] is None for s in sites if s["round"] == "2026-10")
    assert all(s["link"] or (s["lat"] and s["lon"]) for s in sites if s["round"] == "2026-10")


def test_expert_items_follow_expert_reason():
    acceptable = ["fails", "probably_fails"]
    for s in json.loads(SITES_JSON.read_text()):
        if s["round"] != "2026-09":
            assert "expert_items" not in s
            continue
        reason = (s.get("expert_reason") or "").lower()
        want = {}
        if "power line over 80 ft" in reason:
            want["high_300"] = acceptable
        if "lighting pole" in reason or "wall over 15 ft" in reason:
            want["low_50"] = acceptable
        assert s["expert_items"] == want
        if s["expert"] == "approved":
            assert s["expert_items"] == {}
