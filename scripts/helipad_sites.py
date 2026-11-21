"""Real helipad sites through the private lz_check.py ($HELIPAD_PRIVATE/heli_lz; online maps) -> .context/helipads/sites/<slug>/lz.json, and a check
that helipad.rules reproduces lz_check's verdict on each, on the full case and on the case as rendered (visible()).

uv run scripts/helipad_sites.py fetch [--only s1,o1]      # GovMap answers Israeli IPs only: run from Israel, ~3 min/site
uv run scripts/helipad_sites.py refresh                     # re-add the full obstacle lists (helipad_lz_full.py), no network
uv run scripts/helipad_sites.py check                       # exit 1 on any rules (full or visible) != lz_check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from helipad.case import visible  # noqa: E402
from helipad.from_lz import case_from_lz, lz_verdict  # noqa: E402
from helipad.paths import LZ_CHECK as LZ, SITES_JSON as SITES, require  # noqa: E402
from helipad.rules import judge, verdict  # noqa: E402

OUT = ROOT / ".context" / "helipads" / "sites"
FULL = ROOT / "scripts" / "helipad_lz_full.py"
GEO = [x for d in ("shapely", "pyproj", "rasterio", "mercantile", "requests", "numpy", "pillow") for x in ("--with", d)]
TIMEOUT = 900   # s per site; lz_check has no timeout of its own


def _geo(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["uv", "run", "--no-project", *GEO, "python", str(script), *args],
                          capture_output=True, text=True, timeout=TIMEOUT)


def is_complete(lz: dict) -> bool:
    """A fetch with OSM or DEM missing (Overpass often 504s) is kept for inspection but refetched next run."""
    return bool(lz.get("osm_ok") and lz.get("dem_ok"))


def lz_record(data: dict, v: dict) -> dict:
    return {"verdict": v, "govmap": data.get("govmap") or {},
            "osm_ok": (data.get("osm") or {}).get("elements") is not None,
            "dem_ok": bool((v.get("slope") or {}).get("available"))}


def add_full_lists(d: Path) -> None:
    try:
        r = _geo(FULL, str(d))
    except subprocess.TimeoutExpired:
        r = subprocess.CompletedProcess([], 1, "", "timeout")
    print(r.stdout.strip() if not r.returncode else f"{d.name}: full obstacle list FAILED ({r.stderr.strip()[-300:]})")


def refresh(sites_dir: Path) -> None:
    for f in sorted(Path(sites_dir).glob("*/lz.json")):
        add_full_lists(f.parent)


def fetch(only: set[str] | None = None) -> None:
    require(LZ)
    for s in json.loads(require(SITES).read_text()):
        if only and s["slug"] not in only:
            continue
        d = OUT / s["slug"]
        if (d / "lz.json").exists() and is_complete(json.loads((d / "lz.json").read_text())):
            print(f"{s['slug']}: cached")
            continue
        where = f"{s['lat']}, {s['lon']}" if s["lat"] is not None else s["link"]
        try:
            r = _geo(LZ, "run", where, "--aircraft", "1", "--out", str(d))
        except subprocess.TimeoutExpired:
            print(f"{s['slug']}: FAILED (timeout {TIMEOUT} s)")
            continue
        d.mkdir(parents=True, exist_ok=True)
        (d / "lz_check.log").write_text(r.stdout + r.stderr)
        if r.returncode or not (d / "verdict.json").exists():
            print(f"{s['slug']}: FAILED (see {d / 'lz_check.log'})")
            continue
        lz = lz_record(json.loads((d / "data.json").read_text()), json.loads((d / "verdict.json").read_text()))
        (d / "lz.json").write_text(json.dumps(lz, ensure_ascii=False, indent=1))
        print(f"{s['slug']}: {lz_verdict(lz)}")
        add_full_lists(d)


def agreement(sites_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(Path(sites_dir).glob("*/lz.json")):
        lz = json.loads(f.read_text())
        c = case_from_lz(lz, f.parent.name)
        v = visible(c)
        mine, shown, theirs = verdict(c, judge(c)), verdict(v, judge(v)), lz_verdict(lz)
        rows.append({"slug": f.parent.name, "lz": theirs, "rules": mine, "visible": shown, "match": mine == shown == theirs,
                     "osm": bool(lz.get("osm_ok")), "dem": bool(lz.get("dem_ok")), "govmap": c.govmap_ok, "corridors": c.corridors_ok})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "refresh", "check"])
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(set(filter(None, a.only.split(","))) or None)
        return 0
    if a.cmd == "refresh":
        refresh(OUT)
        return 0
    rows = agreement(OUT)
    for r in rows:
        have = " ".join(f"{k}={'y' if r[k] else 'n'}" for k in ("osm", "dem", "govmap", "corridors"))
        print(f"{r['slug']:16} lz={r['lz']:15} rules={r['rules']:15} visible={r['visible']:15} "
              f"{'ok' if r['match'] else 'MISMATCH':8} {have}")
    print(f"{sum(r['match'] for r in rows)}/{len(rows)} match")
    return 0 if all(r["match"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
