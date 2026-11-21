"""Private helipad data (site registry, the lz_check pipeline) lives outside the repo; override with $HELIPAD_PRIVATE."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIVATE = Path(os.environ.get("HELIPAD_PRIVATE") or ROOT / ".context/helipads/private")
SITES_JSON = PRIVATE / "sites.json"
LZ_SCRIPTS = PRIVATE / "heli_lz/scripts"
LZ_CHECK = LZ_SCRIPTS / "lz_check.py"


def require(p: Path) -> Path:
    if not Path(p).exists():
        raise FileNotFoundError(f"private helipad data not found at {p}; set HELIPAD_PRIVATE")
    return Path(p)
