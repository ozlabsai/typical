"""The Enron email sample of typically: download the CMU corpus (cached under .context/enron, never committed), measure a few
executives' mailboxes through typically_mail, and write one as site/data/typically_enron.jsonl.gz.

    uv run --no-sync python scripts/typically_enron.py                    # measure the candidates, print a table
    uv run --no-sync python scripts/typically_enron.py --write kaminski-v # write that mailbox as the sample
"""
import argparse
import gzip
import io
import json
import random
import shutil
import tarfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import typically_mail as tm

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / ".context" / "enron"
TARBALL = CACHE / "enron_mail_20150507.tar.gz"
URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
OUT = REPO / "site" / "data" / "typically_enron.jsonl.gz"
CANDIDATES = ["kaminski-v", "dasovich-j", "kean-s", "farmer-d", "shackleton-s", "jones-t", "lavorato-j", "kitchen-l",
              "haedicke-m", "lay-k", "skilling-j", "delainey-d", "whalley-g", "beck-s", "germany-c", "mann-k"]
MAX_ROWS = 20_000


def maildirs(users: list[str]) -> Path:
    """Extract maildir/<user>/ for each user once (one pass over the 1.7 GB tarball)."""
    root = CACHE / "maildir"
    todo = [u for u in users if not (root / u).exists()]
    if todo:
        if not TARBALL.exists():
            CACHE.mkdir(parents=True, exist_ok=True)
            print(f"downloading {URL} (~423 MB)")
            with urllib.request.urlopen(URL) as r, open(TARBALL.with_suffix(".part"), "wb") as f:
                shutil.copyfileobj(r, f)
            TARBALL.with_suffix(".part").rename(TARBALL)
        want = tuple(f"maildir/{u}/" for u in todo)
        with tarfile.open(TARBALL, "r|gz") as tar:   # streamed: members in archive order
            for m in tar:
                if m.isfile() and m.name.startswith(want):
                    dest = CACHE / m.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(tar.extractfile(m).read())
    return root


def as_zip(d: Path) -> bytes:
    """A user's maildir as the .zip a user would upload."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(p for p in d.rglob("*") if p.is_file()):
            z.write(f, f.relative_to(d.parent).as_posix())
    return buf.getvalue()


def measure(user: str, root: Path) -> tuple[list[dict], dict, list[str]]:
    rows, stats, warnings = tm.read_mail(as_zip(root / user))
    share = lambda c, v: sum(r.get(c) == v for r in rows) / len(rows)
    filed = Counter(r["folder"] for r in rows if r.get("folder"))
    print(f"{user:14} msgs {stats['messages']:6} threads {stats['threads']:6} inbound {stats['inbound']:6}  owner {stats['owner']:32}"
          f" reply {share('reply', 'yes'):.0%}  filed {sum(filed.values()) / len(rows):.0%}\n    within {dict(Counter(r.get('reply_within') for r in rows).most_common())}"
          f"\n    folders {dict(filed.most_common())}")
    for w in warnings:
        print("   ", w)
    return rows, stats, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", nargs="*", default=CANDIDATES)
    ap.add_argument("--write", help="write this user's mailbox as the sample")
    a = ap.parse_args()
    users = [a.write] if a.write else a.users
    root = maildirs(users)
    for u in users:
        rows, stats, _ = measure(u, root)
    if a.write:
        rng = random.Random(0)
        if len(rows) > MAX_ROWS:   # keep every reply (the rarer answer), sample the rest
            yes = [r for r in rows if r.get("reply") == "yes"]
            rows = yes + rng.sample([r for r in rows if r.get("reply") != "yes"], MAX_ROWS - len(yes))
        rng.shuffle(rows)
        with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(f"wrote {len(rows)} rows, {OUT.stat().st_size / 1e6:.1f} MB -> {OUT.relative_to(REPO)}; owner {stats['owner']}")


if __name__ == "__main__":
    main()
