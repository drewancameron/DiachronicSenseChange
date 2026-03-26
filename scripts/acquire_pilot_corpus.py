#!/usr/bin/env python3
"""
Acquire the pilot corpus: targeted texts from Perseus covering
the full Homeric-to-Koine arc for the 12 pilot lemmata.

Focuses on key authors with both Greek text and English translation
available, prioritizing breadth across periods and genres.
"""

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
RAW_DIR = CORPUS_DIR / "raw" / "perseus"
PROVENANCE_LOG = CORPUS_DIR / "raw" / "provenance_log.jsonl"

RAW_BASE = "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master"
API_BASE = "https://api.github.com"

# Priority authors for the pilot, keyed by TLG number
# Covering: Homeric, Archaic, Classical, Hellenistic, Imperial, Koine
PILOT_AUTHORS = {
    # HOMERIC / ARCHAIC
    "tlg0012": {"name": "Homer", "period": "homeric", "works": {
        "tlg001": "Iliad",
        "tlg002": "Odyssey",
    }},
    "tlg0020": {"name": "Hesiod", "period": "archaic", "works": {
        "tlg001": "Theogony",
        "tlg002": "Works and Days",
        "tlg003": "Shield of Heracles",
    }},
    "tlg0019": {"name": "Pindar", "period": "archaic", "works": {}},  # all works

    # CLASSICAL — TRAGEDY
    "tlg0085": {"name": "Aeschylus", "period": "classical", "works": {}},
    "tlg0011": {"name": "Sophocles", "period": "classical", "works": {}},
    "tlg0006": {"name": "Euripides", "period": "classical", "works": {}},
    "tlg0002": {"name": "Aristophanes", "period": "classical", "works": {}},

    # CLASSICAL — HISTORY
    "tlg0016": {"name": "Herodotus", "period": "classical", "works": {
        "tlg001": "Histories",
    }},
    "tlg0003": {"name": "Thucydides", "period": "classical", "works": {
        "tlg001": "History of the Peloponnesian War",
    }},
    "tlg0032": {"name": "Xenophon", "period": "classical", "works": {}},

    # CLASSICAL — PHILOSOPHY
    "tlg0059": {"name": "Plato", "period": "classical", "works": {}},
    "tlg0086": {"name": "Aristotle", "period": "classical", "works": {}},

    # CLASSICAL — ORATORY
    "tlg0014": {"name": "Demosthenes", "period": "classical", "works": {}},
    "tlg0010": {"name": "Isocrates", "period": "classical", "works": {}},
    "tlg0540": {"name": "Lysias", "period": "classical", "works": {}},

    # HELLENISTIC
    "tlg0557": {"name": "Epictetus", "period": "hellenistic", "works": {}},
    "tlg0541": {"name": "Polybius", "period": "hellenistic", "works": {}},

    # IMPERIAL
    "tlg0007": {"name": "Plutarch", "period": "imperial", "works": {}},
    "tlg0062": {"name": "Lucian", "period": "imperial", "works": {}},
    "tlg2003": {"name": "Strabo", "period": "imperial", "works": {}},

    # KOINE / EARLY CHRISTIAN
    "tlg7000": {"name": "New Testament", "period": "koine", "works": {}},
    "tlg0527": {"name": "Septuagint", "period": "koine", "works": {}},
}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def log_provenance(entry: dict) -> None:
    PROVENANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROVENANCE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def fetch_url(url: str, accept: str = "application/json") -> bytes | None:
    req = Request(url, headers={
        "Accept": accept,
        "User-Agent": "DiachronicSenseChange/0.1",
    })
    try:
        with urlopen(req) as resp:
            return resp.read()
    except HTTPError as e:
        if e.code == 403:
            log.warning("Rate limited. Waiting 60s...")
            time.sleep(60)
            try:
                with urlopen(req) as resp:
                    return resp.read()
            except HTTPError:
                return None
        elif e.code == 404:
            return None
        raise


def get_tree() -> list[dict]:
    """Get the full repo tree (single API call)."""
    url = f"{API_BASE}/repos/PerseusDL/canonical-greekLit/git/trees/master?recursive=1"
    data = json.loads(fetch_url(url))
    return data.get("tree", [])


def find_author_files(tree: list[dict], tlg_id: str) -> list[str]:
    """Find all Greek and English XML files for a given TLG author."""
    prefix = f"data/{tlg_id}/"
    files = []
    for item in tree:
        if item["type"] != "blob":
            continue
        if not item["path"].startswith(prefix):
            continue
        name = item["path"].lower()
        if name.endswith(".xml") and ("grc" in name or "eng" in name):
            files.append(item["path"])
    return sorted(files)


def download_file(file_path: str) -> Path | None:
    """Download a file from Perseus raw GitHub."""
    local_path = RAW_DIR / file_path
    if local_path.exists():
        log.info(f"  [cached] {file_path}")
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RAW_BASE}/{file_path}"
    content = fetch_url(url, accept="*/*")
    if content is None:
        log.error(f"  [failed] {file_path}")
        return None

    local_path.write_bytes(content)
    checksum = sha256(content)

    log_provenance({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "download",
        "source": url,
        "local_path": str(local_path),
        "checksum_sha256": checksum,
        "size_bytes": len(content),
    })

    log.info(f"  [downloaded] {file_path} ({len(content):,} bytes)")
    return local_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Acquire pilot corpus from Perseus")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--authors", nargs="*", help="Specific TLG IDs to download")
    args = parser.parse_args()

    log.info("Fetching Perseus repository tree...")
    tree = get_tree()
    log.info(f"Repository has {len(tree)} items")

    authors_to_fetch = args.authors if args.authors else list(PILOT_AUTHORS.keys())
    total_files = 0
    downloaded = 0

    for tlg_id in authors_to_fetch:
        info = PILOT_AUTHORS.get(tlg_id, {"name": tlg_id, "period": "?", "works": {}})
        files = find_author_files(tree, tlg_id)
        grc_count = sum(1 for f in files if "grc" in f.lower())
        eng_count = sum(1 for f in files if "eng" in f.lower())

        log.info(f"\n{'='*50}")
        log.info(f"{info['name']} ({tlg_id}) — {info['period']}")
        log.info(f"  {grc_count} Greek + {eng_count} English files")
        total_files += len(files)

        if args.list_only:
            for f in files:
                print(f"  {f}")
            continue

        for f in files:
            result = download_file(f)
            if result:
                downloaded += 1
            time.sleep(0.3)  # rate limiting

    log.info(f"\n{'='*50}")
    log.info(f"Total: {total_files} files identified, {downloaded} downloaded")


if __name__ == "__main__":
    main()
