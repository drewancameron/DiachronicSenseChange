#!/usr/bin/env python3
"""
Acquire Greek texts and translations from Perseus Digital Library.

Downloads TEI XML files from the PerseusDL GitHub repositories,
organizing them by author and work with provenance logging.
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

# Perseus GitHub API endpoints for canonical Greek/Latin repos
REPOS = [
    "PerseusDL/canonical-greekLit",
    "PerseusDL/canonical-latinLit",  # included for comparative work
]

# We only want Greek texts and English translations for now
GREEK_PATTERN = re.compile(r"grc\d*\.xml$", re.IGNORECASE)
ENG_PATTERN = re.compile(r"eng\d*\.xml$", re.IGNORECASE)

# GitHub API rate limit: 60 requests/hour unauthenticated
API_BASE = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def log_provenance(entry: dict) -> None:
    """Append a provenance entry to the JSONL log."""
    PROVENANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROVENANCE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def fetch_url(url: str, accept: str = "application/json") -> bytes:
    """Fetch a URL with basic rate-limit handling."""
    req = Request(url, headers={"Accept": accept, "User-Agent": "DiachronicSenseChange/0.1"})
    try:
        with urlopen(req) as resp:
            return resp.read()
    except HTTPError as e:
        if e.code == 403:
            log.warning("Rate limited. Waiting 60s...")
            time.sleep(60)
            with urlopen(req) as resp:
                return resp.read()
        raise


def list_xml_files(repo: str, path: str = "data") -> list[dict]:
    """Recursively list XML files in a GitHub repo directory via the API."""
    url = f"{API_BASE}/repos/{repo}/git/trees/master?recursive=1"
    data = json.loads(fetch_url(url))
    files = []
    for item in data.get("tree", []):
        if item["type"] == "blob" and item["path"].startswith(path):
            if GREEK_PATTERN.search(item["path"]) or ENG_PATTERN.search(item["path"]):
                files.append({"path": item["path"], "sha": item["sha"]})
    return files


def download_file(repo: str, file_path: str, local_dir: Path) -> Path | None:
    """Download a single file from GitHub raw content."""
    url = f"{RAW_BASE}/{repo}/master/{file_path}"
    local_path = local_dir / file_path
    if local_path.exists():
        log.info(f"  Already exists: {local_path.name}")
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        content = fetch_url(url, accept="application/xml")
    except HTTPError as e:
        log.error(f"  Failed to download {file_path}: {e}")
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
        "repo": repo,
    })

    log.info(f"  Downloaded: {file_path} ({len(content)} bytes)")
    return local_path


def acquire_repo(repo: str, max_files: int | None = None) -> None:
    """Download Greek and English XML files from a Perseus repo."""
    log.info(f"Listing files in {repo}...")
    files = list_xml_files(repo)
    log.info(f"Found {len(files)} Greek/English XML files")

    if max_files:
        files = files[:max_files]
        log.info(f"Limiting to first {max_files} files")

    local_dir = RAW_DIR / repo.split("/")[-1]
    for i, f in enumerate(files):
        log.info(f"[{i+1}/{len(files)}] {f['path']}")
        download_file(repo, f["path"], local_dir)
        # Be gentle with GitHub's rate limits
        time.sleep(0.5)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Acquire Perseus texts")
    parser.add_argument("--repo", default="PerseusDL/canonical-greekLit",
                        help="GitHub repo to download from")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of files (for testing)")
    parser.add_argument("--list-only", action="store_true",
                        help="List available files without downloading")
    args = parser.parse_args()

    if args.list_only:
        files = list_xml_files(args.repo)
        for f in files:
            print(f["path"])
        print(f"\nTotal: {len(files)} files")
    else:
        acquire_repo(args.repo, max_files=args.max_files)


if __name__ == "__main__":
    main()
