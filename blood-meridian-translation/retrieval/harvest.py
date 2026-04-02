"""Corpus harvester: parse Tesserae .tess and LXX token-per-line files into a
unified corpus.jsonl for downstream embedding.

Usage:
    python -m blood-meridian-translation.retrieval.harvest
    # or simply:
    python blood-meridian-translation/retrieval/harvest.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Iterator

from .config import (
    CORPUS_JSONL,
    DATA_DIR,
    LXX_DIR,
    LXX_PERIOD,
    TESSERAE_DIR,
    TESSERAE_PRIORITY,
)
from .schemas import CorpusRecord

logger = logging.getLogger(__name__)

# ── Tesserae parser ────────────────────────────────────────────────

_TESS_LINE = re.compile(r"^<([^>]+)>[\t ]+(.+)$")


def parse_tesserae_file(
    path: Path, author: str, work: str, period: str
) -> Iterator[CorpusRecord]:
    """Parse a Tesserae .tess file.

    Format: <ref>\\tGreek text  or  <ref> Greek text  (one line per verse/sentence)
    Some files use tab separators, others use spaces after the closing >.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = _TESS_LINE.match(line)
            if m is None:
                continue
            ref_raw, text = m.group(1), m.group(2).strip()
            if not text:
                continue
            # Normalise reference: remove extra spaces
            ref = re.sub(r"\s+", ".", ref_raw.strip())
            record_id = f"{author}.{work}.{ref}"
            yield CorpusRecord(
                record_id=record_id,
                author=author,
                work=work,
                reference=ref,
                period=period,
                text=text,
                source="tesserae",
            )


def _iter_tesserae_priority() -> Iterator[CorpusRecord]:
    """Harvest all priority-author Tesserae files."""
    for key, (pattern, period) in TESSERAE_PRIORITY.items():
        # Derive canonical author / work from the key
        # e.g. "homer" -> author=homer, work=iliad (from filename)
        files = sorted(TESSERAE_DIR.glob(pattern))
        if not files:
            logger.warning("No files matched pattern %s for key %s", pattern, key)
            continue
        logger.info("  %s: %d file(s)", key, len(files))
        for fp in files:
            # Extract author.work from filename like homer.iliad.part.1.tess
            stem = fp.stem  # e.g. homer.iliad.part.1
            parts = stem.split(".")
            author = parts[0]
            # work = everything between author and "part" (or end)
            if "part" in parts:
                work = ".".join(parts[1 : parts.index("part")])
            else:
                work = ".".join(parts[1:]) if len(parts) > 1 else stem
            yield from parse_tesserae_file(fp, author, work, period)


# ── LXX parser ─────────────────────────────────────────────────────


def parse_lxx_file(path: Path) -> Iterator[CorpusRecord]:
    """Parse an LXX Swete token-per-line file, aggregating to verse level.

    Format: chapter.verse.subverse TOKEN   (one token per line)
    We group by chapter.verse to produce one record per verse.
    """
    # Derive book name from filename: 01.Genesis.txt -> Genesis
    fname = path.stem  # e.g. "01.Genesis"
    book_parts = fname.split(".", 1)
    book_name = book_parts[1] if len(book_parts) > 1 else fname

    current_verse: str | None = None
    tokens: list[str] = []

    def _flush() -> CorpusRecord | None:
        if current_verse is None or not tokens:
            return None
        text = " ".join(tokens)
        ref = f"{current_verse}"
        record_id = f"lxx.{book_name}.{ref}"
        return CorpusRecord(
            record_id=record_id,
            author="lxx",
            work=book_name,
            reference=ref,
            period=LXX_PERIOD,
            text=text,
            source="lxx-swete",
        )

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            ref_full, token = parts[0], parts[1]
            # ref_full is e.g. "1.1.1" — group by chapter.verse (first two parts)
            ref_parts = ref_full.split(".")
            if len(ref_parts) >= 2:
                verse_key = f"{ref_parts[0]}.{ref_parts[1]}"
            else:
                verse_key = ref_full

            if verse_key != current_verse:
                rec = _flush()
                if rec is not None:
                    yield rec
                current_verse = verse_key
                tokens = [token]
            else:
                tokens.append(token)

    # flush last verse
    rec = _flush()
    if rec is not None:
        yield rec


def _iter_lxx() -> Iterator[CorpusRecord]:
    """Harvest all LXX files."""
    files = sorted(LXX_DIR.glob("*.txt"))
    if not files:
        logger.warning("No LXX files found in %s", LXX_DIR)
        return
    logger.info("  LXX: %d file(s)", len(files))
    for fp in files:
        yield from parse_lxx_file(fp)


# ── Main harvest ───────────────────────────────────────────────────


def harvest_all() -> int:
    """Harvest all priority corpora and write corpus.jsonl.

    Returns the total number of records written.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    count = 0

    logger.info("Harvesting corpus into %s ...", CORPUS_JSONL)

    with open(CORPUS_JSONL, "w", encoding="utf-8") as out:
        # LXX first (highest priority)
        for rec in _iter_lxx():
            if rec.record_id in seen_ids:
                continue
            seen_ids.add(rec.record_id)
            out.write(rec.model_dump_json() + "\n")
            count += 1

        # Tesserae priority authors
        for rec in _iter_tesserae_priority():
            if rec.record_id in seen_ids:
                continue
            seen_ids.add(rec.record_id)
            out.write(rec.model_dump_json() + "\n")
            count += 1

    logger.info("Harvested %d records (deduplicated from %d seen IDs)",
                count, len(seen_ids))
    return count


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    n = harvest_all()
    print(f"Done. {n} records written to {CORPUS_JSONL}")
