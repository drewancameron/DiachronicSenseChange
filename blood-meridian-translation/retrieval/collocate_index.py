"""Collocate index: PMI-ranked word co-occurrences for key vocabulary.

Loads lemmata from idf_glossary.json, scans the corpus for co-occurring
tokens within a 5-token window, computes pointwise mutual information,
and persists the index to data/collocates.json.
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .config import CORPUS_JSONL, DATA_DIR
from .schemas import CollocateEntry, CorpusRecord

logger = logging.getLogger(__name__)

# Paths
_GLOSSARY_PATH = Path(__file__).resolve().parent.parent / "glossary" / "idf_glossary.json"
_COLLOCATE_PATH = DATA_DIR / "collocates.json"

# Parameters
WINDOW_SIZE = 5
TOP_K_COLLOCATES = 20
MIN_CO_OCCURRENCE = 2


# ---------------------------------------------------------------------------
# Greek normalisation (matching find_occurrences.py)
# ---------------------------------------------------------------------------

def _strip_accents(text: str) -> str:
    """Strip combining marks and lowercase for matching."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.lower()


def _tokenise(text: str) -> list[str]:
    """Simple whitespace tokeniser with punctuation stripping."""
    tokens = text.split()
    cleaned = []
    for t in tokens:
        t = re.sub(r"^[^\w]+|[^\w]+$", "", t, flags=re.UNICODE)
        if t:
            cleaned.append(t)
    return cleaned


# ---------------------------------------------------------------------------
# Glossary extraction
# ---------------------------------------------------------------------------

def load_glossary_lemmata(glossary_path: Path | None = None) -> dict[str, str]:
    """Extract ancient_greek field values from idf_glossary.json.

    Returns a dict mapping normalised (accent-stripped) form -> original form.
    Strips leading articles and asterisks from forms.
    """
    path = glossary_path or _GLOSSARY_PATH
    if not path.exists():
        logger.warning("Glossary not found at %s", path)
        return {}

    with open(path) as f:
        data = json.load(f)

    lemmata: dict[str, str] = {}

    def _extract_from_section(section: dict) -> None:
        for _key, entry in section.items():
            if not isinstance(entry, dict):
                continue
            ag = entry.get("ancient_greek", "")
            if not ag:
                continue
            # May contain slashes for alternatives: "ὁ πηλός / ἡ ἰλύς"
            for variant in ag.split("/"):
                variant = variant.strip()
                # Strip leading asterisk (marks neologisms)
                variant = variant.lstrip("*").strip()
                # Strip articles
                for article in ["ὁ ", "ἡ ", "τό ", "τὸ ", "τά ", "τὰ ", "οἱ ", "αἱ "]:
                    if variant.startswith(article):
                        variant = variant[len(article):]
                        break
                # Some entries are periphrastic — take first word as lemma
                first_word = variant.split()[0] if variant.split() else variant
                norm = _strip_accents(first_word)
                if norm and len(norm) >= 2:
                    lemmata[norm] = first_word

    for section_key, section in data.items():
        if section_key.startswith("_"):
            continue
        if isinstance(section, dict):
            _extract_from_section(section)

    logger.info("Loaded %d glossary lemmata", len(lemmata))
    return lemmata


# ---------------------------------------------------------------------------
# Corpus loader (reads the same corpus.jsonl produced by harvest.py)
# ---------------------------------------------------------------------------

def _load_corpus() -> list[CorpusRecord]:
    """Load harvested records from corpus.jsonl."""
    if not CORPUS_JSONL.exists():
        raise FileNotFoundError(
            f"Corpus not found at {CORPUS_JSONL}. Run 'python3 -m retrieval harvest' first."
        )
    records: list[CorpusRecord] = []
    with open(CORPUS_JSONL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(CorpusRecord.model_validate_json(line))
    return records


# ---------------------------------------------------------------------------
# PMI computation
# ---------------------------------------------------------------------------

def _compute_pmi(
    cooccurrences: dict[str, dict[str, int]],
    word_counts: Counter,
    total_windows: int,
) -> dict[str, list[tuple[str, float, int]]]:
    """Compute PMI for each target lemma's collocates.

    Returns {lemma_norm: [(collocate, pmi_score, raw_count), ...]} sorted by PMI.
    """
    results: dict[str, list[tuple[str, float, int]]] = {}

    for lemma_norm, collocate_counts in cooccurrences.items():
        p_lemma = word_counts[lemma_norm] / total_windows if total_windows > 0 else 0
        if p_lemma == 0:
            continue

        scored = []
        for collocate, co_count in collocate_counts.items():
            if co_count < MIN_CO_OCCURRENCE:
                continue
            p_collocate = word_counts[collocate] / total_windows if total_windows > 0 else 0
            if p_collocate == 0:
                continue
            p_joint = co_count / total_windows
            pmi = math.log2(p_joint / (p_lemma * p_collocate))
            scored.append((collocate, pmi, co_count))

        scored.sort(key=lambda x: x[1], reverse=True)
        results[lemma_norm] = scored[:TOP_K_COLLOCATES]

    return results


# ---------------------------------------------------------------------------
# Corpus scanning and index building
# ---------------------------------------------------------------------------

def build_collocate_index(
    records: list[CorpusRecord] | None = None,
    glossary_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, list[CollocateEntry]]:
    """Scan corpus records and build PMI-ranked collocate index.

    Args:
        records: Corpus records. If None, loads from corpus.jsonl.
        glossary_path: Path to idf_glossary.json (default: auto-detected).
        output_path: Where to write collocates.json (default: data/collocates.json).

    Returns:
        Dict mapping normalised lemma -> list of CollocateEntry.
    """
    if records is None:
        records = _load_corpus()

    out = output_path or _COLLOCATE_PATH
    lemmata = load_glossary_lemmata(glossary_path)
    if not lemmata:
        logger.warning("No glossary lemmata loaded; collocate index will be empty")
        return {}

    lemma_norms = set(lemmata.keys())

    # Count co-occurrences and unigrams
    cooccurrences: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    word_counts: Counter = Counter()
    total_windows = 0

    for record in records:
        tokens = _tokenise(record.text)
        norm_tokens = [_strip_accents(t) for t in tokens]

        word_counts.update(norm_tokens)
        total_windows += max(1, len(norm_tokens))

        for i, nt in enumerate(norm_tokens):
            if nt not in lemma_norms:
                continue
            window_start = max(0, i - WINDOW_SIZE)
            window_end = min(len(norm_tokens), i + WINDOW_SIZE + 1)
            for j in range(window_start, window_end):
                if j == i:
                    continue
                collocate = norm_tokens[j]
                if len(collocate) < 2:
                    continue
                cooccurrences[nt][collocate] += 1

    logger.info(
        "Scanned %d records, %d total token-windows, %d target lemmata with hits",
        len(records), total_windows, len(cooccurrences),
    )

    pmi_results = _compute_pmi(cooccurrences, word_counts, total_windows)

    # Build CollocateEntry objects and serialisable output
    index: dict[str, list[CollocateEntry]] = {}
    serialisable: dict[str, list[dict]] = {}

    for lemma_norm, scored in pmi_results.items():
        original_form = lemmata.get(lemma_norm, lemma_norm)
        entries = []
        ser_entries = []
        for collocate_norm, pmi, count in scored:
            entry = CollocateEntry(
                target_lemma=original_form,
                collocate=collocate_norm,
                frequency=count,
                period="all",  # aggregated across periods
                pmi=round(pmi, 4),
            )
            entries.append(entry)
            ser_entries.append(entry.model_dump())
        index[lemma_norm] = entries
        serialisable[lemma_norm] = ser_entries

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(serialisable, f, ensure_ascii=False, indent=2)
    logger.info("Wrote collocate index to %s (%d lemmata)", out, len(serialisable))

    return index


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def load_collocate_index(path: Path | None = None) -> dict[str, list[CollocateEntry]]:
    """Load pre-built collocate index from disk."""
    p = path or _COLLOCATE_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Collocate index not found at {p}. Run 'python3 -m retrieval collocates' first."
        )
    with open(p) as f:
        data = json.load(f)

    index: dict[str, list[CollocateEntry]] = {}
    for lemma_norm, entries in data.items():
        index[lemma_norm] = [
            CollocateEntry(**e) for e in entries
        ]
    return index


def lookup(
    lemma: str,
    index: dict[str, list[CollocateEntry]] | None = None,
    path: Path | None = None,
) -> list[CollocateEntry]:
    """Look up collocates for a given lemma.

    Accepts either a pre-loaded index or loads from disk.
    The lemma is normalised (accent-stripped) before lookup.
    """
    if index is None:
        index = load_collocate_index(path)

    norm = _strip_accents(lemma)
    return index.get(norm, [])
