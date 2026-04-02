"""Heuristic construction-type tagger for Ancient Greek text.

Uses regex patterns on surface forms — no CLTK or morphological parser needed.
Tags sentences/passages with syntactic construction types used for retrieval filtering.
"""

from __future__ import annotations

import re
import unicodedata

from .schemas import CorpusRecord, ConstructionType


# ---------------------------------------------------------------------------
# Greek normalisation helpers
# ---------------------------------------------------------------------------

def _strip_accents(text: str) -> str:
    """Strip combining marks (accents, breathings) and lowercase."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.lower()


def _token_count(text: str) -> int:
    return len(text.split())


def _clause_boundaries(text: str) -> list[str]:
    """Split on common clause-boundary punctuation."""
    return [c.strip() for c in re.split(r"[·;,.·:—–]", text) if c.strip()]


# ---------------------------------------------------------------------------
# Pattern definitions — each returns a confidence score 0-1
# ---------------------------------------------------------------------------

# Paratactic / coordination markers (normalised)
_PARA_MARKERS = re.compile(
    r"\b(και|δε|δ᾽|τε|αλλα|αλλ᾽|μεν|ουδε|ουτε|μηδε|μητε)\b"
)

# Participial endings (genitive absolute often ends in these)
# Genitive singular: -οντος, -ουσης, -μενου, -μενης, -αντος, -εντος
_GEN_ABS_ENDINGS = re.compile(
    r"\b\w*(οντος|ουσης|μενου|μενης|αντος|εντος|οντων|ουσων|μενων)\b"
)

# Subordination markers
_SUBORD_MARKERS = re.compile(
    r"\b(οτι|ως|ωστε|ινα|οπως|επει|επειδη|διοτι|καθως|πριν|μεχρι|εως|οταν|οτε)\b"
)

# Direct speech markers
_SPEECH_MARKERS = re.compile(
    r"\b(εφη|ειπεν|ειπε|ελεγεν|ελεγε|ελεξεν|λεγει|λεγων|φησιν|φησι|απεκρινατο|απεκριθη|εκελευσεν)\b"
)

# Catalogue / list markers: τε...καί patterns, repeated καί
_CATALOGUE_MARKERS = re.compile(r"\b(τε\s+και|και\s+\w+\s+και)\b")

# Conditional markers — require 3+ chars to avoid matching εἰ as article/particle
_CONDITIONAL_MARKERS = re.compile(
    r"\b(εαν|ειπερ|ειγε|ειτε|εανπερ)\b"
)
# Separate check: εἰ only counts when followed by a verb-like word (rough heuristic)
_EI_CONDITIONAL = re.compile(r"\bει\s+(?:μη|τις|δε|γαρ|ουν|\w{4,})")

# Relative pronoun forms — only longer/unambiguous forms to avoid false positives
# (short forms like ος, η, ο are also articles and pronouns)
_RELATIVE_MARKERS = re.compile(
    r"\b(οστις|ητις|οιτινες|αιτινες|οσος|οποιος|οποια|οποιον)\b"
)


def _score_paratactic(norm: str, clauses: list[str]) -> float:
    """High kai/de density + short clauses -> paratactic narrative."""
    tokens = norm.split()
    if not tokens:
        return 0.0
    marker_hits = len(_PARA_MARKERS.findall(norm))
    density = marker_hits / len(tokens)
    avg_clause = sum(_token_count(c) for c in clauses) / max(len(clauses), 1)
    short_bonus = 1.0 if avg_clause < 10 else 0.5 if avg_clause < 15 else 0.0
    score = min(1.0, density * 4.0) * 0.6 + short_bonus * 0.4
    if marker_hits < 2:
        score *= 0.3
    return score


def _score_participial(norm: str) -> float:
    """Genitive absolute / participial constructions."""
    tokens = norm.split()
    if not tokens:
        return 0.0
    hits = len(_GEN_ABS_ENDINGS.findall(norm))
    if hits < 1:
        return 0.0
    density = hits / len(tokens)
    return min(1.0, density * 8.0)


def _score_periodic(norm: str, clauses: list[str]) -> float:
    """Long sentence + subordination markers -> periodic rhetoric."""
    tokens = norm.split()
    n_tokens = len(tokens)
    if n_tokens < 15:
        return 0.0
    subord_hits = len(_SUBORD_MARKERS.findall(norm))
    subord_density = subord_hits / n_tokens
    length_score = min(1.0, n_tokens / 40.0)
    subord_score = min(1.0, subord_density * 10.0)
    score = length_score * 0.4 + subord_score * 0.6
    if subord_hits < 2:
        score *= 0.3
    return score


def _score_direct_speech(norm: str) -> float:
    """Speech verbs present -> direct speech."""
    hits = len(_SPEECH_MARKERS.findall(norm))
    if hits == 0:
        return 0.0
    return min(1.0, hits * 0.5)


def _score_catalogue(norm: str) -> float:
    """Repeated te...kai or long lists."""
    hits = len(_CATALOGUE_MARKERS.findall(norm))
    kai_count = norm.count("και")
    tokens = norm.split()
    if not tokens:
        return 0.0
    kai_density = kai_count / len(tokens)
    score = min(1.0, hits * 0.3 + kai_density * 3.0)
    if kai_count < 3:
        score *= 0.3
    return score


def _score_conditional(norm: str) -> float:
    """Conditional constructions — requires unambiguous markers."""
    hits = len(_CONDITIONAL_MARKERS.findall(norm))
    ei_hits = len(_EI_CONDITIONAL.findall(norm))
    total = hits + ei_hits
    if total == 0:
        return 0.0
    return min(1.0, total * 0.4)


def _score_relative(norm: str) -> float:
    """Dense relative clauses — only unambiguous relative pronouns."""
    tokens = norm.split()
    if not tokens:
        return 0.0
    hits = len(_RELATIVE_MARKERS.findall(norm))
    if hits < 1:
        return 0.0
    return min(1.0, hits * 0.5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_THRESHOLD = 0.45


def tag(text: str) -> list[ConstructionType]:
    """Tag a Greek text string with detected construction types.

    Returns a list of ConstructionType values whose heuristic score
    exceeds the confidence threshold. Multiple types may co-occur.
    """
    if not text or not text.strip():
        return []

    norm = _strip_accents(text)
    clauses = _clause_boundaries(norm)

    scorers: list[tuple[ConstructionType, float]] = [
        (ConstructionType.PARATACTIC_NARRATIVE, _score_paratactic(norm, clauses)),
        (ConstructionType.PARTICIPIAL_DESCRIPTION, _score_participial(norm)),
        (ConstructionType.PERIODIC_RHETORIC, _score_periodic(norm, clauses)),
        (ConstructionType.DIRECT_SPEECH, _score_direct_speech(norm)),
        (ConstructionType.CATALOGUE, _score_catalogue(norm)),
        (ConstructionType.CONDITIONAL, _score_conditional(norm)),
        (ConstructionType.RELATIVE_CLAUSE, _score_relative(norm)),
    ]

    return [ctype for ctype, score in scorers if score >= _THRESHOLD]


def tag_corpus(records: list[CorpusRecord]) -> dict[str, list[ConstructionType]]:
    """Tag every record in a corpus with its construction types.

    Returns a dict mapping record_id -> list of ConstructionType.
    (CorpusRecord is immutable Pydantic; we return a separate mapping.)
    """
    tags: dict[str, list[ConstructionType]] = {}
    for record in records:
        detected = tag(record.text)
        if detected:
            tags[record.record_id] = detected
    return tags
