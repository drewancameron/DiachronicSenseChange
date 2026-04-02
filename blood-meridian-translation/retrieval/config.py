"""Paths and settings for the Blood Meridian retrieval system."""

from __future__ import annotations

from pathlib import Path

# ── Root paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # DiachronicSenseChange/
CORPUS_RAW = PROJECT_ROOT / "corpus" / "raw"
RETRIEVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = RETRIEVAL_DIR / "data"
INDEX_DIR = DATA_DIR / "indices"
CORPUS_JSONL = DATA_DIR / "corpus.jsonl"

# ── Corpus source directories ──────────────────────────────────────
TESSERAE_DIR = CORPUS_RAW / "tesserae"
LXX_DIR = CORPUS_RAW / "lxx-swete"

# ── Embedding model ────────────────────────────────────────────────
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384  # MiniLM-L12-v2 output dimension
BATCH_SIZE = 256

# ── Scale parameters ───────────────────────────────────────────────
PHRASE_WINDOW_MIN = 3
PHRASE_WINDOW_MAX = 8
PHRASE_STRIDE = 2

PASSAGE_WINDOW_MIN = 2
PASSAGE_WINDOW_MAX = 5

# ── Priority authors and their file patterns ───────────────────────
# Maps a canonical author key to (glob pattern relative to TESSERAE_DIR, period tag)
TESSERAE_PRIORITY = {
    "homer": ("homer.iliad.part.*.tess", "homeric"),
    "homer_odyssey": ("homer.odyssey.part.*.tess", "homeric"),
    "herodotus": ("herodotus.histories.part.*.tess", "classical"),
    "thucydides": ("thucydides.peleponnesian_war.part.*.tess", "classical"),
    "xenophon": ("xenophon.anabasis.tess", "classical"),
    "plutarch": ("plutarch.*.tess", "imperial"),
    "new_testament": ("new_testament.*.tess", "koine"),
}

# LXX files are handled separately (different format)
LXX_PERIOD = "hellenistic"
