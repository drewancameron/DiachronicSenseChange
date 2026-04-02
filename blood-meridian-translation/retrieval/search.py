"""Unified search engine with four retrieval modes.

Modes:
  1. lexical_inspiration  — EN->GRC cross-lingual phrase/sentence search
  2. syntactic_templates  — semantic query filtered by construction type
  3. register_calibration — GRC->GRC nearest-neighbour check
  4. collocation_discovery — PMI-ranked collocate lookup
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from .config import EMBEDDING_MODEL, INDEX_DIR, DATA_DIR
from .schemas import (
    EmbeddedChunk,
    RetrievalResult,
    CollocateEntry,
    Scale,
    ConstructionType,
)
from .collocate_index import lookup as collocate_lookup, load_collocate_index
from .construction_tagger import tag as tag_text

logger = logging.getLogger(__name__)


_PARENT_EMB_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "embeddings"


# ---------------------------------------------------------------------------
# Embedding model (lazy singleton)
# ---------------------------------------------------------------------------

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Loaded embedding model: %s", EMBEDDING_MODEL)
    return _embed_model


def _embed(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Embed texts using the configured sentence-transformer.

    Returns L2-normalised vectors for cosine similarity via inner product.
    """
    model = _get_embed_model()
    if "e5" in EMBEDDING_MODEL.lower():
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + t for t in texts]
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        batch_size=32,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# FAISS index loading
# ---------------------------------------------------------------------------

# Cache: scale -> (faiss index, list of EmbeddedChunk metadata dicts)
_loaded_indices: dict[Scale, tuple[faiss.IndexFlatIP, list[dict]]] = {}

# Cache: record_id -> list of construction types (strings)
_construction_tags: dict[str, list[str]] | None = None


def _load_construction_tags() -> dict[str, list[str]]:
    """Load or compute construction tags for the corpus."""
    global _construction_tags
    if _construction_tags is not None:
        return _construction_tags

    tags_path = DATA_DIR / "construction_tags.json"
    if tags_path.exists():
        with open(tags_path) as f:
            _construction_tags = json.load(f)
        logger.info("Loaded construction tags for %d records", len(_construction_tags))
        return _construction_tags

    # If no cached tags, return empty — user should run the tagger first
    logger.warning(
        "No construction_tags.json found at %s. "
        "Run build_construction_tags() or 'python3 -m retrieval tag' to generate.",
        tags_path,
    )
    _construction_tags = {}
    return _construction_tags


def _load_index(scale: Scale) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Load a FAISS index and its metadata for the given scale.

    Metadata is stored as JSONL (one JSON object per line), aligned with
    the vector indices in the FAISS file.
    """
    if scale in _loaded_indices:
        return _loaded_indices[scale]

    idx_path = INDEX_DIR / f"{scale.value}.index"
    meta_path = INDEX_DIR / f"{scale.value}_meta.jsonl"

    if not idx_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {idx_path}. Run 'python3 -m retrieval embed' first."
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {meta_path}. Run 'python3 -m retrieval embed' first."
        )

    index = faiss.read_index(str(idx_path))

    metadata: list[dict] = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))

    logger.info("Loaded %s index: %d vectors, %d metadata entries",
                scale.value, index.ntotal, len(metadata))
    _loaded_indices[scale] = (index, metadata)
    return index, metadata


def _search_index(
    query_vec: np.ndarray,
    scale: Scale,
    top_k: int,
    period_filter: str | None = None,
    construction_filter: ConstructionType | None = None,
) -> list[RetrievalResult]:
    """Search a FAISS index and return ranked results with metadata."""
    index, metadata = _load_index(scale)
    construction_tags = _load_construction_tags() if construction_filter else {}

    # Over-retrieve to allow for post-filtering
    search_k = top_k * 20 if construction_filter else top_k * 5 if period_filter else top_k
    search_k = min(search_k, index.ntotal)
    if search_k == 0:
        return []

    scores, indices = index.search(query_vec, search_k)

    results: list[RetrievalResult] = []
    rank = 0
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        meta = metadata[idx]
        record_id = meta.get("record_id", "")

        # Period filter
        if period_filter and meta.get("period", "") != period_filter:
            continue

        # Construction type filter
        if construction_filter:
            record_tags = construction_tags.get(record_id, [])
            if construction_filter.value not in record_tags:
                continue

        rank += 1
        chunk = EmbeddedChunk(**meta)
        result = RetrievalResult(
            chunk=chunk,
            score=float(score),
            rank=rank,
        )
        results.append(result)

        if len(results) >= top_k:
            break

    return results


# ---------------------------------------------------------------------------
# Build construction tags (called during pipeline setup)
# ---------------------------------------------------------------------------

def build_construction_tags() -> int:
    """Tag the entire corpus and save to data/construction_tags.json.

    Returns the number of records tagged.
    """
    from .construction_tagger import tag as tag_text_fn
    from .config import CORPUS_JSONL
    from .schemas import CorpusRecord

    if not CORPUS_JSONL.exists():
        raise FileNotFoundError(
            f"Corpus not found at {CORPUS_JSONL}. Run harvest first."
        )

    tags: dict[str, list[str]] = {}
    count = 0
    with open(CORPUS_JSONL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = CorpusRecord.model_validate_json(line)
            detected = tag_text_fn(rec.text)
            if detected:
                tags[rec.record_id] = [ct.value for ct in detected]
                count += 1

    tags_path = DATA_DIR / "construction_tags.json"
    tags_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tags_path, "w") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)

    logger.info("Tagged %d records (of which %d have tags), saved to %s",
                count, len(tags), tags_path)

    # Reset cache
    global _construction_tags
    _construction_tags = tags
    return count


# ---------------------------------------------------------------------------
# Search mode 1: Lexical inspiration (EN -> GRC)
# ---------------------------------------------------------------------------

def lexical_inspiration(
    english_query: str,
    scale: Scale = Scale.PHRASE,
    period_filter: str | None = None,
    top_k: int = 10,
) -> list[RetrievalResult]:
    """Cross-lingual English->Greek retrieval.

    Embeds the English query with a multilingual model and searches
    the Greek FAISS index at the requested scale.
    """
    query_vec = _embed([english_query], is_query=True)
    return _search_index(query_vec, scale, top_k, period_filter=period_filter)


# ---------------------------------------------------------------------------
# Search mode 2: Syntactic templates
# ---------------------------------------------------------------------------

def syntactic_templates(
    semantic_query: str,
    construction_type: ConstructionType,
    scale: Scale = Scale.SENTENCE,
    top_k: int = 10,
) -> list[RetrievalResult]:
    """Semantic search filtered by construction type.

    Embeds the query, searches the sentence-level index, and filters
    results to those tagged with the specified construction type.
    """
    query_vec = _embed([semantic_query], is_query=True)
    return _search_index(
        query_vec, scale, top_k,
        construction_filter=construction_type,
    )


# ---------------------------------------------------------------------------
# Search mode 3: Register calibration (GRC -> GRC)
# ---------------------------------------------------------------------------

def register_calibration(
    greek_draft: str,
    scale: Scale = Scale.SENTENCE,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Greek->Greek nearest-neighbour search for register checking.

    Embeds a draft Greek sentence and finds its nearest neighbours
    in the ancient corpus.
    """
    query_vec = _embed([greek_draft], is_query=True)
    return _search_index(query_vec, scale, top_k)


# ---------------------------------------------------------------------------
# Search mode 4: Collocation discovery
# ---------------------------------------------------------------------------

def collocation_discovery(
    lemma: str,
    period_filter: str | None = None,
) -> list[CollocateEntry]:
    """Look up precomputed collocates for a target lemma."""
    return collocate_lookup(lemma)


# ---------------------------------------------------------------------------
# Unified search dispatcher
# ---------------------------------------------------------------------------

def search(
    query: str,
    mode: str,
    scale: Scale = Scale.SENTENCE,
    construction_type: ConstructionType | None = None,
    period_filter: str | None = None,
    top_k: int = 10,
) -> list[RetrievalResult] | list[CollocateEntry]:
    """Unified search dispatcher.

    Args:
        query: The search query (English, Greek, or lemma depending on mode).
        mode: One of "lexical", "syntactic", "register", "collocate".
        scale: Embedding scale for vector-based modes.
        construction_type: Required for syntactic mode.
        period_filter: Optional period constraint.
        top_k: Number of results.

    Returns:
        List of RetrievalResult or CollocateEntry depending on mode.
    """
    if mode == "lexical":
        return lexical_inspiration(query, scale, period_filter, top_k)
    elif mode == "syntactic":
        if construction_type is None:
            raise ValueError("syntactic mode requires --construction argument")
        return syntactic_templates(query, construction_type, scale, top_k)
    elif mode == "register":
        return register_calibration(query, scale, top_k)
    elif mode == "collocate":
        return collocation_discovery(query, period_filter)
    else:
        raise ValueError(
            f"Unknown search mode: {mode!r}. Choose from: lexical, syntactic, register, collocate"
        )


# ---------------------------------------------------------------------------
# Index status
# ---------------------------------------------------------------------------

def index_status() -> dict[str, dict]:
    """Report on available indices and their sizes."""
    status: dict[str, dict] = {}

    for scale in Scale:
        idx_path = INDEX_DIR / f"{scale.value}.index"
        meta_path = INDEX_DIR / f"{scale.value}_meta.jsonl"
        info: dict = {"exists": idx_path.exists()}
        if idx_path.exists():
            try:
                idx = faiss.read_index(str(idx_path))
                info["vectors"] = idx.ntotal
                info["dimension"] = idx.d
                info["size_mb"] = round(idx_path.stat().st_size / (1024 * 1024), 2)
            except Exception as e:
                info["error"] = str(e)
        if meta_path.exists():
            info["meta_exists"] = True
            # Count lines
            with open(meta_path) as f:
                info["meta_records"] = sum(1 for line in f if line.strip())
        status[scale.value] = info

    # Collocate index
    colloc_path = DATA_DIR / "collocates.json"
    colloc_info: dict = {"exists": colloc_path.exists()}
    if colloc_path.exists():
        with open(colloc_path) as f:
            data = json.load(f)
        colloc_info["lemmata"] = len(data)
        colloc_info["total_collocates"] = sum(len(v) for v in data.values())
        colloc_info["size_mb"] = round(colloc_path.stat().st_size / (1024 * 1024), 2)
    status["collocates"] = colloc_info

    # Construction tags
    tags_path = DATA_DIR / "construction_tags.json"
    tags_info: dict = {"exists": tags_path.exists()}
    if tags_path.exists():
        with open(tags_path) as f:
            data = json.load(f)
        tags_info["tagged_records"] = len(data)
    status["construction_tags"] = tags_info

    # Corpus
    from .config import CORPUS_JSONL
    corpus_info: dict = {"exists": CORPUS_JSONL.exists()}
    if CORPUS_JSONL.exists():
        with open(CORPUS_JSONL) as f:
            corpus_info["records"] = sum(1 for line in f if line.strip())
        corpus_info["size_mb"] = round(CORPUS_JSONL.stat().st_size / (1024 * 1024), 2)
    status["corpus"] = corpus_info

    # Parent embeddings
    parent_emb = _PARENT_EMB_DIR / "embeddings.npy"
    parent_info: dict = {"exists": parent_emb.exists()}
    if parent_emb.exists():
        arr = np.load(str(parent_emb))
        parent_info["vectors"] = arr.shape[0]
        parent_info["dimension"] = arr.shape[1] if arr.ndim > 1 else 0
        parent_info["size_mb"] = round(parent_emb.stat().st_size / (1024 * 1024), 2)
    status["parent_embeddings"] = parent_info

    return status
