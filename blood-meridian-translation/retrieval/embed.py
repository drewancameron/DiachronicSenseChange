"""Multi-scale embedding and FAISS index building.

Produces three FAISS IndexFlatIP indices (phrase, sentence, passage) over the
harvested Greek corpus, following the pattern from subproject/src/retrieval.py.

Usage:
    python -m blood-meridian-translation.retrieval.embed
    # or:
    python blood-meridian-translation/retrieval/embed.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Iterator

import faiss
import numpy as np

from .config import (
    BATCH_SIZE,
    CORPUS_JSONL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    INDEX_DIR,
    PASSAGE_WINDOW_MAX,
    PASSAGE_WINDOW_MIN,
    PHRASE_STRIDE,
    PHRASE_WINDOW_MAX,
    PHRASE_WINDOW_MIN,
)
from .schemas import CorpusRecord, EmbeddedChunk, Scale

logger = logging.getLogger(__name__)


# ── Lazy model loader ──────────────────────────────────────────────

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts, L2-normalize, return float32 array."""
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=len(texts) > 500,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


# ── Corpus loader ──────────────────────────────────────────────────


def _load_corpus() -> list[CorpusRecord]:
    """Load harvested records from corpus.jsonl."""
    records: list[CorpusRecord] = []
    with open(CORPUS_JSONL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(CorpusRecord.model_validate_json(line))
    return records


# ── Chunk generators ───────────────────────────────────────────────


def _generate_sentence_chunks(
    records: list[CorpusRecord],
) -> Iterator[EmbeddedChunk]:
    """One chunk per corpus record (sentence-level)."""
    for i, rec in enumerate(records):
        yield EmbeddedChunk(
            chunk_id=i,
            record_id=rec.record_id,
            scale=Scale.SENTENCE,
            text=rec.text,
            author=rec.author,
            work=rec.work,
            period=rec.period,
        )


def _generate_phrase_chunks(
    records: list[CorpusRecord],
) -> Iterator[EmbeddedChunk]:
    """Sliding-window phrase chunks (3-8 tokens, stride 2) over each record."""
    chunk_id = 0
    for rec in records:
        tokens = rec.text.split()
        if len(tokens) < PHRASE_WINDOW_MIN:
            continue
        for win_size in range(PHRASE_WINDOW_MIN, min(PHRASE_WINDOW_MAX + 1, len(tokens) + 1)):
            for start in range(0, len(tokens) - win_size + 1, PHRASE_STRIDE):
                phrase_tokens = tokens[start : start + win_size]
                yield EmbeddedChunk(
                    chunk_id=chunk_id,
                    record_id=rec.record_id,
                    scale=Scale.PHRASE,
                    text=" ".join(phrase_tokens),
                    author=rec.author,
                    work=rec.work,
                    period=rec.period,
                    token_start=start,
                    token_end=start + win_size,
                )
                chunk_id += 1


def _generate_passage_chunks(
    records: list[CorpusRecord],
) -> Iterator[EmbeddedChunk]:
    """Combine 2-5 consecutive sentences from the same work into passages."""
    chunk_id = 0
    # Group records by (author, work)
    groups: dict[tuple[str, str], list[CorpusRecord]] = {}
    for rec in records:
        key = (rec.author, rec.work)
        groups.setdefault(key, []).append(rec)

    for (author, work), recs in groups.items():
        for win_size in range(PASSAGE_WINDOW_MIN, PASSAGE_WINDOW_MAX + 1):
            for start in range(0, len(recs) - win_size + 1):
                window = recs[start : start + win_size]
                combined_text = " ".join(r.text for r in window)
                # Use the first record's ID as anchor
                yield EmbeddedChunk(
                    chunk_id=chunk_id,
                    record_id=window[0].record_id,
                    scale=Scale.PASSAGE,
                    text=combined_text,
                    author=author,
                    work=work,
                    period=window[0].period,
                )
                chunk_id += 1


# ── Index builder ──────────────────────────────────────────────────


def _build_scale_index(
    scale: Scale,
    chunks: list[EmbeddedChunk],
    force_rebuild: bool = False,
) -> int:
    """Embed chunks, build IndexFlatIP, save index + metadata.

    Returns the number of vectors in the index.
    """
    index_path = INDEX_DIR / f"{scale.value}.index"
    meta_path = INDEX_DIR / f"{scale.value}_meta.jsonl"

    if not force_rebuild and index_path.exists():
        logger.info("Index %s already exists — skipping (use force_rebuild=True)", index_path)
        idx = faiss.read_index(str(index_path))
        return idx.ntotal

    logger.info("Building %s index from %d chunks ...", scale.value, len(chunks))

    # Extract texts
    texts = [c.text for c in chunks]

    # Embed in manageable batches to avoid OOM
    index: faiss.IndexFlatIP | None = None
    embed_batch = 5000
    for start in range(0, len(texts), embed_batch):
        batch = texts[start : start + embed_batch]
        logger.info("  Embedding %s batch %d–%d / %d",
                     scale.value, start, start + len(batch), len(texts))
        vecs = _embed_texts(batch)
        if index is None:
            dim = vecs.shape[1]
            index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        del vecs

    if index is None:
        logger.warning("No vectors produced for scale %s", scale.value)
        return 0

    # Save index
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    # Save metadata (one JSON object per line, aligned with vector index)
    with open(meta_path, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(c.model_dump_json() + "\n")

    logger.info("Saved %s index: %d vectors, dim %d → %s",
                scale.value, index.ntotal, EMBEDDING_DIM, index_path)
    return index.ntotal


# ── Main entry point ───────────────────────────────────────────────


def build_all_indices(force_rebuild: bool = False) -> dict[str, int]:
    """Build phrase, sentence, and passage FAISS indices.

    Returns a dict mapping scale name to vector count.
    """
    records = _load_corpus()
    logger.info("Loaded %d corpus records from %s", len(records), CORPUS_JSONL)

    results: dict[str, int] = {}

    # 1) Sentence-level (one per record — fast, do first)
    sentence_chunks = list(_generate_sentence_chunks(records))
    results["sentence"] = _build_scale_index(Scale.SENTENCE, sentence_chunks, force_rebuild)

    # 2) Phrase-level (sliding windows — many chunks, but short texts)
    # To control memory, we cap total phrase chunks
    MAX_PHRASE_CHUNKS = 500_000
    phrase_chunks: list[EmbeddedChunk] = []
    for chunk in _generate_phrase_chunks(records):
        phrase_chunks.append(chunk)
        if len(phrase_chunks) >= MAX_PHRASE_CHUNKS:
            logger.warning("Capping phrase chunks at %d", MAX_PHRASE_CHUNKS)
            break
    results["phrase"] = _build_scale_index(Scale.PHRASE, phrase_chunks, force_rebuild)
    del phrase_chunks

    # 3) Passage-level (2-5 consecutive sentences)
    # Cap to avoid excessive memory usage
    MAX_PASSAGE_CHUNKS = 200_000
    passage_chunks: list[EmbeddedChunk] = []
    for chunk in _generate_passage_chunks(records):
        passage_chunks.append(chunk)
        if len(passage_chunks) >= MAX_PASSAGE_CHUNKS:
            logger.warning("Capping passage chunks at %d", MAX_PASSAGE_CHUNKS)
            break
    results["passage"] = _build_scale_index(Scale.PASSAGE, passage_chunks, force_rebuild)
    del passage_chunks

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    force = "--force" in sys.argv
    stats = build_all_indices(force_rebuild=force)
    print("\n=== Index Build Summary ===")
    for scale, n in stats.items():
        print(f"  {scale:10s}: {n:>8,d} vectors")
    total = sum(stats.values())
    print(f"  {'TOTAL':10s}: {total:>8,d} vectors")
