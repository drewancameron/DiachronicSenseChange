"""Pydantic data contracts for the retrieval system."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Scale(str, Enum):
    """Granularity of an embedded chunk."""
    PHRASE = "phrase"
    SENTENCE = "sentence"
    PASSAGE = "passage"


class ConstructionType(str, Enum):
    """Syntactic construction tags for filtering."""
    PARATACTIC_NARRATIVE = "paratactic_narrative"
    PARTICIPIAL_DESCRIPTION = "participial_description"
    PERIODIC_RHETORIC = "periodic_rhetoric"
    DIRECT_SPEECH = "direct_speech"
    CATALOGUE = "catalogue"
    CONDITIONAL = "conditional"
    RELATIVE_CLAUSE = "relative_clause"


class CorpusRecord(BaseModel):
    """A single harvested text unit (one sentence / verse)."""
    record_id: str = Field(..., description="Unique identifier: author.work.ref")
    author: str
    work: str
    reference: str
    period: str
    text: str
    source: str = Field(..., description="'tesserae' or 'lxx-swete'")


class EmbeddedChunk(BaseModel):
    """Metadata for one embedded vector in a FAISS index."""
    chunk_id: int
    record_id: str
    scale: Scale
    text: str
    author: str
    work: str
    period: str
    token_start: Optional[int] = None
    token_end: Optional[int] = None


class RetrievalResult(BaseModel):
    """A single retrieval hit returned to the user."""
    chunk: EmbeddedChunk
    score: float
    rank: int


class CollocateEntry(BaseModel):
    """A precomputed collocate for a target lemma."""
    target_lemma: str
    collocate: str
    frequency: int
    period: str
    pmi: float = Field(0.0, description="Pointwise mutual information score")
