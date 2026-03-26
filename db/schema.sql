-- Diachronic Sense Change: Core SQLite Schema
-- Designed for provenance-rich occurrence-level sense evidence

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-----------------------------------------------------------
-- 1. SOURCE LAYER: authors, documents, editions, translations
-----------------------------------------------------------

CREATE TABLE IF NOT EXISTS authors (
    author_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    name_greek TEXT,
    tlg_id TEXT,
    floruit_start INTEGER,  -- approximate year (negative = BCE)
    floruit_end INTEGER,
    period TEXT CHECK(period IN (
        'homeric', 'archaic', 'classical',
        'hellenistic', 'imperial', 'koine'
    )),
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS documents (
    document_id INTEGER PRIMARY KEY,
    author_id INTEGER REFERENCES authors(author_id),
    title TEXT NOT NULL,
    title_greek TEXT,
    tlg_work_id TEXT,
    genre TEXT,
    approximate_date INTEGER,  -- negative = BCE
    period TEXT CHECK(period IN (
        'homeric', 'archaic', 'classical',
        'hellenistic', 'imperial', 'koine'
    )),
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS editions (
    edition_id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(document_id),
    editor TEXT,
    publisher TEXT,
    year INTEGER,
    source_url TEXT,
    rights_status TEXT CHECK(rights_status IN (
        'public_domain', 'creative_commons', 'fair_use',
        'restricted', 'unknown'
    )),
    format TEXT,  -- 'xml', 'txt', 'html', 'pdf'
    checksum TEXT,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS translations (
    translation_id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(document_id),
    translator TEXT NOT NULL,
    language TEXT DEFAULT 'en',
    year INTEGER,
    source_url TEXT,
    rights_status TEXT CHECK(rights_status IN (
        'public_domain', 'creative_commons', 'fair_use',
        'restricted', 'unknown'
    )),
    has_notes INTEGER DEFAULT 0,  -- boolean: translator's notes present
    checksum TEXT,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS commentaries (
    commentary_id INTEGER PRIMARY KEY,
    document_id INTEGER REFERENCES documents(document_id),
    author_name TEXT NOT NULL,  -- commentator
    title TEXT,
    year INTEGER,
    source_url TEXT,
    rights_status TEXT CHECK(rights_status IN (
        'public_domain', 'creative_commons', 'fair_use',
        'restricted', 'unknown'
    )),
    commentary_type TEXT CHECK(commentary_type IN (
        'line_commentary', 'lexicon', 'grammar',
        'monograph', 'article', 'scholia'
    )),
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-----------------------------------------------------------
-- 2. PASSAGE LAYER: segments, passages, tokens
-----------------------------------------------------------

CREATE TABLE IF NOT EXISTS segments (
    segment_id INTEGER PRIMARY KEY,
    edition_id INTEGER NOT NULL REFERENCES editions(edition_id),
    segment_type TEXT CHECK(segment_type IN (
        'book', 'chapter', 'section', 'line_range', 'paragraph'
    )),
    reference TEXT NOT NULL,  -- e.g. "Il.1.1-10" or "Rep.509b"
    raw_text TEXT,
    cleaned_text TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS passages (
    passage_id INTEGER PRIMARY KEY,
    segment_id INTEGER NOT NULL REFERENCES segments(segment_id),
    document_id INTEGER NOT NULL REFERENCES documents(document_id),
    reference TEXT NOT NULL,          -- canonical ref e.g. "Il.1.1"
    greek_text TEXT NOT NULL,
    greek_context_window TEXT,        -- broader context around passage
    word_count INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tokens (
    token_id INTEGER PRIMARY KEY,
    passage_id INTEGER NOT NULL REFERENCES passages(passage_id),
    surface_form TEXT NOT NULL,
    lemma TEXT,
    pos_tag TEXT,
    morphology TEXT,           -- morphological parse string
    token_offset INTEGER,      -- position within passage
    is_target INTEGER DEFAULT 0,  -- flagged as target for analysis
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tokens_lemma ON tokens(lemma);
CREATE INDEX IF NOT EXISTS idx_tokens_passage ON tokens(passage_id);

-----------------------------------------------------------
-- 3. ALIGNMENT LAYER: linking Greek to translations/notes
-----------------------------------------------------------

CREATE TABLE IF NOT EXISTS alignments (
    alignment_id INTEGER PRIMARY KEY,
    passage_id INTEGER NOT NULL REFERENCES passages(passage_id),
    translation_id INTEGER REFERENCES translations(translation_id),
    commentary_id INTEGER REFERENCES commentaries(commentary_id),
    aligned_text TEXT NOT NULL,       -- the English/scholarly text
    alignment_type TEXT CHECK(alignment_type IN (
        'translation', 'footnote', 'commentary',
        'lexicon_entry', 'scholion', 'preface'
    )),
    alignment_confidence REAL CHECK(alignment_confidence BETWEEN 0 AND 1),
    alignment_method TEXT,            -- 'manual', 'reference_match', 'llm'
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-----------------------------------------------------------
-- 4. SENSE INVENTORY
-----------------------------------------------------------

CREATE TABLE IF NOT EXISTS sense_inventory (
    sense_id INTEGER PRIMARY KEY,
    lemma TEXT NOT NULL,
    sense_label TEXT NOT NULL,         -- short label e.g. "order/arrangement"
    sense_description TEXT,            -- longer definition
    lsj_reference TEXT,               -- LSJ entry cross-ref
    period_earliest TEXT,              -- earliest attested period
    period_latest TEXT,                -- latest attested period
    parent_sense_id INTEGER REFERENCES sense_inventory(sense_id),
    revision INTEGER DEFAULT 1,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sense_lemma ON sense_inventory(lemma);

-----------------------------------------------------------
-- 5. EVIDENCE AND LABELLING LAYER
-----------------------------------------------------------

CREATE TABLE IF NOT EXISTS occurrences (
    occurrence_id INTEGER PRIMARY KEY,
    token_id INTEGER NOT NULL REFERENCES tokens(token_id),
    lemma TEXT NOT NULL,
    passage_id INTEGER NOT NULL REFERENCES passages(passage_id),
    document_id INTEGER NOT NULL REFERENCES documents(document_id),
    greek_context TEXT NOT NULL,       -- local Greek window
    period TEXT,
    genre TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_occurrences_lemma ON occurrences(lemma);

CREATE TABLE IF NOT EXISTS candidate_labels (
    label_id INTEGER PRIMARY KEY,
    occurrence_id INTEGER NOT NULL REFERENCES occurrences(occurrence_id),
    sense_id INTEGER NOT NULL REFERENCES sense_inventory(sense_id),
    confidence REAL CHECK(confidence BETWEEN 0 AND 1),
    is_primary INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS evidence_spans (
    evidence_id INTEGER PRIMARY KEY,
    label_id INTEGER NOT NULL REFERENCES candidate_labels(label_id),
    alignment_id INTEGER REFERENCES alignments(alignment_id),
    evidence_text TEXT NOT NULL,       -- exact quoted span
    evidence_type TEXT CHECK(evidence_type IN (
        'translation_rendering', 'translator_note',
        'commentary_discussion', 'lexicon_gloss',
        'scholion', 'llm_inference'
    )),
    source_identity TEXT,             -- translator/commentator name
    directness TEXT CHECK(directness IN (
        'direct', 'inferential', 'contextual'
    )),
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS register_labels (
    register_id INTEGER PRIMARY KEY,
    occurrence_id INTEGER NOT NULL REFERENCES occurrences(occurrence_id),
    register TEXT CHECK(register IN (
        'unmarked', 'poetic_elevated', 'archaizing',
        'homericizing', 'quoted_allusive', 'scholastic',
        'colloquial', 'technical', 'uncertain'
    )),
    confidence REAL CHECK(confidence BETWEEN 0 AND 1),
    evidence_text TEXT,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-----------------------------------------------------------
-- 6. ADJUDICATION LAYER
-----------------------------------------------------------

CREATE TABLE IF NOT EXISTS adjudications (
    adjudication_id INTEGER PRIMARY KEY,
    occurrence_id INTEGER NOT NULL REFERENCES occurrences(occurrence_id),
    adjudicator TEXT NOT NULL,
    final_sense_id INTEGER REFERENCES sense_inventory(sense_id),
    final_register TEXT,
    agreement_level TEXT CHECK(agreement_level IN (
        'clear', 'probable', 'disputed', 'genuinely_ambiguous'
    )),
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-----------------------------------------------------------
-- 7. MODEL AND EXPERIMENT LAYER
-----------------------------------------------------------

CREATE TABLE IF NOT EXISTS model_splits (
    split_id INTEGER PRIMARY KEY,
    occurrence_id INTEGER NOT NULL REFERENCES occurrences(occurrence_id),
    split_name TEXT NOT NULL,         -- 'train', 'val', 'test', 'gold'
    split_version INTEGER DEFAULT 1,
    exclusion_reason TEXT,            -- e.g. 'leave_author_out:Plato'
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS embeddings (
    embedding_id INTEGER PRIMARY KEY,
    occurrence_id INTEGER NOT NULL REFERENCES occurrences(occurrence_id),
    model_name TEXT NOT NULL,
    model_version TEXT,
    embedding BLOB,                   -- serialized vector
    layer INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

-----------------------------------------------------------
-- 8. PROVENANCE
-----------------------------------------------------------

CREATE TABLE IF NOT EXISTS provenance_events (
    event_id INTEGER PRIMARY KEY,
    entity_type TEXT NOT NULL,        -- 'evidence_span', 'candidate_label', etc.
    entity_id INTEGER NOT NULL,
    action TEXT NOT NULL,             -- 'created', 'updated', 'reviewed'
    agent TEXT NOT NULL,              -- 'llm:claude-3.5', 'human:reviewer1'
    model_version TEXT,
    prompt_hash TEXT,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_provenance_entity
    ON provenance_events(entity_type, entity_id);
