# Embedding Retrieval Design for Blood Meridian Translation

## Available Infrastructure

From the parent project (`../models/embeddings/`):
- **6,126 word-sense embeddings** (768-dim, float32)
- Keyed by: lemma, sense, period, author, register, surface_form
- Periods: homeric, archaic, classical, hellenistic, koine, imperial
- This gives us **lexical-level** retrieval — "what senses does this word have across periods?"

## What We Need to Build

### Index 1: Multi-scale sentence/phrase embeddings

Embed Greek text from our retrieval corpus at three granularities:

| Scale | Window | Source corpus priority |
|-------|--------|----------------------|
| Phrase (3–8 tokens) | Sliding window, stride 2 | LXX, NT, Herodotus, Thucydides |
| Sentence | Full sentence | All prose corpus |
| Passage (2–5 sentences) | Paragraph-level | Narrative passages only |

**Embedding model:** Use a multilingual sentence-transformer that handles polytonic Greek.
Candidates:
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (fast, good enough)
- The Krahn et al. Ancient Greek sentence embeddings (ALP 2023) if available
- Fine-tuned model from the parent project's WP4

**Query strategy:** Embed McCarthy's English at the same three scales, then retrieve Greek nearest neighbours. Cross-lingual retrieval works because multilingual transformers map semantically similar content close together regardless of language.

### Index 2: Construction-type index (symbolic)

Tag Greek sentences by syntactic pattern:
- `paratactic_narrative` — καί-chains, asyndeton, short main clauses
- `participial_description` — genitive absolute, circumstantial participles
- `periodic_rhetoric` — long hypotactic periods with subordination
- `direct_speech` — dialogue, speech acts
- `catalogue` — lists, enumerations
- `conditional` — εἰ/ἐάν constructions
- `relative_clause` — dense relative clause embedding

This is a tag filter, not an embedding — used to constrain retrieval results to syntactically appropriate exemplars.

### Index 3: Collocate index

For key vocabulary (the IDF glossary words), precompute:
- Top 20 collocates within a 5-word window across the corpus
- Grouped by period and register
- This answers "what words naturally cluster with πηλός?" without needing to run a full search

## Four Use Cases

### 1. Lexical inspiration
**When:** Composing a draft, looking for the right word
**Query:** English phrase or rough Greek paraphrase
**Retrieval:** Cross-lingual phrase-level nearest neighbours
**Output:** Ranked list of Greek phrases with source/period metadata
**Example:**
- Query: "rags of snow"
- Returns: ῥάκη νιφάδων (hypothetical), χιὼν κατεσπαρμένη, etc.
- The translator picks the best fit or is inspired by an unexpected image

### 2. Syntactic templates
**When:** Structuring a clause, choosing between coordination and subordination
**Query:** Construction-type tag + semantic query
**Retrieval:** Greek sentences matching both the construction type and semantic field
**Output:** Exemplar sentences showing how Greek handles this pattern
**Example:**
- Query: `paratactic_narrative` + "travel, landscape, solitude"
- Returns: Herodotus travel descriptions, LXX wilderness narratives
- The translator sees how Greek authors chain short clauses in landscape description

### 3. Register calibration
**When:** After drafting, checking that the Greek sounds right
**Query:** Embed the draft Greek sentence, find its nearest Greek neighbours
**Retrieval:** Sentence-level kNN in the Greek-only space
**Output:** The 5 nearest ancient Greek sentences + their source metadata
**If** the neighbours are from the right period/register → draft is on target
**If** the neighbours are from the wrong register → revise
**Example:**
- Draft: ὁ δὲ παῖς πτώσσει παρὰ τὸ πῦρ
- Nearest neighbours: Homer Il. 4.371 (πτώσσοντες), Hes. Op. 524 → Homeric register confirmed ✓

### 4. Collocation discovery
**When:** Choosing modifiers, verbs, prepositions for a target noun
**Query:** Target lemma
**Retrieval:** Precomputed collocate list from the collocate index
**Output:** Ranked collocates with period/register tags
**Example:**
- Query: πηλός (clay)
- Returns: πλάσσω (shape, Genesis), ποιέω (make), κεραμεύς (potter, Jeremiah), εἶδος (form, Plato)
- This tells us πηλός + εἶδος is a real Greek collocation — confirming our ἄλλο τι εἶδος πηλοῦ

## Corpus Priorities for Embedding

1. **LXX / Septuagint** — highest priority. BM's register maps here best.
2. **New Testament** — narrative portions (Gospels, Acts) for paratactic narrative; Pauline epistles for theological vocabulary.
3. **Herodotus** — ethnographic description, travel, warfare, foreign peoples.
4. **Thucydides** — analytical rhetoric (the judge's register).
5. **Homer** — selectively, for epic landscape and violence vocabulary.
6. **Xenophon** — military narrative, practical prose.
7. **Plutarch** — biographical narrative, moral reflection.
8. **Philo** — philosophical/theological Greek that bridges Septuagintal and classical.

## Implementation Plan

1. **Harvest** sentence-segmented Greek from Perseus/Scaife for the priority corpora
2. **Embed** at three scales using multilingual sentence-transformer
3. **Tag** with construction types (can be done with simple heuristics + OdyCy parses)
4. **Build** collocate index from lemmatised corpus
5. **Wire** into the drafting workflow: before composing each passage, run retrieval and present top results alongside the MG bridge notes
6. **Wire** register calibration as a post-draft check

## Integration with Existing Parent Project

The parent project's 6,126 word-sense embeddings serve use case 4 directly — they are period-tagged sense vectors. For a given lemma, we can:
- Find which senses are attested in which periods
- Find semantically similar lemmata across periods
- Check whether our word choice is period-appropriate

The new sentence/phrase embeddings extend this to the clause level.
