# Project Plan: Blood Meridian — Ancient Greek Translation

## Purpose

Produce a scholarly Ancient Greek translation of selected passages from Cormac McCarthy's *Blood Meridian, or The Evening Redness in the West* (1985), accompanied by Reed-Kellogg sentence diagrams and a reader's apparatus for rare words and cases.

The target register is **not** strictly Attic. McCarthy's prose — paratactic, asyndetic, strongly biblical in cadence — maps more naturally onto a **Koine / Septuagintal register** with epic colouring where the narrative demands it. The system should be free to draw on Herodotean, Thucydidean, Septuagintal, and New Testament Greek as exemplar sources, not only classical Attic.

---

## Constraints and Resources

### What we have
- The diachronic sense-change models and embedding space (`../models/`)
- The Atticising translation pipeline (`../subproject/`) — retrieval, discourse bridge, constrained drafting, morphosyntactic validation
- Ancient Greek corpus and retrieval DB (`../corpus/`, `../db/`)
- Morphology / Wiktionary forms (`../config/wiktionary_forms.json`)
- Sense inventory with period-aware disambiguation (`../config/sense_inventory_clean.json`)

### What we don't have
- A published Modern Greek translation of *Blood Meridian* (no ebook exists)
- We **do** have a fragment: roughly the first ~10 pages in Modern Greek, useful for extracting idiom and seeing how a Greek translator handled McCarthy's register
- Copyright prevents redistribution of source text; all source passages are for internal research use only

---

## Work Packages

### WP0. Source Text Preparation

**Objective:** select, segment, and store the English passages we intend to translate.

**Tasks**
- Choose an initial passage set (5–10 passages of varying character: landscape description, dialogue, violence, biblical-register set-pieces, the judge's monologues)
- Segment each passage into sentences / clauses
- Tag each sentence with discourse type: narrative, dialogue, description, philosophical excursus, biblical cadence
- Store in a structured format (`passages/` directory, one JSON per passage with metadata)

**Note on copyright:** source text is held locally for research only; never committed to a public repository or redistributed.

---

### WP1. Inverse Document Frequency Glossary (IDF)

**Objective:** build a standardised bilingual glossary so that high-frequency words, character names, and modern material-culture terms are translated consistently throughout.

This is the **first thing to lock down** before any passage-level translation begins.

**Categories**

1. **Character names and epithets**
   - "the kid" — decide once: ὁ παῖς? τὸ παιδίον? ὁ νεανίσκος? (register implications differ)
   - "the judge" — ὁ δικαστής? ὁ κριτής? (Septuagintal κριτής has the right biblical resonance)
   - "Glanton" and other proper names — transliteration convention (Γλάντων? Γλάντον?)
   - "the Delawares", "the Comanche" — ethnonym policy

2. **Material culture and anachronisms**
   - guns / rifles — τὰ ὅπλα (generic)? τὰ πυροβόλα? a periphrasis?
   - gunpowder — ἡ πυρῖτις? descriptive phrase?
   - horses, mules, saddles — these have good classical equivalents
   - dollars, coins — ἀργύριον / νόμισμα
   - scalps / scalping — needs a fixed periphrasis or neologism

3. **Recurring vocabulary**
   - desert / plain / mesa — ἡ ἔρημος, τὸ πεδίον, etc.
   - blood — αἷμα (straightforward but track collocations)
   - fire, sun, dust, bones — the McCarthy elemental lexicon

**Deliverable:** `glossary/idf_glossary.json` — a machine-readable glossary with:
- English term
- chosen Greek rendering(s) with register tag
- justification / exemplar citation
- flag for terms still under discussion

**Process:** draft candidates, then review against the Modern Greek fragment and Google Translate output (see WP2) to check for idiom we might be missing.

---

### WP2. Modern Greek Idiom Bridge

**Objective:** use Modern Greek as an intermediate discourse layer, exactly as in the news-translation pipeline, but calibrated for literary rather than journalistic prose.

**Sources of Modern Greek signal**

1. **The existing ~10-page Greek fragment** of *Blood Meridian*
   - Extract and catalogue: how did the translator handle names? guns? the judge? landscape vocabulary?
   - Note clause-ordering choices and discourse packaging
   - This is our most valuable idiom resource — a professional translator's solutions to exactly our problems

2. **Google Translate as a rough bridge**
   - Run each English passage through Google Translate (EN → Modern Greek)
   - The output will be mediocre but useful for:
     - identifying natural Modern Greek clause order for McCarthy's parataxis
     - surfacing Modern Greek vocabulary in the right semantic fields
     - flagging where GT breaks down (these are our hardest translation problems)
   - Store GT output alongside the English source; never treat it as authoritative

3. **Manual Modern Greek paraphrase where needed**
   - For passages where GT is useless and the fragment doesn't cover, write a simplified Modern Greek paraphrase to expose discourse structure

**Deliverable:** for each passage, a `modern_greek_bridge.json` containing:
- fragment translator's renderings (where available)
- GT output
- extracted clause-order and vocabulary notes
- idiom decomposition notes

---

### WP3. Embedding-Based Retrieval for Composition

**Objective:** retrieve Ancient Greek sentences and phrases whose syntax, cadence, or semantic field can guide the Greek composition — reusing the parent project's embedding infrastructure.

**Key difference from the news pipeline:** we are not restricted to Attic. The retrieval should weight:
- **Septuagint / LXX** — for biblical cadence, landscape description, violence, divine/cosmic register
- **New Testament narrative** — for paratactic, asyndetic clause chaining (καί ... καί ... καί mirrors McCarthy's "and ... and ... and")
- **Herodotus** — for ethnographic description, travel, warfare
- **Thucydides** — for the judge's analytical/philosophical monologues
- **Homer** (selectively) — for epic simile and landscape

**Tasks**
- Extend the parent project's retrieval index to include Septuagintal and NT texts if not already present
- For each passage, run the normalised English through the semantic index
- Retrieve candidate Greek exemplars ranked by:
  - semantic similarity
  - register match (tagged by source corpus)
  - construction-type match (paratactic narrative, participial description, direct speech, etc.)
- Present retrieved exemplars to the drafting stage

**Deliverable:** `retrieval/` module that wraps the parent project's embedding search with BM-specific register weighting.

---

### WP4. Constrained Greek Drafting

**Objective:** compose Ancient Greek for each passage, conditioned on the IDF glossary, Modern Greek bridge, and retrieved exemplars.

**Register target:** Koine-adjacent with freedom to reach into classical and Septuagintal registers. Specifically:
- Default to Koine morphology and syntax (common dialect, not strictly Attic)
- Use Septuagintal vocabulary and cadence for biblical-register passages
- Use classical construction types for the judge's rhetoric
- Preserve McCarthy's parataxis: chains of καί, asyndeton, short main clauses
- Do **not** over-subordinate — McCarthy's power comes from coordination, not hypotaxis

**Process per passage**
1. Load IDF glossary entries relevant to the passage
2. Load Modern Greek bridge notes
3. Load top-N retrieved exemplars
4. Generate constrained draft via LLM with explicit instructions:
   - follow the glossary for standardised terms
   - match the source's clause rhythm
   - cite which exemplar influenced each clause
   - mark uncertain renderings
5. Generate a more literal alternative draft
6. Run both through morphosyntactic validation (reuse `../subproject/src/validate/`)

**Deliverable:** for each passage, a `drafts/` directory containing:
- primary draft (polytonic Greek, UTF-8)
- literal alternative
- clause-by-clause justification notes
- confidence and uncertainty annotations

---

### WP5. Reed-Kellogg Diagrams and Reader's Apparatus

**Objective:** produce, for each translated passage, a side table containing:

1. **Reed-Kellogg sentence diagrams** of the Greek output
   - Subject | Verb | Object on the main line
   - Modifiers on angled branches
   - Subordinate clauses on pedestals
   - Participial phrases on stepped lines
   - Purpose: make the Greek syntax visually parseable for a reader who knows some Greek but would struggle with long Koine periods

2. **Vocabulary and case glosses**
   - For every word that is rare (< some frequency threshold in our corpus) or whose case usage is non-obvious, provide:
     - lemma
     - morphological parse (e.g. "aorist passive participle, nominative masculine singular")
     - case reason (e.g. "genitive absolute", "accusative of respect", "dative of instrument")
     - English gloss
   - Frequency threshold: flag anything outside the top ~2000 lemmata in the combined corpus

3. **Register notes**
   - Where a word or construction is drawn from a specific register (Septuagintal, Homeric, Thucydidean), note the source and why

**Format:** JSON + optional rendered output (HTML or LaTeX). Each passage gets a `apparatus/` file linking diagram data, vocabulary glosses, and register notes to the Greek text by sentence/clause index.

**Implementation notes**
- Reed-Kellogg diagrams can be generated as structured data (nested JSON trees) and rendered via a simple script
- The morphological parse can be seeded from the validation pipeline's OdyCy/Morpheus output
- Frequency data comes from the parent project's corpus statistics

---

## Directory Structure

```
blood-meridian-translation/
  README.md
  PROJECT_PLAN.md
  glossary/
    idf_glossary.json          # standardised term translations
  passages/
    001_opening.json           # source text + metadata per passage
    002_judge_sermon.json
    ...
  bridge/
    001_opening_bridge.json    # Modern Greek bridge notes per passage
    ...
  retrieval/
    retrieve.py                # wrapper around parent project's embedding search
    config.yaml                # register weighting, corpus filters
  drafts/
    001_opening/
      primary.txt              # polytonic Greek draft
      literal.txt              # more literal alternative
      notes.json               # clause-by-clause justification
    ...
  apparatus/
    001_opening/
      diagrams.json            # Reed-Kellogg diagram data
      vocab_glosses.json       # rare-word and case glosses
      register_notes.json      # source-register annotations
    ...
  scripts/
    build_glossary.py          # IDF glossary tooling
    gt_bridge.py               # Google Translate bridge
    extract_fragment_idiom.py  # extract patterns from the MG fragment
    generate_apparatus.py      # produce diagrams + glosses from parsed output
    render_diagrams.py         # render Reed-Kellogg to HTML/LaTeX
  config/
    register_weights.yaml      # retrieval weighting by source corpus
    passage_manifest.yaml      # list of passages with status
```

---

## Workflow Summary

```
   English passage
        │
        ▼
   [WP0] Segment & tag discourse type
        │
        ├──────────────────────┐
        ▼                      ▼
   [WP1] IDF glossary     [WP2] MG bridge
   lookup for this            (fragment + GT)
   passage's terms             │
        │                      │
        └──────┬───────────────┘
               ▼
   [WP3] Embedding retrieval
   (Septuagint, NT, classical prose)
               │
               ▼
   [WP4] Constrained Greek drafting
   + morphosyntactic validation
               │
               ▼
   [WP5] Reed-Kellogg diagrams
   + vocabulary apparatus
   + register notes
               │
               ▼
        Human review
```

---

## Build Order

1. **WP1 first** — lock the glossary before anything else. Character names and gun vocabulary will propagate everywhere; changing them later is expensive.
2. **WP0 + WP2 in parallel** — passage selection and Modern Greek bridge can proceed simultaneously.
3. **WP3** — retrieval setup, extending parent project indices if needed.
4. **WP4** — drafting, starting with one or two pilot passages.
5. **WP5** — apparatus generation, once we have stable drafts to annotate.

---

## Pilot Passages (Suggested)

1. **Opening paragraph** ("See the child...") — McCarthy's most iconic cadence; short, paratactic, biblical
2. **A landscape passage** — desert description, testing elemental vocabulary
3. **A judge monologue** — testing philosophical/rhetorical register
4. **A violence set-piece** — testing action vocabulary, scalping terminology
5. **A dialogue passage** — testing speech conventions, colloquial register within the frame

---

## Risks

### McCarthy's parataxis vs. Greek hypotaxis
Greek naturally subordinates; McCarthy refuses to. The translation must resist the gravitational pull toward participial subordination and preserve the relentless coordination. Septuagintal and NT narrative Greek (with its Semitic-influenced καί-chaining) is our best ally here.

### Material culture
Guns, gunpowder, scalping have no classical equivalents. The IDF glossary must make defensible, consistent choices early. Periphrasis is usually safer than neologism.

### The judge's vocabulary
The judge speaks in a register that mixes natural philosophy, theology, and legal rhetoric. No single Greek author maps onto this; we'll need to blend Thucydidean analysis with Septuagintal cosmic vocabulary.

### Copyright
*Blood Meridian* is under copyright. Source text is held locally for private research. No source passages are committed to version control or redistributed.
