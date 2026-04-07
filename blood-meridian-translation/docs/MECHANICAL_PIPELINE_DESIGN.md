# Mechanical-First Translation Pipeline

## Core Idea

Instead of asking an LLM to translate from scratch (error-prone on vocabulary, morphology, syntax), build a **mechanical first draft** that is grammatically correct but stilted, then ask the LLM only to **polish** it for naturalness.

```
English source
    ↓
[1] Parse & analyse (stanza)
    ↓
[2] Vocabulary lookup (glossary + LSJ)
    ↓
[3] Construction mapping (EN pattern → GRC pattern)
    ↓
[4] Morphological generation (lemma + features → inflected form)
    ↓
[5] Assembly (word order, particles, connectives)
    ↓
Mechanical Greek draft (correct but stilted)
    ↓
[6] LLM polish (Opus/Sonnet: make it read naturally)
    ↓
Final Greek text
```

## Stage 1: Parse & Analyse

**Tool**: stanza English pipeline

For each English sentence, produce:
- Dependency tree (subject, verb, objects, modifiers)
- Clause structure (main clause, subordinate clauses, relative clauses)
- Verb features (tense, aspect, mood, voice)
- Named entities / proper nouns

**Already built**: `conditional_guide.py` identifies conditionals, temporals, purpose clauses. `build_fingerprint_index_fast.py` extracts 16-dim structural features. `label_constructions.py` tags clause types.

**New work needed**: Full sentence decomposition into an ordered list of translatable units (clauses → phrases → words).

## Stage 2: Vocabulary Lookup

For each English content word, find the Greek lemma:

1. **Locked glossary** (`glossary/idf_glossary.json`): hand-verified translations for domain-specific terms (pistol → πιστόλιον, schoolmaster → γραμματιστής, etc.)
2. **LSJ reverse lookup**: English word → Greek lemma. Build from `lsj_multi.json` by inverting the definition→lemma mapping.
3. **Proper noun table**: consistent transliteration rules (Toadvine → Τοαδβίνης, etc.)
4. **Function words**: articles, prepositions, particles mapped by grammatical role, not by English word.

**New data needed**: Inverted LSJ (English → Greek). ~90K entries give us good coverage.

**Key principle**: Every vocabulary choice is deterministic and traceable. No LLM guessing.

## Stage 3: Construction Mapping

This is where our existing infrastructure shines.

For each English clause, identify its construction type and map to the most likely Greek construction:

| English construction | Greek construction | Source |
|---|---|---|
| "if X, then Y" (future) | ἐάν + subjunctive, future indicative | `conditional_guide.py` |
| "when X, Y" (past) | ἐπεί/ὅτε + indicative | `construction_taxonomy.yaml` |
| relative clause | ὅς/ἥ/ὅ + appropriate case | `extract_parallel_constructions.py` |
| "he said X" | ἔφη + direct speech (no ὅτι for McCarthy's style) | style rule |
| comma splice | asyndeton (McCarthy-specific) | style rule |
| fragment | fragment (no added verb) | style rule |
| coordinated clauses ("and...and") | καί...καί chain | style rule |
| passive voice | passive or middle | `cond_distributions.json` P(GRC\|EN) |

**Already built**: 
- `cond_distributions.json`: P(GRC construction | EN construction) from 142K parallel pairs
- `structural_match.py`: k-NN matching for context-aware construction selection
- `construction_patterns.py`: 30+ Grew DSL patterns for Greek constructions
- `construction_taxonomy.yaml`: 40+ named constructions with examples

**New work needed**: 
- A **dispatcher** that takes a stanza parse tree and applies the appropriate construction mapping rule
- McCarthy-specific style rules (parataxis preservation, fragment handling)

## Stage 4: Morphological Generation

Given a Greek lemma + target grammatical features (case, number, gender, tense, mood, voice), generate the inflected form.

**Approach**: Morpheus in reverse. We have:
- `morpheus_cache.json` with 5K+ parsed forms — we can invert this
- Wiktionary paradigm tables (available via API)
- A rule-based generator for regular patterns (contract verbs, thematic nouns)

**Example**:
```
lemma=λαμβάνω, tense=aorist, voice=active, mood=indicative, person=3rd, number=singular
→ ἔλαβεν
```

**New work needed**:
- Build or source a morphological generator. Options:
  a. Invert our Morpheus cache (covers common forms)
  b. Use `declinator` or similar AG morphology library
  c. Build a lookup table from Wiktionary paradigms (batch download)
  d. Fine-tune a small seq2seq model on (lemma+features → form) pairs from our cache

## Stage 5: Assembly

Combine the generated words into a Greek sentence:

1. **Word order**: Apply Greek default orders (verb-initial for narrative, SOV for emphasis). McCarthy's English word order often maps well to Greek narrative order.
2. **Articles**: Add/omit based on Greek rules (no article for predicates, article for known referents).
3. **Particles**: μέν/δέ for contrast, γάρ for explanation, οὖν for consequence — chosen by discourse role, not by LLM whim.
4. **Connectives**: καί chains for McCarthy's "and...and" style. Asyndeton for comma splices.
5. **Enclitics/proclitics**: Position τε, γε, δή, ἄν correctly.

**Key principle**: This stage can be largely rule-based for McCarthy's paratactic style, which maps naturally to Septuagintal Greek narrative.

## Stage 6: LLM Polish

The LLM receives:
- The mechanical Greek draft
- The English source (for reference)
- Specific instructions: "Make this read naturally as Koine prose. Do NOT change vocabulary or grammatical constructions. Only adjust word order, add/remove particles, and smooth transitions."

This is a much simpler task than translation from scratch. The LLM is good at stylistic polishing; it's bad at getting morphology right from first principles.

**Expected improvement**: The LLM will:
- Reorder words for rhythm and emphasis
- Add discourse particles (δή, γε, μέν...δέ)
- Smooth participle placement
- Fix any mechanical awkwardness

**Expected preservation**: The LLM should NOT:
- Change vocabulary (we chose it deliberately)
- Alter grammatical constructions (we mapped them from the corpus)
- Add subordination where McCarthy has coordination
- Expand fragments into full sentences

## Implementation Plan

### Phase 1: Inverted LSJ + vocabulary pipeline
- Build English→Greek lemma lookup from LSJ
- Integrate with locked glossary and proper noun table
- Test: can we mechanically select Greek lemmas for every content word in Ch.I?

### Phase 2: Sentence decomposer
- Use stanza to parse each English sentence into clause tree
- Identify construction type per clause
- Output: ordered list of (clause_type, head_verb, arguments, modifiers)

### Phase 3: Construction dispatcher
- For each clause type, apply the appropriate Greek construction rule
- Use `cond_distributions.json` for probabilistic choices
- Use `structural_match.py` for context-aware neighbor matching
- Output: Greek clause skeleton (lemmas in target case/tense/mood)

### Phase 4: Morphological generator
- Build inflection lookup from Morpheus cache + Wiktionary
- Generate inflected forms from (lemma, features) pairs
- Output: inflected Greek words

### Phase 5: Assembler + LLM polish
- Rule-based word ordering and particle insertion
- LLM polish call (single Sonnet or Opus call per chunk)
- Mechanical checks (existing Morpheus/Grew pipeline)

## What This Gives Us

1. **Traceable vocabulary**: every Greek word choice has a documented source (glossary, LSJ, corpus)
2. **Correct morphology**: generated by rule, not guessed by LLM
3. **Appropriate constructions**: chosen from attested parallel corpus data
4. **McCarthy's style preserved**: parataxis/asyndeton/fragments enforced mechanically
5. **Cheaper**: one polish call instead of three translation calls
6. **Debuggable**: when a gloss is wrong, we can trace it to the specific lookup that failed

## Existing Code to Reuse

| Component | Existing code | Status |
|---|---|---|
| English parsing | stanza pipeline | Ready |
| Construction identification | `conditional_guide.py`, `label_constructions.py` | Ready |
| Construction probabilities | `cond_distributions.json` | Ready (from 142K pairs) |
| Structural matching | `structural_match.py`, `pair_library.py` | Ready |
| Vocabulary glossary | `glossary/idf_glossary.json` | Ready (~200 locked terms) |
| LSJ definitions | `glossary/lsj_multi.json` | Ready (90K entries) |
| Morpheus parsing | `scripts/morpheus_check.py` | Ready (fixed) |
| Construction validation | `check_constructions.py` | Ready |
| Mechanical checking | Morpheus + Grew pipeline | Ready |
| LLM polish | `translate_v6.py` Stage 2 (Opus improve) | Reusable |

## What's New to Build

1. **Inverted LSJ** (English → Greek lemma lookup)
2. **Sentence decomposer** (stanza parse → clause tree → translatable units)
3. **Construction dispatcher** (clause type → Greek construction skeleton)
4. **Morphological generator** (lemma + features → inflected form)
5. **Assembler** (words + particles + word order → draft sentence)
