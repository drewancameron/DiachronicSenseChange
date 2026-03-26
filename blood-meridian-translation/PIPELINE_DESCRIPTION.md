# Structurally-Guided Machine Translation of Literary English into Ancient Greek

## A Pipeline for Corpus-Informed, Construction-Aware Translation

### Abstract

We describe a pipeline for translating literary English prose (Cormac McCarthy's *Blood Meridian*) into Ancient Greek (Koine register with Attic vocabulary), using a combination of large language models, dependency-parsed parallel corpora, empirically-mined grammar patterns, and named construction taxonomies. The pipeline addresses a fundamental problem in LLM-assisted translation into classical languages: models default to over-subordination, over-participialisation, and incorrect morphological agreement, producing Greek that is grammatically plausible but stylistically wrong. Our approach provides the LLM with structural guidance derived from 37,000 sentence-aligned English–Ancient Greek parallel pairs, a 44-construction taxonomy with detection patterns, contextual vocabulary attestation from 839,000 Greek passages, and automated grammar verification using graph-pattern matching against 417,000 tokens of gold-standard treebank data. The pipeline produces first-attempt translations that are competitive with manually reviewed versions, and automated revision passes that correctly resolve construction mismatches identified by the verification layer.

---

### 1. Introduction

Translating modern literary prose into Ancient Greek presents challenges that differ fundamentally from typical machine translation tasks. The target language is not spoken; there are no native speakers to consult; the register must be historically appropriate (Koine structure, Attic vocabulary, Septuagintal colouring for biblical allusion); and the source text—McCarthy's paratactic, asyndetic, fragment-heavy prose—has no direct precedent in the classical corpus. Large language models can produce Ancient Greek, but without structural guidance they default to patterns that, while grammatically defensible, fail to preserve the source author's syntactic signature.

This paper describes a pipeline that addresses these challenges through four innovations:

1. **Structural fingerprinting and parallel retrieval**: Each source sentence is decomposed into a structural fingerprint using dependency parsing, then matched against 37,000 pre-parsed English–Greek parallel pairs to find sentences with equivalent syntactic shape. The matched Greek translations show the translator how real classicists handled similar structures.

2. **Named construction taxonomy**: A 44-construction taxonomy (genitive absolute, accusative + infinitive, future more vivid conditional, oath-conditional, etc.) is applied to both the source English and the retrieved Greek parallels, providing the LLM with named grammatical signposts rather than abstract structural labels.

3. **Empirical grammar verification**: The output is verified against patterns mined from 417,000 tokens of gold-annotated Ancient Greek treebank data (Universal Dependencies Perseus and PROIEL corpora), using the Grew graph-pattern matching engine.

4. **Demand-driven vocabulary attestation**: Each content word is checked against a four-tier attestation hierarchy (Morpheus API → 57K-sentence retrieval corpus → 839K-passage database → manual whitelist), and contextual vocabulary suggestions are drawn from the parallel corpus with source attribution.

### 2. Architecture

The pipeline operates in two modes: **guided translation** (first attempt) and **guided revision** (repair of existing translation). Both modes use the same four-layer prompt structure.

#### 2.1 Overview

```
Source text (McCarthy English)
    │
    ├──→ Sentence segmentation
    │
    ├──→ Per-sentence structural fingerprinting (stanza dependency parse)
    │       │
    │       └──→ Nearest-neighbour retrieval against 37K pre-indexed parallel pairs
    │               │
    │               └──→ Structural parallels with Greek translations
    │
    ├──→ Construction identification (English side)
    │       │
    │       └──→ Named construction labels + taxonomy lookup
    │               (e.g., "Modal/Future Relative → ὅς + future indicative")
    │
    ├──→ Contextual vocabulary lookup
    │       │
    │       └──→ Attested Greek words for each content word in grammatical context
    │
    └──→ Prompt assembly (rules + structure + constructions + vocabulary)
            │
            └──→ LLM translation (Claude Opus)
                    │
                    └──→ Automated verification
                            ├── Morpheus morphological check (word attestation)
                            ├── Stanza construction mismatch detection
                            ├── Grew grammar pattern matching
                            └── If issues found → revision prompt → LLM fix → re-verify
```

#### 2.2 Data Sources

| Resource | Size | Purpose |
|----------|------|---------|
| Sentence-aligned EN↔GRC parallel pairs | 37,294 pairs | Structural matching, vocabulary |
| Source: Perseus Digital Library translations | — | Thucydides, Herodotus, Xenophon, Plato, Plutarch, Aristotle, Demosthenes, Sophocles, etc. |
| UD Ancient Greek treebanks (Perseus + PROIEL) | 31,001 sentences, 417,000 tokens | Grammar rule mining, Grew pattern validation |
| Greek passage database | 839,000 passages | Word attestation, vocabulary lookup |
| Morpheus API cache | 2,300+ forms | Morphological analysis, gender/case verification |
| Retrieval corpus (LXX, Homer, Herodotus, etc.) | 57,000 sentences | Corpus-based word attestation fallback |

### 3. Structural Fingerprinting and Parallel Retrieval

#### 3.1 Fingerprinting

Each English sentence is parsed with the stanza dependency parser (English UD model) and reduced to a 16-dimensional feature vector encoding:

- Word count (normalised)
- Sentence type (simple / compound / complex / compound-complex / fragment)
- Count of relative clauses (detected via `acl:relcl` dependency relation)
- Count of conditional clauses (detected via `advcl` + `mark` = "if"/"unless")
- Count of temporal clauses (detected via `advcl` + `mark` = "when"/"while"/etc.)
- Count of causal/purpose clauses
- Presence of passive voice (`nsubj:pass`)
- Count of coordination (`conj` relations)
- Fragment indicator (no finite root verb)
- Count of complement clauses (`ccomp`, `xcomp`)
- Count of parataxis relations
- Root argument count (number of distinct dependency relations from the root)

The same fingerprinting is applied to all 37,294 English sentences in our parallel corpus during a one-time indexing step. The feature vectors are stored as a NumPy array; the associated metadata (English text, Greek text, source work, period, structural labels) is stored as JSON lines.

#### 3.2 Retrieval

At translation time, the source sentence is fingerprinted and the k nearest neighbours are retrieved by Euclidean distance over the feature vectors. This is an O(n) scan over 37K vectors—effectively instant. Source diversity is enforced: at most one match per classical author, to avoid over-representation of any single translator's style.

The retrieved pairs serve as **worked examples** in the prompt: the LLM sees how a professional translator of Thucydides or Plutarch handled a structurally similar English sentence, including the specific Greek constructions chosen.

#### 3.3 Comparison with embedding-based retrieval

We tested multilingual sentence embeddings (paraphrase-multilingual-MiniLM-L12-v2) for cross-lingual EN↔GRC retrieval. The model achieves mean cosine similarity of only 0.131 on our aligned pairs—essentially random—because it was trained on Modern Greek, not Ancient Greek with polytonic orthography. Structural fingerprinting, which operates on language-independent dependency relations, avoids this problem entirely.

For verification of new alignments (extending the parallel corpus), we use an indirect approach: translate the Greek to English via Claude Haiku, then compute EN↔EN embedding similarity between the machine translation and the human translation. This achieves mean similarity of 0.632, with 79% of pairs above 0.5.

### 4. Construction Taxonomy and Labelling

#### 4.1 Taxonomy

We define 44 named Greek constructions across 10 categories, drawing on standard reference grammars (Smyth, *Greek Grammar*). Each construction has:

- A **name** (e.g., "Future More Vivid", "Genitive Absolute", "Oath-Conditional")
- An **English trigger pattern** (how to recognise it in the source)
- A **Greek pattern** (the specific Greek form: mood, tense, particles)
- An **attested example** with source attribution

The categories are:

| Category | Count | Examples |
|----------|-------|---------|
| Conditionals | 7 | Future More Vivid (ἐάν + subj.), Past Contrafactual (εἰ + aor. ind., ἄν + aor. ind.) |
| Oaths | 2 | Oath-Conditional (ἦ μήν + εἰ μή + subj.) |
| Temporal | 5 | Past Temporal (ὅτε + aor. ind.), Prior (πρίν + inf.) |
| Case constructions | 9 | Genitive Absolute, Dative of Time When, Accusative of Duration |
| Infinitives | 3 | Articular Infinitive (τό + inf.), Accusative + Infinitive |
| Purpose/Result | 5 | Purpose (ἵνα + subj.), Natural Result (ὥστε + inf.) |
| Relatives | 4 | Defining Relative (ὅς + ind.), Modal/Future Relative (ὅς + fut.) |
| Participles | 3 | Circumstantial, Supplementary, Attributive |
| Voice/Aspect | 4 | Middle Voice (interest), Aorist (punctual), Perfect (resultative) |
| Speech | 2 | Direct Speech (postposed ἔφη), Exclamation (ὦ + voc.) |

#### 4.2 Detection

**English side**: Construction labels are assigned using stanza dependency features combined with heuristic rules. For example, a "Modal/Future Relative" is detected when an `acl:relcl` dependent has a child auxiliary with lemma "would"/"could"/"might". An "Oath-Conditional" is detected by regex pattern matching ("damn...if", "I swear") combined with dependency confirmation of an `advcl` with conditional marker.

**Greek side**: Construction labels are assigned using Grew (Graph Rewriting for NLP) pattern matching against the stanza-parsed CoNLL-U output. Each construction corresponds to a declarative Grew pattern. For example, the Genitive Absolute pattern:

```
pattern { P [VerbForm=Part, Case=Gen]; S [Case=Gen, upos=NOUN|PROPN|PRON] }
```

These patterns were validated against the UD Ancient Greek treebanks (26,492 sentences), confirming expected hit rates (e.g., Genitive Absolute: 952 instances, Conditional εἰ + indicative: 515).

#### 4.3 Conservative labelling

Labels are only emitted when they would change the translation decision. Common constructions (middle voice, circumstantial participle) are suppressed as uninformative. The principle: a wrong or vague label is worse than no label. Of 44 constructions in the taxonomy, 11 are used for Greek-side labelling and 15 for English-side labelling.

### 5. Grammar Verification

#### 5.1 Multi-tier attestation

Each word in the output is checked against four sources in sequence:

1. **Morpheus API** (Perseus/Alpheios): morphological analysis with lemma, part of speech, case, gender, number. Cache of 2,300+ forms.
2. **Retrieval corpus** (57K sentences from LXX, Homer, Herodotus, Thucydides, Xenophon, Plutarch): accent-stripped surface form matching.
3. **Diachronic database** (839K passages): exact substring search in `greek_text` field.
4. **Manual whitelist**: known Morpheus false negatives (e.g., ῥακῶν, ἐπῳάζεται).

A word flagged as "unattested" by all four tiers is likely invented by the LLM and should be replaced.

#### 5.2 Grew-based grammar checking

The Grew graph-pattern matching engine operates on stanza-parsed CoNLL-U output of the translation. Patterns check:

- **Preposition governance**: 49 prepositions with empirically-determined case requirements (mined from 417K tokens of treebank data). E.g., ἐν requires dative (99.9% in treebank).
- **Verb government**: 216 verbs with non-accusative primary objects (genitive or dative), mined from treebank. E.g., μάχομαι takes dative (65% of 147 attestations).
- **Neuter plural agreement**: Neuter plural subjects require singular verbs (Attic rule). Pattern detects violations.
- **Article-noun agreement**: Case, gender, number agreement between determiners and nouns (demoted to "noisy" severity due to stanza parse error rate of 7-17%).

A Morpheus cross-reference layer corrects stanza's morphological feature errors (175 corrections across 14 passages) using the Morpheus API as ground truth for gender, case, and number.

#### 5.3 Construction mismatch detection

The stanza-based construction checker parses both English source and Greek output, comparing:

- Relative clause count (English has N, Greek has M → flag if M < N, indicating conversion to participle)
- Coordination density (McCarthy's "and...and...and" chains should be preserved as καί chains)
- Conditional count
- Fragment count (McCarthy's verbless sentences should remain verbless)

### 6. Prompt Architecture

#### 6.1 Four-layer structure

The translation prompt comprises four layers, each addressing a different aspect of the translation task:

**Layer 1: Translation Rules** (~2,500 tokens)
Standing rules that apply to every passage: sentence structure mirroring, neuter plural agreement, time expression case usage, word order principles, vocabulary constraints, register specification (Koine/Attic), and a particle style guide specific to McCarthy's asyndetic prose.

**Layer 2: Structural Guidance** (~100-500 tokens per sentence)
For each source sentence: a grammatical description ("Complex sentence, 7 words, 1 relative clause"), the named construction label if applicable ("Defining Relative → ὅς + indicative"), and 2-3 structurally matched parallel pairs from the corpus with Greek translations and source attribution.

**Layer 3: Construction Guide** (~50-200 tokens per detected construction)
Named Greek constructions identified in the source, with the specific Greek pattern and an attested example. Only emitted for constructions that would change the translation decision.

**Layer 4: Vocabulary Guidance** (~50-100 tokens per content word)
For each non-trivial content word: attested Greek translations from the parallel corpus, contextualised by grammatical role (e.g., "'father' [noun, as subject]: Thucydides ὁ τοῦ Σιτάλκου πατήρ; Plutarch ὁ Ξέρξου πατήρ").

#### 6.2 Revision prompt variant

The revision prompt adds the current Greek translation and specific issues identified by the verification layer, but retains the same four-layer guidance structure. This ensures the LLM has the same quality of information for repair as for first-attempt translation.

### 7. Evaluation

#### 7.1 First-attempt translation quality

We compare first-attempt translations (with the four-layer prompt) against manually reviewed and corrected translations for the same passages. On passage 003 (*The mother dead these fourteen years...*), the guided first attempt:

- Correctly preserves the modal/future relative clause (ὃ αὐτὴν ἀπάξει) that required manual correction in the unguided version
- Correctly uses asyndeton where the unguided version inserted τε
- Preserves McCarthy's verbless final sentence as a participle phrase rather than expanding to a finite verb
- Produces a creative metaphor (νεοττεύει, "nests", for "broods") that arguably improves on the reviewed version

#### 7.2 Automated revision effectiveness

On passage 010 (*Reverend Green dialogue*), the revision pass:

- Correctly converted 4 articular participles to relative clauses in a single pass, guided by the "Defining Relative" construction label
- The oath-conditional "damn my eyes if I wont shoot" was correctly rendered as ἦ μὴν εἰ μὴ κατατοξεύσω on first attempt when the construction guide was present; without it, the LLM produced garbled double-εἰ μή

#### 7.3 Grammar verification results

Across all 14 passages of Chapter I:

| Metric | Before pipeline | After pipeline |
|--------|----------------|---------------|
| Morpheus unattested word warnings | 55 | 3 |
| Grew grammar violations | 4 | 3 (stanza parse noise) |
| Review pipeline grammar warnings | 0 | 0 |
| Construction mismatches (relative clause loss) | 13 | 12 |

### 8. Discussion

#### 8.1 The structural matching advantage

The key insight of this pipeline is that **structural similarity between sentences is a better retrieval signal than semantic similarity** for translation guidance. When translating "He turned to the man who spoke," the most useful reference is not a sentence about turning or speaking, but a sentence with the same dependency structure (main clause + short defining relative clause, ~7 words). The structural fingerprint captures this; semantic embeddings do not, especially across a 2,400-year language gap.

#### 8.2 Named constructions as LLM steering

LLMs have implicit knowledge of Ancient Greek grammar but apply it unreliably. They know that ἦ μήν introduces an oath, but without explicit activation they default to simpler patterns. The construction taxonomy serves as a **retrieval cue for the model's own knowledge**: naming "Oath-Conditional" and providing the pattern ἦ μήν + εἰ μή is sufficient to activate correct generation.

#### 8.3 Conservative labelling

The decision to label conservatively—only when it changes the translation—proved critical. Early versions of the pipeline produced 20+ labels per passage, many trivial ("Simple Coordination", "Middle Voice"). These diluted the signal and occasionally misled the model. The current 11 Greek-side and 15 English-side high-value labels focus on constructions where the LLM is known to make errors without guidance.

#### 8.4 Stanza limitations for Ancient Greek

The stanza AG dependency parser has significant limitations: it mislabels the gender of first-declension masculine nouns (κριτής parsed as feminine), misidentifies case forms (σκότους parsed as accusative instead of genitive), and misattaches articles to the wrong nouns (7-17% error rate in gold treebanks). The Morpheus cross-reference layer mitigates these errors for morphological features, but structural misattachment remains a source of noise in construction detection.

#### 8.5 The parallel corpus as a living resource

The pipeline's parallel corpus grows incrementally. Each verified translation pair (whether from the existing Perseus alignments or created on demand via Haiku translation + embedding verification) is cached for future use. As more passages are translated and reviewed, the structural fingerprint index and vocabulary lookup improve.

### 9. Conclusion

We have demonstrated that structurally-guided prompting, informed by a large parallel corpus and a named construction taxonomy, substantially improves LLM translation into Ancient Greek. The pipeline reduces the manual review burden by catching construction mismatches, morphological errors, and vocabulary inventions automatically, while providing the LLM with the specific grammatical knowledge needed for each source sentence. The approach is generalisable to other classical language pairs where parallel corpora and dependency parsers exist.

### Appendix A: Software

All code is available in the `blood-meridian-translation/` directory of the project repository. Key scripts:

| Script | Purpose |
|--------|---------|
| `make.py` | Pipeline orchestrator (modes: translate, revise, all) |
| `translate.py` | Four-layer guided translation prompt builder + LLM caller |
| `auto_revise.py` | Four-layer guided revision prompt builder + verify loop |
| `build_fingerprint_index.py` | Stanza-based structural fingerprint indexer (37K pairs) |
| `label_constructions.py` | Unified EN/GRC construction labeller (taxonomy + Grew patterns) |
| `conditional_guide.py` | English conditional/temporal/modal construction identifier |
| `vocab_lookup.py` | Contextual vocabulary retrieval from parallel corpus |
| `describe_structure.py` | Human-readable grammatical descriptions (EN and GRC) |
| `tree_decompose.py` | Nested hierarchical tree decomposition of UD parses |
| `grammar_engine.py` | Stanza-based grammar rule checker (YAML rules + mined rules) |
| `grew_check.py` | Grew graph-pattern grammar checker (UD treebank-backed) |
| `morpheus_check.py` | Morpheus API word attestation checker (4-tier) |
| `mine_grammar_rules.py` | Empirical grammar pattern miner from UD treebanks |
| `check_constructions.py` | EN↔GRC construction mismatch detector |

### Appendix B: Construction Taxonomy

The full 44-construction taxonomy is defined in `config/construction_taxonomy.yaml`. Grew detection patterns are in `config/construction_patterns.py`. The taxonomy draws on Smyth's *Greek Grammar* (sections 900–2687) but is structured for machine detection rather than pedagogical exposition.
