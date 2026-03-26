# Construction-Conditional Translation Model

## Goal

Given an English sentence with a specific syntactic construction (relative clause, conditional, coordination chain, fragment, etc.), predict the **distribution** of Ancient Greek construction types that a competent translator would use — not just the most probable, but the full probability mass.

For example:
```
P(GRC_construction | EN = relative_clause) = {
    relative_clause:      0.62,
    articular_participle: 0.24,
    genitive_absolute:    0.05,
    finite_clause:        0.09,
}
```

This distribution conditions our translation prompts: if the probability of preserving a relative clause is 62%, and the English has one, the prompt should default to keeping it — but flag the 24% articular-participle alternative as acceptable.

## Data

**Source**: 142K sentence-aligned EN↔GRC pairs in `db/diachronic.db` (alignments table joined to passages), spanning Thucydides, Herodotus, NT, Xenophon, Plutarch, drama, philosophy.

**Parsing**: Both sides parsed with stanza:
- English: `en` pipeline → `tokenize,pos,lemma,depparse`
- Greek: `grc` pipeline → `tokenize,pos,lemma,depparse`

## Architecture

### 1. Construction Taxonomy

Define construction types as **subtree patterns** over UD dependency trees.

**English constructions** (source-side):
| ID | Pattern | UD signature |
|----|---------|-------------|
| `EN_RELCL` | Relative clause | `acl:relcl` dep from noun |
| `EN_COORD` | Coordination chain | `conj` chain ≥ 2 |
| `EN_COND` | Conditional | `advcl` with `mark=if` |
| `EN_FRAG` | Fragment | No root verb |
| `EN_PASV` | Passive | `nsubj:pass` |
| `EN_INF` | Infinitive complement | `xcomp` with VerbForm=Inf |
| `EN_PURP` | Purpose clause | `advcl` with `mark=to` + Inf |
| `EN_TEMP` | Temporal clause | `advcl` with `mark=when/while/after/before` |
| `EN_PART` | Participial phrase | `advcl` with VerbForm=Part |
| `EN_DIR_SP` | Direct speech | `parataxis` or `ccomp` with speech verb |

**Greek constructions** (target-side):
| ID | Pattern | UD signature |
|----|---------|-------------|
| `GRC_RELCL` | Relative clause | `acl:relcl` with PRON child (ὅς) |
| `GRC_ART_PART` | Articular participle | Participle with DET child |
| `GRC_GEN_ABS` | Genitive absolute | Participle in Gen with Gen noun child |
| `GRC_COORD` | Coordination | `conj` chain |
| `GRC_COND_REAL` | Real conditional | `advcl` + `mark=εἰ` + Indicative |
| `GRC_COND_UNREAL` | Unreal conditional | `advcl` + `mark=εἰ` + past tense |
| `GRC_COND_FV` | Future-less-vivid | `advcl` + `mark=ἐάν` + Subjunctive |
| `GRC_FRAG` | Fragment | No root verb |
| `GRC_PASV` | Passive | Voice=Pass |
| `GRC_MID` | Middle | Voice=Mid |
| `GRC_INF_ART` | Articular infinitive | Inf with DET |
| `GRC_INF_ACC` | Acc+Inf | Inf with Acc nsubj |
| `GRC_PURP_HINA` | Purpose (ἵνα) | `advcl` + `mark=ἵνα` + Sub |
| `GRC_PURP_INF` | Purpose infinitive | bare Inf of purpose |
| `GRC_TEMP` | Temporal clause | `advcl` + temporal marker |
| `GRC_ASYNDETON` | Asyndeton | No conjunction between clauses |

### 2. Extraction Pipeline

```
For each (EN_text, GRC_text) pair in alignments:
    en_doc  = stanza_en(EN_text)
    grc_doc = stanza_grc(GRC_text)

    en_constructions  = extract_constructions(en_doc)   → [(type, subtree), ...]
    grc_constructions = extract_constructions(grc_doc)  → [(type, subtree), ...]

    # Align constructions between EN and GRC sides
    pairs = align_constructions(en_constructions, grc_constructions)

    yield pairs  → [(EN_RELCL, GRC_RELCL), (EN_COORD, GRC_COORD), ...]
```

**Construction alignment**: Within a sentence pair, match EN constructions to GRC constructions by:
1. Position (first relcl in EN → first relcl-like thing in GRC)
2. Head-word alignment (if we have word-level alignment from e.g. eflomal or cosine similarity)
3. Fallback: one-to-one in order of appearance

### 3. Conditional Distribution Model

**Base model**: Simple count-based MLE

```
P(GRC_type | EN_type) = count(EN_type, GRC_type) / count(EN_type)
```

This works well when we have enough data for each EN type. With 142K pairs, common constructions (relative clauses, coordination) will have thousands of examples.

### 4. Tree Kernel Smoothing for Rare Constructions

McCarthy's English is unusual: short paratactic fragments, bare noun phrases, "and...and...and" chains. Some of his constructions may have no exact match in our Thucydides/NT parallel corpus.

**Solution**: Define a **distance over dependency subtrees** and use kernel smoothing.

#### Tree Edit Distance

For two dependency subtrees T₁, T₂, define:

```
d(T₁, T₂) = tree_edit_distance(T₁, T₂)
```

where tree edit distance counts the minimum number of:
- Node insertion
- Node deletion
- Node relabelling (change deprel or upos)

**Efficient computation**: Use Zhang-Shasha algorithm (O(n²m²) but subtrees are small, typically < 20 nodes).

#### Kernel Smoothing

For a novel English construction `e*` not seen in training:

```
P(GRC_type | e*) = Σᵢ K(e*, eᵢ) · P(GRC_type | eᵢ) / Σᵢ K(e*, eᵢ)
```

where `K(e*, eᵢ) = exp(-d(e*, eᵢ)² / 2σ²)` is a Gaussian kernel over tree edit distance, and the sum is over all training examples `eᵢ`.

**σ selection**: Cross-validate on held-out Thucydides/NT pairs. Start with σ = 3 (allowing ~3 edit operations of smoothing).

#### Multi-Scale Decomposition

To ensure nothing is ever completely matchless, decompose sentences into **nested phrases**:

```
Sentence → Clauses → Phrases → Words
```

At each scale, extract construction patterns:

| Scale | What we extract |
|-------|----------------|
| **Sentence** | Overall construction type (simple, compound, complex, fragment) |
| **Clause** | Clause type (main, relative, temporal, conditional, purpose, result) |
| **Phrase** | NP structure (bare, articular, with participle), PP governance, VP voice/aspect |
| **Word** | Morphological choice (case, tense, mood, voice) |

The conditional distribution is computed at each scale independently:

```
P(GRC_clause | EN_clause)     — e.g., EN relative → GRC relative vs participle
P(GRC_phrase | EN_phrase)      — e.g., EN "the man who" → GRC "ὁ ἀνήρ ὅς" vs "ὁ ἀνήρ ὁ ..."
P(GRC_morpheme | EN_morpheme)  — e.g., EN past simple → GRC aorist vs imperfect
```

Even if a full McCarthy sentence has no match, its constituent clauses and phrases will have matches. The kernel smoother combines evidence across scales:

```
P_combined = α_sent · P_sentence + α_clause · P_clause + α_phrase · P_phrase
```

where α weights are proportional to the number of training matches at each scale (more data at finer scales → higher weight).

### 5. Phrase Extraction

Use stanza's dependency parse to extract subtrees:

```python
def extract_phrases(doc):
    """Extract nested phrases from a stanza parse."""
    phrases = []
    for sent in doc.sentences:
        # Sentence level
        phrases.append(('sentence', sent_to_tree(sent)))

        # Clause level: each verb and its dependents
        for word in sent.words:
            if word.upos == 'VERB' and word.deprel in ('root', 'advcl', 'acl:relcl', 'ccomp', 'xcomp'):
                clause_tree = extract_subtree(sent, word)
                clause_type = classify_clause(word, sent)
                phrases.append(('clause', clause_type, clause_tree))

        # Phrase level: each NP/PP/VP
        for word in sent.words:
            if word.upos in ('NOUN', 'PROPN') and word.deprel in ('nsubj', 'obj', 'obl', 'nmod'):
                np_tree = extract_np(sent, word)
                phrases.append(('np', np_tree))
            if word.upos == 'ADP':
                pp_tree = extract_pp(sent, word)
                phrases.append(('pp', pp_tree))

    return phrases
```

### 6. Integration with Translation Pipeline

The model produces, for each English construction in a source sentence, a **distribution card**:

```json
{
    "english_construction": "relative_clause",
    "english_text": "the creature who would carry her off",
    "distribution": {
        "relative_clause": 0.62,
        "articular_participle": 0.24,
        "genitive_absolute": 0.05,
        "other": 0.09
    },
    "recommendation": "Keep as relative clause (62% in parallel corpus)",
    "nearest_examples": [
        {"en": "the man who spoke", "grc": "ὁ ἀνὴρ ὃς ἔλεξεν", "source": "Thuc. 1.72"},
        {"en": "the woman who bore him", "grc": "ἡ γυνὴ ἡ τεκοῦσα αὐτόν", "source": "NT Luke 11:27"}
    ]
}
```

This card is injected into the translation prompt as a **signpost**:

```
For the relative clause "the creature who would carry her off":
  - 62% of parallel translations keep it as a relative clause → ὃ/ὅς + finite verb
  - 24% convert to articular participle → τὸ μέλλον ἀπάξειν
  - Recommendation: relative clause (majority + matches English structure)
  - Examples from corpus: [...]
```

### 7. Implementation Plan

| Step | Task | Data | Output |
|------|------|------|--------|
| 1 | Parse all 142K aligned pairs with stanza (EN + GRC) | alignments + passages | CoNLL-U pairs |
| 2 | Extract construction patterns at sentence/clause/phrase level | CoNLL-U pairs | construction_pairs.jsonl |
| 3 | Compute conditional distributions P(GRC\|EN) | construction_pairs | cond_distributions.json |
| 4 | Implement Zhang-Shasha tree edit distance | — | tree_distance.py |
| 5 | Build kernel smoother with cross-validated σ | construction_pairs | smoother model |
| 6 | For each BM English sentence, compute distribution cards | BM passages + model | signpost cards |
| 7 | Inject cards into auto_revise.py prompts | cards + templates | targeted prompts |

**Compute budget**: Step 1 is the bottleneck — parsing 142K × 2 sentences with stanza. At ~100 sentences/sec for English, ~50/sec for GRC, this is ~1 hour. Can be parallelised or done incrementally.

### 8. Evaluation

- **Intrinsic**: Hold out 10% of aligned pairs. For each EN construction, predict GRC construction. Measure accuracy and calibration of predicted distributions.
- **Extrinsic**: Run auto_revise on BM passages with and without construction cards. Compare checker outputs and human review quality.
- **Baseline**: Current system (rules only, no empirical distributions) vs model-informed prompts.

### 9. Extensions

- **Author-conditioned distributions**: P(GRC | EN, author=Thucydides) vs P(GRC | EN, author=NT). Since BM targets Koine/Septuagintal register, weight NT examples higher.
- **Period-conditioned**: P(GRC | EN, period=Hellenistic) for our register target.
- **Active learning**: When the human reviewer overrides the model's recommendation, update the distribution (Bayesian posterior update).
