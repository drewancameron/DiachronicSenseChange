# Project Plan: Diachronic Sense Change Modelling for Ancient Greek

## Working title

**Scholar-grounded diachronic sense change modelling for Ancient Greek: extracting sense evidence from translations, translator’s notes, and philological scholarship to build Greek-only contextual models and macro-historical analyses**

---

## 1. Executive summary

This project will build a new research pipeline for modelling lexical sense change across the history of Ancient Greek, from Homeric Greek through Koine. Its central methodological innovation is a **third supervision strategy** for historical lexical semantics.

Existing work in this area has largely proceeded in two ways:

1. **Small expert-annotated datasets**, in which scholars label occurrences of a target word by sense.
2. **Synthetic or augmented datasets**, in which language models generate historically controlled examples to supplement scarce training data.

This project will develop a third approach:

3. **Scholarship-distilled supervision**, in which a large language model (LLM) reads aligned translations, translator’s notes, commentaries, lexicographic discussions, and older philological scholarship, and extracts from them a structured database of sense evidence for real Greek passages.

The LLM will **not** be used to invent senses or to serve as the final semantic model. Instead, it will act as an evidence-construction and corpus-management agent. The downstream statistical and machine-learning models will be trained on **Greek occurrences and labels only**. The English and scholarly apparatus will supervise the Greek model, but will not dominate the representation space.

The project has two intellectual goals:

- to create a provenance-rich database of occurrence-level sense evidence for Ancient Greek;
- to test whether Greek-only contextual embedding models can recover meaningful diachronic semantic structure and reveal macro-patterns of lexical change that are difficult to detect through close reading alone.

The long-range ambition is not merely to improve word sense disambiguation for Ancient Greek, but to build a framework for asking larger historical questions: which kinds of words change most rapidly, which semantic fields move together, which genres preserve or revive older semantic neighborhoods, and how lexical change interacts with broader linguistic developments such as shifts in mood, syntax, and register.

---

## 2. Core research questions

### RQ1. Evidence construction
Can translations, translator’s notes, commentaries, and older philological scholarship be converted into a reliable, provenance-rich database of sense evidence for Ancient Greek lexical occurrences?

### RQ2. Representation learning
Can Greek-only contextual models trained on such evidence learn occurrence spaces that distinguish:

- lexical sense,
- register,
- archaizing or historically marked usage,
- genre,
- period,
- and diachronic semantic drift?

### RQ3. Historical interpretation
Can such models reveal macro-patterns of semantic change across Ancient Greek that are not readily visible from local close reading alone?

### RQ4. Methodological comparison
How does scholarship-distilled supervision compare with:

- small expert-labeled datasets,
- translation-alignment baselines,
- and synthetic historical example generation?

### RQ5. Philological value
Can the resulting system function as a research companion for classicists and historians of language, by surfacing candidate trajectories, bridge cases, and historically marked reactivations of older senses?

---

## 3. Main hypotheses

### H1. Scholarship can be operationalized
A substantial amount of usable occurrence-level sense evidence can be extracted from existing translations, notes, lexica, and commentaries without requiring the project to hand-label every occurrence from scratch.

### H2. Greek-only models can learn more than dictionary categories
If trained on provenance-rich occurrence labels, contextual models of Greek will recover not only discrete senses but structured relationships among senses, registers, and historical layers.

### H3. Register must be modelled explicitly
A model that ignores register and historical stance will systematically confuse genuine semantic persistence with archaizing, poetic, elevated, or mock-archaic usage.

### H4. Macro-patterns will emerge only at scale
Even noisy local evidence will support meaningful higher-level discoveries once aggregated across many occurrences, words, authors, genres, and periods.

### H5. Translation should supervise, not define, the model
Better historical-semantic performance will be obtained when English translations and notes are used to create labels, but the final representation learner is trained on Greek contexts rather than bilingual text.

---

## 4. Project scope

### Chronological scope
The project will cover a long diachronic arc from **Homeric Greek to Koine Greek**. Exact sub-periodisation will remain adjustable, but a working scheme could be:

- Homeric / Archaic
- Classical
- Hellenistic
- Imperial / Second Sophistic where relevant
- Koine / early Christian corpora where rights and corpus design permit

### Lexical scope
The project will begin with a **pilot lexicon** of carefully chosen target lemmata, likely 10–20 items in the first phase, selected to represent different semantic behaviors:

- clearly polysemous words with known diachronic movement,
- words with relatively stable meanings,
- culturally central words with rich philological commentary,
- and words liable to archaizing or allusive reuse.

Examples may include words like *kosmos*, but final selection should balance frequency, philological tractability, genre spread, and availability of aligned notes and scholarship.

### Corpus scope
The project will prioritize:

- Greek texts already available in machine-readable form,
- public-domain translations and scholarship,
- older commentaries and lexica no longer under copyright,
- aligned or alignable materials where Greek passages can be linked to translations and notes.

### Rights scope
The rights strategy is simple and advantageous: the project will focus as far as possible on **older scholarship and translations that are out of copyright**, both because they are legally tractable and because much of the foundational philological work on Ancient Greek semantics belongs to the nineteenth and early twentieth centuries.

---

## 5. Project design principles

1. **The LLM constructs evidence; it does not replace philology.**
2. **All supervision must remain inspectable and provenance-rich.**
3. **The Greek model must learn from Greek text, not from English paraphrase.**
4. **Register and historical stance are first-class variables, not afterthoughts.**
5. **Ambiguity is real and must be represented explicitly.**
6. **The pipeline must support both local interpretation and macro-scale quantitative analysis.**
7. **Human adjudication is reserved for the most informative and disputed cases, not wasted on everything equally.**

---

## 6. Staged work packages

The project is structured below as a **36-month plan**, though the architecture can be scaled down to a shorter pilot if needed.

# WP1. Corpus, rights, and source architecture (Months 1–6)

## Aim
To define the corpus, establish a public-domain materials strategy, acquire texts and scholarship, and design the source architecture that will support later extraction and modelling.

## Objectives

- Identify target Greek corpora spanning Homeric to Koine Greek.
- Identify public-domain translations, commentaries, lexica, grammars, and philological studies.
- Download, normalize, and catalogue these materials.
- Separate **core text**, **footnotes**, **translator discussions**, **publisher paratext**, and **other apparatus**.
- Build a metadata schema covering date, author, genre, edition, translator, commentary type, source provenance, and rights status.

## Key tasks

### Task 1.1 Corpus inventory
Compile a master bibliography and source inventory for:

- Greek primary texts,
- English translations,
- older commentaries,
- older lexical and grammatical scholarship,
- and any aligned datasets already available.

### Task 1.2 Rights and provenance screening
Record public-domain status, edition details, source URLs, and any usage restrictions.

### Task 1.3 Automated acquisition
Use Claude Code or comparable scripted tooling to orchestrate download, file naming, version control, deduplication, and checksum logging.

### Task 1.4 Document segmentation
Develop parsers and heuristics to split sources into:

- Greek running text,
- translation running text,
- footnotes,
- translator’s prefaces,
- lexical notes,
- scholarly discussions,
- apparatus-like materials,
- and publisher or scanning noise.

### Task 1.5 Metadata and database scaffolding
Design the initial SQLite schema for storing documents, segments, passage IDs, alignments, and provenance fields.

## Deliverables

- D1.1 Source inventory and corpus bibliography
- D1.2 Rights and provenance register
- D1.3 Cleaned corpus acquisition scripts
- D1.4 Initial SQLite schema and source ingest pipeline

## Risks

- Inconsistent OCR in older scholarship
- Messy footnote formatting and scanner artefacts
- Weak or missing passage alignment across editions

## Mitigation

- Prefer born-digital or well-curated text sources where possible
- Use staged cleaning with audit logs rather than destructive preprocessing
- Preserve raw, cleaned, and structured versions side by side

---

# WP2. Passage segmentation, alignment, and evidence extraction (Months 4–12)

## Aim
To transform the acquired materials into a structured occurrence-level evidence environment in which Greek passages can be linked to translations, notes, and scholarly commentary.

## Objectives

- Segment Greek texts into passages and token-centered contexts.
- Align Greek passages to translations and linked notes.
- Use an LLM to extract structured sense-relevant evidence from translations and scholarly apparatus.
- Store evidence in an inspectable database.

## Key tasks

### Task 2.1 Passage and token segmentation
Define the unit of analysis for modelling and annotation. This will likely include:

- document,
- book/chapter/section,
- passage,
- target token occurrence,
- local Greek context window.

### Task 2.2 Translation alignment
Link Greek passages to one or more translation segments. Where exact alignment is unavailable, use sentence or clause alignment with confidence scores.

### Task 2.3 Note and commentary alignment
Link translator’s notes and commentary sections back to:

- passage level,
- token level where possible,
- or lemma level where exact anchoring is unclear.

### Task 2.4 LLM extraction prompts and controls
Design prompt templates that instruct the LLM to:

- identify which sense or senses a passage supports,
- extract the exact evidential wording from translation or note,
- distinguish direct evidence from interpretive inference,
- record uncertainty,
- identify archaizing or stylistically marked usage,
- and propose but not silently enforce new sense distinctions.

### Task 2.5 Evidence record generation
Populate a structured table with fields such as:

- occurrence ID,
- lemma,
- Greek context,
- candidate sense,
- confidence,
- alternative sense,
- evidential text span,
- evidence type (translation / note / lexicon / commentary),
- translator or commentator identity,
- register flag,
- historical stance flag,
- and extraction provenance.

## Deliverables

- D2.1 Passage and alignment framework
- D2.2 LLM extraction prompt library
- D2.3 Evidence database v1
- D2.4 QA report on extraction precision and failure modes

## Risks

- LLM hallucination or over-interpretation
- Overweighting strongly worded translators
- False confidence in weak alignments

## Mitigation

- Require evidential span citation for every extracted claim
- Store alternative interpretations rather than collapsing to one label too early
- Benchmark extraction against a hand-reviewed sample

---

# WP3. Annotation schema, sense inventory, and adjudication framework (Months 8–16)

## Aim
To define the annotation ontology that will govern the project’s sense labels, register labels, uncertainty encoding, and adjudication workflow.

## Objectives

- Develop a controlled but revisable sense inventory for the pilot lemmata.
- Represent ambiguity and bridge cases explicitly.
- Add annotation dimensions for register and historical stance.
- Define adjudication protocols for human review.

## Key tasks

### Task 3.1 Pilot lemma selection
Select an initial set of target words representing varied diachronic behaviors.

### Task 3.2 Sense inventory design
Construct lemma-specific sense inventories informed by:

- existing expert annotations,
- lexica,
- older scholarship,
- and evidence already extracted in WP2.

The inventory must remain revisable, since historical semantics often resists a rigid static taxonomy.

### Task 3.3 Register and stance schema
Define labels such as:

- ordinary / unmarked,
- poetic-elevated,
- archaizing,
- explicitly Homericizing,
- quoted or allusive,
- scholastic-philological,
- uncertain.

### Task 3.4 Ambiguity encoding
Allow one occurrence to hold:

- a primary sense,
- one or more alternative candidate senses,
- confidence values,
- and notes on whether the ambiguity appears intrinsic or merely evidential.

### Task 3.5 Adjudication workflow
Create a workflow for selective human review of:

- high-value ambiguous occurrences,
- edge cases,
- likely system failures,
- and candidate new sense distinctions.

## Deliverables

- D3.1 Annotation guidelines manual
- D3.2 Pilot lemma inventory
- D3.3 Register and stance schema
- D3.4 Adjudicated gold sample for evaluation

## Risks

- Excessively rigid sense taxonomy
- Poor inter-annotator agreement on fuzzy cases
- Overproduction of unnecessary fine-grained distinctions

## Mitigation

- Prefer layered annotation over forced singular labels
- Use pairwise similarity judgements in addition to categorical labels
- Revise inventories iteratively after pilot adjudication

---

# WP4. Greek-only representation learning and model development (Months 12–24)

## Aim
To train Greek-only contextual representation models on the occurrence-level database and test whether they recover sense, register, and diachronic structure.

## Objectives

- Build modelling datasets from the evidence database.
- Train contextual Greek models using occurrence labels.
- Compare classification, contrastive, and multi-task objectives.
- Investigate whether embedding spaces encode interpretable semantic structure.

## Key tasks

### Task 4.1 Dataset construction for modelling
Generate training and validation sets from the adjudicated and weakly supervised data. Possible task formulations include:

- sense classification,
- same-sense / different-sense prediction,
- word-in-context similarity,
- multi-label uncertainty-aware prediction,
- and joint modelling of sense plus register.

### Task 4.2 Baseline models
Implement baselines using:

- expert-labeled data only,
- translation-alignment derived data without richer note extraction,
- and optionally synthetic historical usage generation as a comparison condition.

### Task 4.3 Contextual encoder training
Train Greek-only models using local Greek contexts around target tokens. Architectures may include:

- masked-language-model-derived encoders fine-tuned for occurrence tasks,
- contrastive encoders,
- multi-task contextual models,
- or architectures inspired by recent work on interpretable transformer embedding structure.

### Task 4.4 Layered context modelling
Introduce heads or auxiliary objectives for:

- sense,
- register,
- chronology,
- and possibly genre.

This is essential to prevent a model from confusing semantic difference with stylistic elevation or historical imitation.

### Task 4.5 Embedding-space probing
Probe the learned occurrence spaces to test whether they geometrically encode:

- sense clustering,
- period separation,
- register separation,
- stable versus changing words,
- and possible reactivation of older semantic neighborhoods.

## Deliverables

- D4.1 Modelling dataset release candidate
- D4.2 Baseline comparison report
- D4.3 Greek-only contextual models v1
- D4.4 Embedding probing and interpretability report

## Risks

- Sparse labels for some words or periods
- Overfitting to genre or authorial idiosyncrasy
- Confounding between sense and register

## Mitigation

- Use hierarchical or multi-task modelling
- Downweight noisy labels and preserve confidence estimates
- Evaluate with leave-author-out, leave-genre-out, and leave-translator-out splits where possible

---

# WP5. Diachronic inference and macro-pattern analysis (Months 20–30)

## Aim
To move from occurrence-level predictions to broader historical claims about semantic change in Ancient Greek.

## Objectives

- Estimate sense prevalence trajectories over time.
- Identify words with stable, shifting, bifurcating, or revived semantic profiles.
- Analyze cross-word and cross-domain patterns.
- Relate semantic change to broader language developments.

## Key tasks

### Task 5.1 Diachronic aggregation
Model how occurrence-level labels and embeddings vary across periods, genres, and authors.

### Task 5.2 Semantic trajectory typology
Develop a typology of change patterns such as:

- persistence,
- gradual drift,
- bifurcation,
- narrowing,
- broadening,
- register confinement,
- archaizing reactivation,
- and allusive re-use.

### Task 5.3 Cross-lexeme macro-analysis
Ask whether particular kinds of words change together. Candidate dimensions include:

- cosmological terms,
- ethical and political vocabulary,
- aesthetic vocabulary,
- religious vocabulary,
- and socially embedded institutional language.

### Task 5.4 Correlation with broader linguistic developments
Test whether semantic shifts cluster with wider changes in the language, such as:

- changes in mood distribution,
- stylistic simplification or elaboration,
- syntactic developments,
- and genre transition.

### Task 5.5 Historical interpretation workshop
Conduct close-reading workshops with classicists to interpret surprising model outputs and determine whether they represent:

- genuine historical insight,
- corpus artefact,
- translator effect,
- or modelling error.

## Deliverables

- D5.1 Diachronic semantic trajectory report
- D5.2 Macro-pattern analysis paper draft
- D5.3 Historical interpretation case studies

## Risks

- Temptation to overinterpret geometric regularities
- Weak causal linkage between lexical and grammatical change
- Confounds from corpus composition

## Mitigation

- Treat macro-findings as hypotheses to be tested philologically
- Include corpus-balance diagnostics in all analyses
- Separate descriptive pattern discovery from causal claims

---

# WP6. Evaluation, validation, and robustness testing (Months 18–32)

## Aim
To evaluate the evidence extraction layer, the modelling layer, and the interpretive claims using both computational and philological criteria.

## Objectives

- Validate the evidence-construction pipeline.
- Compare supervision regimes.
- Test robustness across words, periods, translators, and genres.
- Develop evaluation protocols that reflect ancient-language realities.

## Key tasks

### Task 6.1 Intrinsic evaluation
Compare model predictions against:

- existing expert-annotated datasets,
- internally adjudicated gold sets,
- and pairwise similarity judgements where categorical labels are unstable.

### Task 6.2 Supervision regime comparison
Compare performance under:

- expert labels only,
- translation alignment only,
- scholarship-distilled supervision,
- and synthetic augmentation.

### Task 6.3 Generalization tests
Run:

- leave-period-out,
- leave-author-out,
- leave-genre-out,
- and leave-translator-out evaluations.

### Task 6.4 Ablation studies
Test the marginal value of:

- translator’s notes,
- commentary,
- lexicographic sources,
- register labels,
- confidence weighting,
- and multi-task training.

### Task 6.5 Philological validation
Ask domain experts whether model outputs support plausible historical readings and whether surfaced “discoveries” withstand close inspection.

## Deliverables

- D6.1 Evaluation protocol report
- D6.2 Robustness and ablation study report
- D6.3 Expert validation memo

## Risks

- Lack of a single unquestionable gold standard
- Divergence between computational accuracy and philological usefulness

## Mitigation

- Use plural evaluation criteria
- Report uncertainty and disagreement explicitly
- Treat expert dissent as informative rather than merely problematic

---

# WP7. Tooling, dissemination, and reusable infrastructure (Months 28–36)

## Aim
To turn the project outputs into durable research infrastructure and publishable resources.

## Objectives

- Release cleaned public-domain corpora where permissible.
- Release annotation guidelines, schema, and selected datasets.
- Release modelling code and reproducible pipelines.
- Publish interpretive and methodological findings.

## Key tasks

### Task 7.1 Dataset and schema release
Prepare a documented release of:

- schema definitions,
- adjudicated examples,
- evidence structures,
- and selected cleaned public-domain materials where legally allowed.

### Task 7.2 Reproducible pipeline release
Release code for:

- acquisition,
- cleaning,
- segmentation,
- alignment,
- evidence extraction,
- model training,
- and evaluation.

### Task 7.3 Scholarly outputs
Prepare articles on:

- the evidence-construction methodology,
- Greek-only contextual modelling,
- and macro-patterns of semantic change.

### Task 7.4 Interface or notebook toolkit
Develop a lightweight research interface or notebook workflow allowing scholars to inspect:

- occurrence-level evidence,
- competing labels,
- embedding neighborhoods,
- and diachronic trajectories.

## Deliverables

- D7.1 Public release package
- D7.2 Reproducible code repository
- D7.3 Final project report
- D7.4 Journal article submissions

---

## 7. Database design sketch

A simple but powerful implementation would use SQLite as the core working database, with export paths to Parquet or similar formats for large-scale modelling.

### Core tables

- `documents`
- `editions`
- `authors`
- `translations`
- `commentaries`
- `segments`
- `passages`
- `tokens`
- `occurrences`
- `alignments`
- `notes`
- `sense_inventory`
- `candidate_labels`
- `evidence_spans`
- `register_labels`
- `adjudications`
- `model_splits`
- `embeddings`
- `provenance_events`

### Important fields

- source type
- period
- genre
- translator identity
- commentator identity
- lemma
- surface form
- target-token offset
- Greek context window
- candidate sense
- confidence
- alternative sense
- register / stance flag
- evidence span text
- evidence source pointer
- extraction model version
- human review status

This design keeps every model label tethered to a recoverable chain of scholarly evidence.

---

## 8. Technical stack

### Orchestration
- Claude Code or equivalent coding agent for acquisition, cleaning, parser development, and pipeline automation
- Git-based version control
- structured logs and audit trails

### Storage
- SQLite as the working research database
- versioned flat files for raw and cleaned sources
- optional vector store or embedding cache for modelling experiments

### NLP / modelling
- Python-based training stack
- Greek tokenization / morphological support where available
- transformer-based contextual encoders
- contrastive and multi-task training recipes
- statistical models for diachronic prevalence estimation

### Validation and analysis
- Jupyter or Quarto notebooks
- statistical testing and visualization
- interpretable embedding probing

---

## 9. Staffing and roles

A compact team could include:

### Principal Investigator
- project direction
- historical semantics framing
- model interpretation
- supervision of all work packages

### Research Software / NLP Research Associate
- corpus acquisition and cleaning
- database engineering
- alignment and extraction pipeline
- model training infrastructure

### Classical Philology / Ancient Greek Research Associate
- lemma selection
- annotation design
- adjudication of difficult cases
- historical interpretation of outputs

### Optional support
- short-term RA support for metadata validation and source preparation
- advisory input from a classicist with commentary/lexicography expertise
- advisory input from an NLP semantic-change specialist

---

## 10. Milestones

### Month 6
- corpus inventory complete
- rights register complete
- source ingestion and cleaning pipeline running

### Month 12
- aligned passage and note framework complete
- evidence database v1 populated
- initial extraction QA complete

### Month 16
- annotation guidelines and pilot sense inventory complete
- adjudicated sample available

### Month 24
- Greek-only models trained and benchmarked
- initial embedding-space probing complete

### Month 30
- diachronic and macro-pattern analyses complete
- philological interpretation workshops complete

### Month 36
- datasets, code, papers, and final report released

---

## 11. Main risks across the whole project

### Risk 1. The extracted evidence is too noisy
**Response:** treat extraction as weak supervision, preserve confidence, and adjudicate selectively rather than assuming clean gold data.

### Risk 2. The model learns translation habits rather than Greek semantics
**Response:** restrict the final model to Greek input only; use translator identity only for evaluation and bias checks.

### Risk 3. Sense distinctions prove unstable across long periods
**Response:** use layered annotations, pairwise similarity tasks, and revisable inventories instead of a single rigid taxonomy.

### Risk 4. Macro-patterns are statistically visible but historically trivial
**Response:** integrate close-reading validation and historical interpretation workshops before making substantive claims.

### Risk 5. Corpus imbalance distorts results
**Response:** track period, genre, author, and source composition throughout; use balanced evaluation splits and explicit diagnostics.

---

## 12. Why this project matters

This project matters because it addresses a real bottleneck in the computational study of historical languages: **semantic supervision is scarce, expensive, and difficult to validate**. Ancient Greek is an especially demanding case because there are no native speakers, the historical span is vast, the philological tradition is deep, and meaning is often mediated through generations of commentary and translation.

The project’s value lies in taking that mediation seriously rather than pretending it does not exist. Instead of discarding translations and notes as secondary material, it converts them into a structured source of semantic evidence. Instead of asking an LLM to solve historical semantics directly, it uses the LLM to assemble the evidential substrate on which better philological and statistical models can be built. And instead of stopping at word-level disambiguation, it seeks broader insights into the historical organization of meaning across the Greek language.

In that sense, the project offers both:

- a practical new method for low-resource historical semantic modelling;
- and a substantive humanities research programme capable of generating new claims about Ancient Greek lexical history.

---

## 13. Immediate pilot recommendation

Before attempting the full lexical and chronological range, the project should run a **focused pilot** on a small number of carefully chosen lemmata with strong supporting scholarship and available translations/commentary.

The pilot should aim to demonstrate:

1. that the LLM can extract usable sense evidence with provenance;
2. that Greek-only models trained on that evidence outperform weaker baselines;
3. that register and historical stance labels improve diachronic interpretation;
4. and that at least one meaningful macro-pattern or historical case study emerges from the analysis.

A successful pilot would justify scaling from a small demonstrator to a broader lexicon and more ambitious historical questions.

---

## 14. Indicative methodological bibliography

- Giusfredi, F., et al. “Modelling Semantic Change in Ancient Greek with the Greek Annotated String Corpus.”
- Zafar, A. M. I., and N. Nicholls. “Embedded Bayesian Models of Diachronic Lexical Semantic Change in Ancient Greek.”
- Keersmaekers, A., et al. “Using Translation Data to Generate Training Data for Word Sense Disambiguation in Ancient Greek.”
- Cassotti, P., and N. Tahmasebi. “Sense-specific Historical Word Usage Generation.”
- Grindrod, S. H., and P. Grindrod. “Semantics in RoBERTa Token Embedding Space.”
- AGREE and related Ancient Greek lexical semantic evaluation resources.

---

## 15. Closing statement

The project is best understood as a **philology-first computational semantics programme**. Its distinctive contribution is to build a bridge between the accumulated interpretive labour of translators and scholars, and the large-scale statistical machinery now available for contextual representation learning. If successful, it will provide both a reusable infrastructure for Ancient Greek lexical semantics and a new way of asking large historical questions about how meaning changes across time.
