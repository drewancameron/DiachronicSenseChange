# Blood Meridian: Ancient Greek Translation

## Purpose

Translate passages from Cormac McCarthy's *Blood Meridian, or The Evening Redness in the West* (1985) into Ancient Greek, leveraging the diachronic sense-change and Atticising translation infrastructure developed in the parent project.

McCarthy's prose — paratactic, archaic in register, dense with biblical and classical allusion — makes *Blood Meridian* an unusually well-suited candidate for classical-language translation. The text's own stylistic debts to Herodotus, the King James Bible, and epic narrative overlap significantly with the linguistic registers our existing models already handle.

## Relationship to Parent Project

This subproject builds on the machinery in the main repository:

| Component | Location | Reuse |
|-----------|----------|-------|
| Diachronic sense-change models | `../models/` | Period-aware lexical selection — choose classical-era senses over later ones |
| Sense inventory & embeddings | `../config/sense_inventory_clean.json`, `../models/embeddings/` | Disambiguate target Greek vocabulary; find nearest attested sense for novel concepts |
| Atticising translation pipeline | `../subproject/` | Core retrieval + morphosyntax-aware generation + validation loop |
| Corpus & retrieval DB | `../corpus/`, `../db/` | Parallel-passage retrieval for stylistic grounding |
| Morphology / Wiktionary forms | `../config/wiktionary_forms.json` | Inflection and form generation |
| Pilot lemmata & expansion cycle | `../config/pilot_lemmata.yaml`, `../scripts/` | Vocabulary expansion workflow when McCarthy's lexis falls outside current coverage |

## What This Subproject Adds

- **Source text management** — passage selection, segmentation, and alignment for *Blood Meridian*
- **Domain-specific lexicon** — McCarthy's vocabulary (scalp-hunting, desert geography, archaic Americana) mapped to Greek equivalents with scholarly justification
- **Register calibration** — tuning output style to match the source's parataxis, asyndeton, and epic cadence rather than defaulting to neutral Attic prose
- **Translation artifacts** — drafts, commentary, and revision history for each passage

## Project Plan

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full work-package breakdown, build order, and workflow diagram.

## Getting Started

```
cd /Users/ec4029/Documents/Code/DiachronicSenseChange/blood-meridian-translation
```

Refer to the parent project's `requirements.txt` and `subproject/pyproject.toml` for dependencies.
