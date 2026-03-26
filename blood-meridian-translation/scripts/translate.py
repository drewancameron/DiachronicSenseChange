#!/usr/bin/env python3
"""
Guided translation: structural signposts FIRST, not as revision afterthought.

For each McCarthy sentence:
  1. Decompose structure (fingerprint)
  2. Find parallel EN↔GRC pairs from corpus
  3. Build grammatical description + parallels
  4. LLM translates WITH this guidance from the start
  5. Checkers verify
  6. If issues remain: targeted revision pass

Usage:
  python3 scripts/translate.py 001_see_the_child              # translate one passage
  python3 scripts/translate.py 001_see_the_child --dry-run     # show prompt only
  python3 scripts/translate.py --all                           # translate all passages
"""

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
PASSAGES = ROOT / "passages"
DRAFTS = ROOT / "drafts"
CONFIG = ROOT / "config"
GLOSSARY = ROOT / "glossary"

sys.path.insert(0, str(SCRIPTS))

import numpy as np

# Prefer stanza index if available (more accurate), fall back to regex
STANZA_INDEX = ROOT / "models" / "fingerprint_index"
FAST_INDEX = ROOT / "models" / "fingerprint_index_fast"

_index_type = None

def _load_best_index():
    global _index_type
    stanza_meta = STANZA_INDEX / "metadata.jsonl"
    if stanza_meta.exists():
        import subprocess
        n_lines = int(subprocess.check_output(["wc", "-l", str(stanza_meta)]).split()[0])
        if n_lines >= 10000:  # use stanza if it has decent coverage
            from build_fingerprint_index import load_index as load_stanza, fingerprint_stanza, fingerprint_label
            _index_type = "stanza"
            return load_stanza()

    from build_fingerprint_index_fast import load_index as load_fast
    _index_type = "fast"
    return load_fast()


def _fingerprint_query(text: str):
    """Fingerprint a query sentence using the best available method."""
    if _index_type == "stanza":
        from build_fingerprint_index import fingerprint_stanza, fingerprint_label, _get_en_nlp
        nlp = _get_en_nlp()
        doc = nlp(text)
        if doc.sentences:
            return fingerprint_stanza(doc.sentences[0]), fingerprint_label(doc.sentences[0])
        return np.zeros(16, dtype=np.float32), {}
    else:
        from build_fingerprint_index_fast import fingerprint, label
        return fingerprint(text), label(text)


# ====================================================================
# 1. Build per-sentence structural guidance
# ====================================================================

def describe_grammar(sent_text: str, lbl: dict) -> str:
    """Describe the grammar of an English sentence in plain language."""
    parts = []

    wc = lbl["word_count"]

    # Main clause type
    if not lbl.get("speech") and wc <= 4 and sent_text.strip().endswith("."):
        # Check for imperative
        first_word = sent_text.split()[0].lower() if sent_text.split() else ""
        if first_word not in ("he", "she", "it", "they", "the", "a", "this", "that", "his", "her"):
            parts.append(f"Imperative, {wc} words")
        else:
            parts.append(f"Short declarative, {wc} words")
    elif lbl["type"] == "fragment":
        parts.append(f"Verbless fragment, {wc} words")
    elif lbl["type"] == "simple":
        parts.append(f"Simple sentence, {wc} words")
    elif lbl["type"] == "compound":
        parts.append(f"Compound sentence (multiple independent clauses), {wc} words")
    elif lbl["type"] == "complex":
        parts.append(f"Complex sentence (main + subordinate clause), {wc} words")
    elif lbl["type"] == "compound_complex":
        parts.append(f"Compound-complex sentence, {wc} words")

    # Subordinate clauses
    if lbl.get("relative"):
        parts.append(f"{lbl['relative']} relative clause(s) — preserve as ὅς/ἥ/ὅ + finite verb")
    if lbl.get("conditional"):
        parts.append(f"conditional clause (εἰ/ἐάν)")
    if lbl.get("temporal"):
        parts.append(f"temporal clause")

    # Coordination
    if lbl.get("coordination", 0) >= 2:
        parts.append(f"{lbl['coordination']} coordinations — preserve καί chain (McCarthy's parataxis)")
    elif lbl.get("coordination", 0) == 1:
        parts.append(f"one coordination (καί)")

    # Voice
    if lbl.get("passive"):
        parts.append("passive voice")

    # Speech
    if lbl.get("speech"):
        parts.append("contains direct speech verb")

    # Comma splice detection
    if ", " in sent_text and lbl["type"] in ("compound", "simple"):
        clause_count = sent_text.count(",") + 1
        if clause_count >= 2:
            parts.append("comma splice → asyndeton in Greek (no δέ)")

    return ". ".join(parts) + "."


def find_parallels(sent_text: str, features_arr, metadata, k: int = 3) -> list[dict]:
    """Find structurally similar parallel pairs."""
    q, _ = _fingerprint_query(sent_text)
    dists = np.sqrt(np.sum((features_arr - q) ** 2, axis=1))
    top_idx = np.argsort(dists)[:k * 5]

    # Diverse source selection
    _, q_lbl = _fingerprint_query(sent_text)
    results = []
    seen_sources = set()

    for idx in top_idx:
        m = metadata[idx]
        src = m["source"]
        if src in seen_sources:
            continue
        # Skip metadata-only entries
        if len(m.get("english", "")) < 15 or len(m.get("greek", "")) < 15:
            continue
        seen_sources.add(src)
        results.append({
            "distance": round(float(dists[idx]), 3),
            "english": m["english"],
            "greek": m["greek"],
            "source": m["source"],
            "label": m.get("label", {}),
            "construction_labels": m.get("construction_labels", []),
        })
        if len(results) >= k:
            break

    return results


_taxonomy = None

def _load_taxonomy() -> dict:
    """Load construction taxonomy for name → pattern/example lookup."""
    global _taxonomy
    if _taxonomy is not None:
        return _taxonomy
    import yaml
    tax_path = CONFIG / "construction_taxonomy.yaml"
    if tax_path.exists():
        raw = yaml.safe_load(open(tax_path))
        _taxonomy = {}
        for category, items in raw.items():
            for item in items:
                _taxonomy[item["name"]] = item
    else:
        _taxonomy = {}
    return _taxonomy


def _construction_guidance(labels: list[str]) -> list[str]:
    """Look up construction labels in taxonomy and return guidance lines."""
    tax = _load_taxonomy()
    lines = []
    for lbl in labels:
        entry = tax.get(lbl)
        if entry:
            lines.append(f"  → **{lbl}**: {entry['greek_pattern']}")
            if entry.get("example_grc"):
                lines.append(f"    e.g. {entry['example_grc']}")
            elif entry.get("example"):
                lines.append(f"    e.g. {entry['example']}")
        else:
            lines.append(f"  → **{lbl}**")
    return lines


def build_sentence_guidance(sent_text: str, sent_idx: int,
                             features_arr, metadata) -> str:
    """Build structural guidance for one sentence."""
    _, lbl = _fingerprint_query(sent_text)
    grammar = describe_grammar(sent_text, lbl)
    parallels = find_parallels(sent_text, features_arr, metadata, k=2)

    # Get construction labels
    try:
        from label_constructions import label_english
        en_labels = label_english(sent_text)
    except Exception:
        en_labels = []

    lines = [f'### {sent_idx}. "{sent_text[:80]}{"..." if len(sent_text) > 80 else ""}"']
    lines.append(f"Grammar: {grammar}")

    # Add named construction guidance (only when we have something helpful)
    if en_labels:
        constr_lines = _construction_guidance(en_labels)
        if constr_lines:
            lines.extend(constr_lines)

    if parallels:
        lines.append("Parallels:")
        for p in parallels:
            pl = p.get("label", {})
            en_short = p["english"][:70]
            gr_short = p["greek"][:70]
            # Check if this parallel has construction labels
            p_labels = p.get("construction_labels", [])
            label_note = ""
            if p_labels:
                grc_labels = [l for l in p_labels if not l.startswith("EN:")]
                if grc_labels:
                    label_note = f" [{', '.join(grc_labels)}]"
            lines.append(f'  "{en_short}"')
            lines.append(f'    → {gr_short}{label_note}')
            lines.append(f'    [{p["source"][:30]}]')

    lines.append("")
    return "\n".join(lines)


# ====================================================================
# 2. Build full translation prompt
# ====================================================================

def load_rules() -> str:
    rules = (CONFIG / "translation_prompt_rules.md").read_text("utf-8")
    particles = (CONFIG / "particle_guide.md").read_text("utf-8")
    return rules + "\n\n" + particles


def load_glossary(en_text: str = "") -> str:
    """Load locked glossary terms, filtered to those relevant to the passage.

    Scans the English source text for keywords that match glossary entries,
    then includes only the relevant locked terms plus any from matching
    semantic domains. This gives the LLM context-specific vocabulary rules.
    """
    glossary_path = GLOSSARY / "idf_glossary.json"
    if not glossary_path.exists():
        return ""
    data = json.load(open(glossary_path))

    en_lower = en_text.lower()
    all_locked = []
    relevant_locked = []

    for category, entries in data.items():
        if category.startswith("_") or not isinstance(entries, dict):
            continue
        for key, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("status") != "locked":
                continue
            en_term = entry.get("english", "")
            ag_term = entry.get("ancient_greek", "").replace("*", "")
            register = entry.get("register", "")
            justification = entry.get("justification", "")

            line = f"  {en_term} = {ag_term}"
            if register and register not in ("classical", "koine"):
                line += f"  [{register}]"
            all_locked.append(line)

            # Check if this term is relevant to the current passage
            keywords = [en_term.lower()] + key.replace("_", " ").lower().split()
            if any(kw in en_lower for kw in keywords if len(kw) > 2):
                # Add justification for context-specific terms
                detail = f"  {en_term} = {ag_term}"
                if justification:
                    detail += f"\n    ({justification[:120]})"
                relevant_locked.append(detail)

    if not all_locked:
        return ""

    sections = ["Preferred translations (use these unless context demands otherwise):"]
    if relevant_locked:
        sections.append("\n### Especially relevant to this passage:")
        sections.extend(relevant_locked)
        sections.append("\n### Full list:")
    sections.extend(all_locked)
    return "\n".join(sections)


def build_vocab_guidance(en_text: str) -> str:
    """Build vocabulary section from parallel corpus lookups."""
    try:
        from vocab_lookup import extract_content_words, lookup_word_in_corpus
    except ImportError:
        return ""

    content_words = extract_content_words(en_text)
    if not content_words:
        return ""

    entries = []
    for w in content_words:
        hits = lookup_word_in_corpus(w["lemma"], w["upos"], w["context"], max_results=2)
        if not hits:
            continue

        role_desc = f" — {w['context']}" if w['context'] else ""
        lines = [f"'{w['text']}' [{w['upos'].lower()}{role_desc}]:"]
        for hit in hits:
            en_short = hit["english"][:60]
            gr_short = hit["greek"][:60]
            lines.append(f"  [{hit['source'][:25]}] \"{en_short}\"")
            lines.append(f"    → {gr_short}")
        entries.append("\n".join(lines))

    if not entries:
        return ""

    return "## Vocabulary Guidance (from parallel corpus)\n\n" + "\n\n".join(entries)


# ====================================================================
# Polysemy disambiguation
# ====================================================================

# English words with multiple senses where the wrong Greek equivalent
# is a common LLM error. Each entry: word → list of (context_clue, sense, greek).
# Only the sense matching the passage context is emitted.
POLYSEMOUS = {
    "swell": [
        (["sea", "ocean", "wave", "water", "ship", "boat", "tide", "shore"],
         "sea swell, wave", "κῦμα, οἶδμα (NOT οἴδημα which is medical swelling)"),
        (["pride", "anger", "chest", "heart", "emotion"],
         "swell with emotion", "ὀγκόω, ἐπαίρω"),
        (["wound", "injury", "skin", "bruise"],
         "medical swelling", "οἴδημα, οἰδέω"),
    ],
    "float": [
        (["river", "flatboat", "lumber", "raft", "downstream"],
         "raft/flatboat", "σχεδία (NOT πλέω)"),
        (["water", "swim", "surface"],
         "float on water", "ἐπιπλέω, πλέω"),
    ],
    "port": [
        (["ship", "harbor", "harbour", "dock", "sail"],
         "harbour", "λιμήν"),
        (["city", "town", "streets", "walk"],
         "port city", "ἐμπόριον, λιμήν"),
    ],
    "harbor": [
        (["ship", "boat", "sail", "dock"],
         "harbour for ships", "λιμήν"),
        (["wolf", "wolves", "animal", "shelter", "woods", "forest"],
         "shelter/conceal", "τρέφω, κρύπτω, ὑποδέχομαι"),
    ],
    "draw": [
        (["water", "well", "bucket"],
         "draw water", "ἀντλέω, ἀρύω"),
        (["sword", "knife", "pistol", "gun", "weapon", "boot"],
         "draw a weapon", "σπάω, ἐξέλκω"),
        (["picture", "line", "sketch"],
         "draw/sketch", "γράφω"),
    ],
    "make": [
        (["way", "road", "path", "through"],
         "make one's way", "πορεύομαι, χωρέω"),
        (["fire", "camp"],
         "make a fire", "ἀνάπτω πῦρ, καίω"),
    ],
    "run": [
        (["out", "fort", "town", "expelled"],
         "run out / expelled", "ἐξελαύνω (pass: ἐξηλάθη)"),
        (["fast", "flee", "chase"],
         "run/flee", "τρέχω, φεύγω"),
    ],
    "charge": [
        (["crime", "law", "accused", "court", "arrest"],
         "legal charge", "αἰτία, ἔγκλημα"),
        (["horse", "attack", "battle", "rush"],
         "military charge", "ἐφορμή, ἐπιδρομή"),
    ],
    "break": [
        (["float", "raft", "lumber", "apart"],
         "break up (dismantle)", "διαλύω"),
        (["bone", "neck", "arm"],
         "break/fracture", "κατάγνυμι"),
    ],
    "dollar": [
        ([], "money (NEVER δολλάριον)", "ἀργύριον, στατήρ, χρήματα"),
    ],
    "dollars": [
        ([], "money (NEVER δολλάρια)", "ἀργύρια, στατῆρες, χρήματα"),
    ],
    "revival": [
        (["reverend", "preacher", "tent", "sermon", "god", "church"],
         "religious revival meeting", "ἀναζωπύρησις (NOT ἀνάστασις)"),
    ],
}


def build_domain_notes(en_text: str) -> str:
    """Scan English for polysemous words and emit the contextually correct sense."""
    import re
    en_lower = en_text.lower()
    words_in_text = set(re.findall(r'\b\w+\b', en_lower))

    notes = []
    for word, senses in POLYSEMOUS.items():
        # Match the word or common inflections (swells→swell, charges→charge)
        matched = word in words_in_text
        if not matched:
            matched = any(w.startswith(word) for w in words_in_text)
        if not matched:
            continue
        # Find best matching sense by context clues
        best_sense = None
        best_score = -1
        for clues, sense_desc, greek in senses:
            if not clues:
                # No context needed (e.g. "dollar" is always wrong)
                best_sense = (sense_desc, greek)
                best_score = 999
                break
            score = sum(1 for c in clues if c in en_lower)
            if score > best_score:
                best_score = score
                best_sense = (sense_desc, greek)

        if best_sense:
            sense_desc, greek = best_sense
            notes.append(f"  '{word}' here = {sense_desc} → {greek}")

    if not notes:
        return ""
    return "## Polysemy Warnings (check these carefully)\n" + "\n".join(notes)


# ====================================================================
# Adaptive prompt: complexity scoring and tiered prompt assembly
# ====================================================================

def score_complexity(en_text: str) -> dict:
    """Score a passage's syntactic complexity to determine prompt weight.

    Returns a dict with individual scores and an overall tier:
      'light'  — dialogue, short declaratives, philosophical speech
      'medium' — mixed narration with some subordination
      'heavy'  — long participial chains, temporal nesting, complex subordination
    """
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', en_text) if s.strip()]
    if not sents:
        return {"tier": "light", "scores": {}}

    try:
        from label_constructions import label_english
    except ImportError:
        return {"tier": "medium", "scores": {}}

    n_sents = len(sents)
    total_words = len(en_text.split())
    avg_sent_len = total_words / max(1, n_sents)

    # Count structural features
    n_relative = 0
    n_conditional = 0
    n_temporal = 0
    n_fragments = 0
    n_coord_chains = 0
    n_dialogue = 0

    for sent in sents:
        labels = label_english(sent)
        for l in labels:
            if "Relative" in l:
                n_relative += 1
            if "Contrafactual" in l or "Vivid" in l:
                n_conditional += 1
            if "Temporal" in l:
                n_temporal += 1
            if "Fragment" in l:
                n_fragments += 1
            if "Coordination Chain" in l:
                n_coord_chains += 1

        # Detect dialogue (short sentences with speech verbs or question marks)
        if len(sent.split()) < 10 and ("?" in sent or
            any(w in sent.lower() for w in ["said", "cried", "asked", "spat"])):
            n_dialogue += 1

    # Complexity score
    subordination = n_relative + n_conditional + n_temporal
    dialogue_frac = n_dialogue / max(1, n_sents)
    fragment_frac = n_fragments / max(1, n_sents)

    scores = {
        "n_sents": n_sents,
        "avg_sent_len": round(avg_sent_len, 1),
        "subordination": subordination,
        "coord_chains": n_coord_chains,
        "dialogue_frac": round(dialogue_frac, 2),
        "fragment_frac": round(fragment_frac, 2),
    }

    # Tier assignment
    if dialogue_frac > 0.3 and subordination <= 2:
        tier = "light"
    elif avg_sent_len < 12 and subordination <= 1:
        tier = "light"
    elif subordination >= 4 or n_coord_chains >= 3 or avg_sent_len > 30:
        tier = "heavy"
    else:
        tier = "medium"

    scores["tier"] = tier
    return scores


def build_translation_prompt(passage_id: str, features_arr, metadata,
                              force_tier: str = None) -> str:
    """Build an adaptive translation prompt scaled to passage complexity.

    Tiers:
      light  — register + polysemy + soft glossary. No parallel examples,
               no structural mirroring. Lets Opus find natural Greek.
      medium — adds construction labels and key rules. No parallel examples.
      heavy  — full apparatus: parallels, vocab guidance, structural mirroring.
    """
    # Load English source
    p_path = PASSAGES / f"{passage_id}.json"
    if not p_path.exists():
        return ""
    en_text = json.load(open(p_path)).get("text", "")
    if not en_text:
        return ""

    # Determine tier
    complexity = score_complexity(en_text)
    tier = force_tier or complexity.get("tier", "medium")
    print(f"    Complexity: {complexity}")
    print(f"    Prompt tier: {tier}")

    # Common to all tiers: polysemy warnings and glossary
    domain_notes = build_domain_notes(en_text)
    glossary = load_glossary(en_text)

    # --- TIER: LIGHT ---
    if tier == "light":
        prompt = f"""You are translating Cormac McCarthy's Blood Meridian into Ancient Greek (Koine register with Attic vocabulary). Use polytonic orthography.

The target register is literary Koine with classical vocabulary — draw on the idiom of the Septuagint for biblical/religious contexts, Plato for philosophical speech, Thucydides/Xenophon for narration. Choose the register that suits the speaker and context naturally.

{domain_notes}

## Vocabulary Notes (preferred, not mandatory)
{glossary}

Treat these as preferences. If context demands a different word, use it.

## English Source
{en_text}

## Instructions
1. Translate into Ancient Greek. Match McCarthy's sentence structure where Greek allows it naturally.
2. Preserve his parataxis and asyndeton — but use particles (δέ, γάρ, οὖν) where Greek genuinely needs them for readability.
3. Preserve fragments as fragments. Preserve dialogue line breaks.
4. Every word must be attestable in Morpheus/LSJ.
5. Output ONLY the Greek text.
"""
        return prompt

    # --- TIER: MEDIUM ---
    # Add construction labels and core rules, but no parallel examples
    rules = load_rules()

    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', en_text) if s.strip()]

    # Construction labels only (no parallels)
    construction_notes = []
    try:
        from label_constructions import label_english
        for i, sent in enumerate(sents, 1):
            labels = label_english(sent)
            if labels:
                short = sent[:60] + ("..." if len(sent) > 60 else "")
                construction_notes.append(f'  {i}. "{short}" — {", ".join(labels)}')
    except Exception:
        pass

    construction_guide = ""
    try:
        from conditional_guide import identify_constructions, format_for_prompt
        findings = identify_constructions(en_text)
        if findings:
            seen = set()
            unique = []
            for f in findings:
                key = (f["type"], f["text"][:40])
                if key not in seen:
                    seen.add(key)
                    unique.append(f)
            construction_guide = format_for_prompt(unique)
    except Exception:
        pass

    constr_text = "\n".join(construction_notes) if construction_notes else ""

    if tier == "medium":
        prompt = f"""You are translating Cormac McCarthy's Blood Meridian into Ancient Greek (Koine with Attic vocabulary).

## Translation Rules
{rules}

{domain_notes}

## Vocabulary Notes (preferred, not mandatory)
{glossary}

Treat locked terms as strong preferences. If context demands a different word, use it — but note your reasoning.

## Construction Labels Detected in Source
{constr_text}

{construction_guide}

## English Source
{en_text}

## Instructions
1. Translate into Ancient Greek (Koine/Attic register).
2. The construction labels show what Greek constructions to consider — use them as guidance, not rigid rules.
3. McCarthy's comma splices → asyndeton generally, but use connective particles where Greek needs them.
4. Preserve relative clauses as ὅς/ἥ/ὅ + finite verb where natural. Participles are acceptable if more idiomatic.
5. Preserve fragments as fragments.
6. Every word must be attestable in Morpheus/LSJ.
7. Output ONLY the Greek text.
"""
        return prompt

    # --- TIER: HEAVY ---
    # Construction labels + a few semantically-matched style models.
    # No per-word vocabulary lookups — they over-constrain word choice.
    # Instead: holistic style models from thematically similar classical prose.

    style_models = _find_style_models(en_text, k=4)

    prompt = f"""You are translating Cormac McCarthy's Blood Meridian into Ancient Greek (Koine with Attic vocabulary).

## Translation Rules
{rules}

{domain_notes}

## Vocabulary Notes (preferred, not mandatory)
{glossary}

## Style Models

The following passages from the classical corpus are thematically close to the passage you are translating. Let them guide your ear and register — not your dictionary. Notice their clause structure, particle usage, and how they handle similar subject matter in Greek.

{style_models}

## Construction Labels Detected in Source
{constr_text}

{construction_guide}

## English Source
{en_text}

## Instructions
1. Translate into Ancient Greek (Koine/Attic register).
2. Mirror McCarthy's sentence structure where Greek allows it naturally — but always prefer idiomatic Greek over literal English mirroring.
3. McCarthy's "and...and...and" coordination chains → preserve as καί chains. Do NOT subordinate.
4. McCarthy's comma splices → asyndeton generally. But use particles (δέ, γάρ, οὖν) where Greek genuinely needs them.
5. Preserve relative clauses as ὅς/ἥ/ὅ + finite verb where natural. Participles are acceptable if more idiomatic.
6. Preserve fragments as fragments.
7. Every word must be attestable in Morpheus/LSJ.
8. Output ONLY the Greek text, matching McCarthy's paragraph formatting.
"""
    return prompt


def _find_style_models(en_text: str, k: int = 4) -> str:
    """Find thematically similar passages from the classical corpus using embeddings.

    Returns a formatted string with k Greek passages and their sources,
    chosen for thematic (not structural) similarity to the English source.
    """
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from retrieval.search import lexical_inspiration, Scale
        hits = lexical_inspiration(en_text, Scale.SENTENCE, period_filter=None, top_k=k * 3)
    except Exception as e:
        print(f"    Style model retrieval failed: {e}")
        return "(no style models available)"

    if not hits:
        return "(no style models available)"

    # Deduplicate by author and pick diverse sources
    seen_authors = set()
    selected = []
    for hit in hits:
        author = getattr(hit.chunk, 'author', '') or ''
        if author in seen_authors:
            continue
        seen_authors.add(author)
        greek = hit.chunk.text[:200]
        source = f"{author}, {getattr(hit.chunk, 'work', '')}"
        period = getattr(hit.chunk, 'period', '')
        selected.append(f'  [{source} ({period})]:\n    {greek}')
        if len(selected) >= k:
            break

    if not selected:
        return "(no style models available)"

    return "\n\n".join(selected)


def build_revision_prompt(passage_id: str, en_text: str, grc_text: str,
                           issues: str, features_arr, metadata) -> str:
    """Build a revision prompt with structural guidance + specific issues."""
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', en_text) if s.strip()]

    guidance_parts = []
    for i, sent in enumerate(sents, 1):
        guidance_parts.append(
            build_sentence_guidance(sent, i, features_arr, metadata)
        )

    guidance = "\n".join(guidance_parts)
    rules = load_rules()
    glossary = load_glossary(en_text)

    prompt = f"""You are revising an Ancient Greek (Koine/Attic) translation of McCarthy's Blood Meridian.

## Translation Rules
{rules}

## {glossary}

## English Source
{en_text}

## Current Greek Translation
{grc_text}

## Structural Guidance
{guidance}

## Issues to Fix
{issues}

## Instructions
1. Fix the flagged issues while preserving everything that is correct.
2. Follow the structural guidance for construction choices.
3. Output ONLY the complete revised Greek text.
4. Every word must be attestable in Morpheus/LSJ.
"""
    return prompt


# ====================================================================
# 3. Call LLM
# ====================================================================

def call_llm(prompt: str, model: str = "claude-opus-4-20250514") -> str:
    import anthropic
    client = anthropic.Anthropic()
    collected = []
    with client.messages.stream(
        model=model,
        max_tokens=16384,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            collected.append(text)
    return "".join(collected).strip()


# ====================================================================
# 4. Main
# ====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("passages", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="claude-opus-4-20250514")
    args = parser.parse_args()

    # Load fingerprint index (stanza if available, else regex)
    features_arr, metadata = _load_best_index()
    print(f"Loaded fingerprint index ({_index_type}): {len(metadata)} sentences")

    if args.all:
        passage_ids = sorted(p.stem for p in PASSAGES.glob("*.json"))
    elif args.passages:
        passage_ids = args.passages
    else:
        parser.print_help()
        return

    for pid in passage_ids:
        print(f"\n{'='*60}")
        print(f"  {pid}")
        print(f"{'='*60}")

        # Check if draft already exists
        draft_path = DRAFTS / pid / "primary.txt"
        if draft_path.exists():
            existing = draft_path.read_text("utf-8").strip()
            if existing and not args.dry_run:
                print(f"  Draft exists ({len(existing)} chars). Use auto_revise.py to revise.")
                continue

        prompt = build_translation_prompt(pid, features_arr, metadata)
        if not prompt:
            print(f"  No source text found")
            continue

        if args.dry_run:
            print(prompt)
            print(f"\n  [Prompt: {len(prompt)} chars]")
            continue

        print(f"  Translating with {args.model}...")
        result = call_llm(prompt, model=args.model)

        # Save
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        draft_path.write_text(result + "\n", encoding="utf-8")
        print(f"  ✓ Saved ({len(result)} chars)")


if __name__ == "__main__":
    main()
