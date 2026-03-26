"""
Build a translation prompt from a config dict.

Each dimension controls a section of the prompt. The builder assembles
only the sections that are enabled.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "config"
GLOSSARY = ROOT / "glossary"

sys.path.insert(0, str(SCRIPTS))


def build_prompt(en_text: str, config: dict) -> str:
    """Assemble a translation prompt based on config levels."""

    sections = []

    # --- Register instruction ---
    reg = config["register_instruction"]
    if reg == "minimal":
        sections.append(
            "Translate the following into Ancient Greek (Koine register, Attic vocabulary). "
            "Polytonic orthography."
        )
    elif reg == "moderate":
        sections.append(
            "You are translating Cormac McCarthy's Blood Meridian into Ancient Greek "
            "(Koine register with Attic vocabulary). Use polytonic orthography.\n\n"
            "The target is literary prose — draw on whatever classical register suits "
            "the context: Septuagintal for religious, Platonic for philosophical, "
            "Thucydidean/Xenophontic for narration."
        )
    elif reg == "detailed":
        sections.append(
            "You are translating Cormac McCarthy's Blood Meridian into Ancient Greek "
            "(Koine register with Attic vocabulary). Use polytonic orthography.\n\n"
            "McCarthy's prose is paratactic, asyndetic, and fragment-heavy. "
            "The Greek should preserve this — resist the urge to subordinate or "
            "add connective tissue that McCarthy deliberately omits. But where Greek "
            "genuinely needs a particle (δέ for scene shifts, γάρ for explanations), "
            "use it. The register should be literary Koine: Septuagintal colouring "
            "for religious contexts, Platonic for philosophy, Thucydidean for military "
            "narration."
        )

    # --- Rules document ---
    rules_level = config["rules_document"]
    if rules_level == "core_only":
        rules_path = CONFIG / "translation_prompt_rules.md"
        if rules_path.exists():
            text = rules_path.read_text("utf-8")
            # Take only up to the word order section (core rules)
            cut = text.find("## Word Order")
            if cut > 0:
                sections.append("## Core Translation Rules\n" + text[:cut].strip())
            else:
                sections.append("## Translation Rules\n" + text)
    elif rules_level == "full":
        rules_path = CONFIG / "translation_prompt_rules.md"
        particles_path = CONFIG / "particle_guide.md"
        if rules_path.exists():
            sections.append("## Translation Rules\n" + rules_path.read_text("utf-8"))
        if particles_path.exists():
            sections.append(particles_path.read_text("utf-8"))

    # --- Structural mirroring ---
    mir = config["structural_mirroring"]
    if mir == "soft":
        sections.append(
            "## Structure Guidance\n"
            "Prefer the most natural Greek idiom. Guard against these LLM pitfalls:\n"
            "- Preserve coordination chains (and...and...and → καί chains)\n"
            "- Preserve fragments as fragments\n"
            "- Preserve direct speech structure\n"
            "- Don't over-subordinate or over-participialise\n"
            "But always prefer idiomatic Greek over slavish English mirroring."
        )
    elif mir == "hard":
        sections.append(
            "## Structure Mirroring (STRICT)\n"
            "Mirror the syntactic construction type of each English sentence:\n"
            "- Relative clause → ὅς/ἥ/ὅ + finite verb, NOT participle\n"
            "- Coordination → καί chain, NOT subordination\n"
            "- Fragment → fragment, NOT full sentence\n"
            "- Conditional → Greek conditional matching Smyth's categories\n"
            "Do NOT convert between construction types."
        )

    # --- Particle suppression ---
    part = config["particle_suppression"]
    if part == "soft":
        sections.append(
            "## Particles\n"
            "McCarthy is asyndetic — prefer asyndeton for his comma splices. "
            "But use δέ for genuine scene transitions and temporal shifts, "
            "γάρ for explanatory clauses, οὖν for conclusions. "
            "The goal is natural Greek, not zero particles."
        )
    elif part == "hard":
        sections.append(
            "## Particles (STRICT)\n"
            "No δέ unless genuine contrast. No γάρ unless directly explanatory. "
            "McCarthy's comma splices → asyndeton. His 'and' → καί only."
        )

    # --- Polysemy / vocab guidance ---
    voc = config["vocab_guidance"]
    if voc in ("polysemy_only", "polysemy_and_corpus"):
        from translate import build_domain_notes
        notes = build_domain_notes(en_text)
        if notes:
            sections.append(notes)

    if voc == "polysemy_and_corpus":
        from translate import build_vocab_guidance
        vg = build_vocab_guidance(en_text)
        if vg:
            sections.append(vg)

    # --- Glossary ---
    gl = config["glossary"]
    if gl in ("soft", "hard"):
        glossary_path = GLOSSARY / "idf_glossary.json"
        if glossary_path.exists():
            data = json.load(open(glossary_path))
            en_lower = en_text.lower()
            lines = []
            for category, entries in data.items():
                if category.startswith("_") or not isinstance(entries, dict):
                    continue
                for key, entry in entries.items():
                    if not isinstance(entry, dict) or entry.get("status") != "locked":
                        continue
                    en_term = entry.get("english", "")
                    ag_term = entry.get("ancient_greek", "").replace("*", "")
                    # Only include if relevant to passage
                    keywords = [en_term.lower()] + key.replace("_", " ").lower().split()
                    if any(kw in en_lower for kw in keywords if len(kw) > 2):
                        lines.append(f"  {en_term} = {ag_term}")

            if lines:
                label = "MUST use" if gl == "hard" else "Preferred (override if context demands)"
                sections.append(f"## Vocabulary ({label}):\n" + "\n".join(lines))

    # --- Construction labels ---
    cl = config["construction_labels"]
    if cl in ("labels_only", "labels_and_taxonomy"):
        try:
            from label_constructions import label_english
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', en_text) if s.strip()]
            label_lines = []
            for i, sent in enumerate(sents, 1):
                labels = label_english(sent)
                if labels:
                    short = sent[:60] + ("..." if len(sent) > 60 else "")
                    label_lines.append(f'  {i}. "{short}" — {", ".join(labels)}')
            if label_lines:
                sections.append("## Construction Labels\n" + "\n".join(label_lines))
        except Exception:
            pass

    if cl == "labels_and_taxonomy":
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
                sections.append(format_for_prompt(unique))
        except Exception:
            pass

    # --- Parallel examples ---
    par = config["parallel_examples"]
    if par in ("2_style_models", "4_style_models"):
        k = 2 if par == "2_style_models" else 4
        try:
            sys.path.insert(0, str(ROOT))
            from retrieval.search import lexical_inspiration, Scale
            hits = lexical_inspiration(en_text, Scale.SENTENCE, period_filter=None, top_k=k * 3)
            seen_authors = set()
            models = []
            for hit in hits:
                author = getattr(hit.chunk, 'author', '') or ''
                if author in seen_authors:
                    continue
                seen_authors.add(author)
                greek = hit.chunk.text[:200]
                source = f"{author}, {getattr(hit.chunk, 'work', '')}"
                period = getattr(hit.chunk, 'period', '')
                models.append(f"  [{source} ({period})]:\n    {greek}")
                if len(models) >= k:
                    break
            if models:
                sections.append(
                    "## Style Models\n"
                    "These thematically similar passages exemplify the register we want. "
                    "Let them guide your ear, not your dictionary.\n\n"
                    + "\n\n".join(models)
                )
        except Exception:
            pass

    elif par == "per_sentence":
        try:
            from translate import _load_best_index, build_sentence_guidance
            features_arr, metadata = _load_best_index()
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', en_text) if s.strip()]
            parts = []
            for i, sent in enumerate(sents, 1):
                parts.append(build_sentence_guidance(sent, i, features_arr, metadata))
            sections.append("## Per-Sentence Structural Parallels\n" + "\n".join(parts))
        except Exception:
            pass

    # --- Source text ---
    sections.append(f"## English Source\n{en_text}")

    # --- Instructions ---
    sections.append(
        "## Instructions\n"
        "1. Translate into Ancient Greek. Output ONLY the Greek text.\n"
        "2. Every word must be attestable in Morpheus/LSJ.\n"
        "3. Match McCarthy's paragraph formatting."
    )

    return "\n\n".join(sections)
