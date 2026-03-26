#!/usr/bin/env python3
"""
Automated review pipeline for Blood Meridian translations.

For each passage, runs:
  1. RETRIEVAL — lexical inspiration + register calibration
  2. GRAMMAR CHECK — heuristic rules for case, agreement, preposition governance
  3. REPORT — structured JSON with suggestions and flags

Usage:
  python3 scripts/review_pipeline.py                    # review all passages
  python3 scripts/review_pipeline.py 001_see_the_child  # review one passage
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "drafts"
PASSAGES = ROOT / "passages"
APPARATUS = ROOT / "apparatus"
REPORTS = ROOT / "review"
REPORTS.mkdir(exist_ok=True)

# Lazy imports for retrieval (heavy)
_search_loaded = False


def _ensure_search():
    global _search_loaded
    if not _search_loaded:
        sys.path.insert(0, str(ROOT))
        _search_loaded = True


# ====================================================================
# PART 1: Retrieval-informed review
# ====================================================================

def run_retrieval_review(passage_id: str, english_text: str, greek_text: str) -> dict:
    """Run retrieval queries and return structured suggestions."""
    _ensure_search()
    from retrieval.search import (
        lexical_inspiration, register_calibration,
        collocation_discovery, Scale,
    )

    results = {"passage_id": passage_id, "retrieval": {}}

    # 1a. Register calibration — where does our draft sit?
    try:
        reg_hits = register_calibration(greek_text, Scale.SENTENCE, top_k=5)
        periods = [h.chunk.period for h in reg_hits]
        sources = [f"{h.chunk.period}/{h.chunk.record_id}" for h in reg_hits]
        assessment, reg_issues = _assess_register(periods)
        results["retrieval"]["register"] = {
            "dominant_period": max(set(periods), key=periods.count) if periods else "unknown",
            "neighbours": sources,
            "assessment": assessment,
            "issues": reg_issues,
        }
    except Exception as e:
        results["retrieval"]["register"] = {"error": str(e)}

    # 1b. Lexical inspiration — key phrases from English
    key_phrases = _extract_key_phrases(english_text)
    phrase_hits = []
    for phrase in key_phrases[:3]:  # limit to top 3 to keep it fast
        try:
            hits = lexical_inspiration(phrase, Scale.PHRASE, top_k=3)
            for h in hits:
                phrase_hits.append({
                    "query": phrase,
                    "greek": h.chunk.text[:100],
                    "source": h.chunk.record_id,
                    "period": h.chunk.period,
                    "score": round(h.score, 3),
                })
        except Exception:
            pass
    results["retrieval"]["lexical_suggestions"] = phrase_hits

    return results


def _assess_register(periods: list[str]) -> tuple[str, list[dict]]:
    """Assess register and flag drift away from Koine target.

    Target: Koine structure with classical/Attic vocabulary richness.
    Acceptable neighbours: koine, hellenistic, classical, imperial.
    Flagged: strongly homeric (archaic syntax) or strongly imperial (late drift).
    """
    if not periods:
        return "no data", []

    dominant = max(set(periods), key=periods.count)
    count = periods.count(dominant)

    if count >= 4:
        label = f"strongly {dominant}"
    elif count >= 3:
        label = f"mostly {dominant}"
    else:
        label = f"mixed ({', '.join(periods)})"

    issues = []
    # Koine + classical/hellenistic are on-target
    good_periods = {"koine", "hellenistic", "classical"}
    good_count = sum(1 for p in periods if p in good_periods)

    if dominant == "homeric" and count >= 3:
        issues.append({
            "type": "register_drift",
            "severity": "info",
            "message": f"register leans Homeric ({count}/5 neighbours) — check for archaic syntax. Attic vocabulary is fine, but Homeric clause structure may be too epic for BM's Koine target.",
        })
    elif good_count <= 1 and len(periods) >= 3:
        issues.append({
            "type": "register_drift",
            "severity": "info",
            "message": f"few Koine/classical neighbours ({good_count}/5) — register may be drifting. Target is Koine structure with Attic vocabulary.",
        })

    return label, issues


def _extract_key_phrases(english: str) -> list[str]:
    """Extract distinctive phrases worth querying (skip common ones)."""
    sentences = re.split(r'[.!?]+', english)
    phrases = []
    for s in sentences:
        s = s.strip()
        if len(s.split()) > 3:
            phrases.append(s)
    # Sort by length descending (longer = more distinctive)
    phrases.sort(key=len, reverse=True)
    return phrases[:5]


# ====================================================================
# PART 2: Grammar checker (heuristic)
# ====================================================================

def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.lower()


# Preposition governance rules
PREP_RULES = {
    # preposition: (required_case, description)
    "ἐν": ("dative", "ἐν + δοτ."),
    "εν": ("dative", "ἐν + δοτ."),
    "εἰς": ("accusative", "εἰς + αἰτ."),
    "εις": ("accusative", "εἰς + αἰτ."),
    "ἐκ": ("genitive", "ἐκ + γεν."),
    "εκ": ("genitive", "ἐκ + γεν."),
    "ἐξ": ("genitive", "ἐξ + γεν."),
    "εξ": ("genitive", "ἐξ + γεν."),
    "ἀπό": ("genitive", "ἀπό + γεν."),
    "απο": ("genitive", "ἀπό + γεν."),
    "πρό": ("genitive", "πρό + γεν."),
    "προ": ("genitive", "πρό + γεν."),
    "μετά": ("genitive_or_accusative", "μετά + γεν./αἰτ."),
    "μετα": ("genitive_or_accusative", "μετά + γεν./αἰτ."),
    "παρά": ("any", "παρά + γεν./δοτ./αἰτ."),
    "ὑπέρ": ("genitive_or_accusative", "ὑπέρ + γεν./αἰτ."),
    "ὑπό": ("genitive_or_accusative", "ὑπό + γεν./αἰτ."),
    "πρός": ("accusative", "πρός + αἰτ. (Koine)"),
    "διά": ("genitive_or_accusative", "διά + γεν./αἰτ."),
    "κατά": ("genitive_or_accusative", "κατά + γεν./αἰτ."),
    "περί": ("genitive_or_accusative", "περί + γεν./αἰτ."),
    "ἐπί": ("any", "ἐπί + γεν./δοτ./αἰτ."),
}

# Common case endings (simplified, accent-stripped)
GENITIVE_ENDINGS = [
    "ου", "ης", "ων", "εως", "ους", "ας",  # singular + plural
]
DATIVE_ENDINGS = [
    "ῳ", "ᾳ", "οις", "αις", "εσι", "εσιν", "ησι", "ησιν",
]
ACCUSATIVE_ENDINGS = [
    "ον", "ην", "αν", "ους", "ας", "α", "εις", "ιν",
]

# Article forms by case (accent-stripped)
ARTICLES = {
    # Note: το and τα are ambiguous (nom/acc neuter) — listed as accusative
    # since preposition + το/τα is always accusative
    "nominative": {"ο", "η", "οι", "αι"},
    "genitive": {"του", "της", "των"},
    "dative": {"τῳ", "τω", "τη", "τοις", "ταις"},
    "accusative": {"τον", "την", "το", "τους", "τας", "τα"},
}


def _guess_case(word_norm: str) -> list[str]:
    """Guess possible cases from surface endings. Very rough."""
    cases = []
    for ending in GENITIVE_ENDINGS:
        if word_norm.endswith(ending):
            cases.append("genitive")
            break
    for ending in DATIVE_ENDINGS:
        if word_norm.endswith(ending):
            cases.append("dative")
            break
    for ending in ACCUSATIVE_ENDINGS:
        if word_norm.endswith(ending):
            cases.append("accusative")
            break
    # Check if it looks nominative (no clear oblique ending)
    if not cases:
        cases.append("nominative")
    return cases


def _article_case(art_norm: str) -> str | None:
    """Return the case of an article form, or None if not an article."""
    for case, forms in ARTICLES.items():
        if art_norm in forms:
            return case
    return None


def check_grammar(greek_text: str) -> list[dict]:
    """Run heuristic grammar checks on Greek text. Returns list of issues."""
    issues = []
    tokens = greek_text.split()
    norms = [_strip_accents(t.strip(".,·;:—–«»()[]")) for t in tokens]

    for i, (tok, norm) in enumerate(zip(tokens, norms)):
        # Check 1: Preposition governance
        prep_rule = PREP_RULES.get(norm) or PREP_RULES.get(tok.strip(".,·;"))
        if prep_rule and i + 1 < len(tokens):
            required_case, desc = prep_rule
            if required_case == "any":
                continue
            # Look at next non-article word
            next_idx = i + 1
            # Skip article
            if next_idx < len(tokens):
                art_case = _article_case(norms[next_idx])
                if art_case:
                    if required_case == "dative" and art_case != "dative":
                        issues.append({
                            "type": "preposition_case",
                            "severity": "warning",
                            "position": i,
                            "word": tok,
                            "context": " ".join(tokens[max(0,i-1):i+4]),
                            "message": f"{desc} but article looks {art_case}",
                        })
                    elif required_case == "genitive" and art_case not in ("genitive",):
                        issues.append({
                            "type": "preposition_case",
                            "severity": "warning",
                            "position": i,
                            "word": tok,
                            "context": " ".join(tokens[max(0,i-1):i+4]),
                            "message": f"{desc} but article looks {art_case}",
                        })
                    elif required_case == "accusative" and art_case not in ("accusative",):
                        issues.append({
                            "type": "preposition_case",
                            "severity": "warning",
                            "position": i,
                            "word": tok,
                            "context": " ".join(tokens[max(0,i-1):i+4]),
                            "message": f"{desc} but article looks {art_case}",
                        })

        # Check 2: Pleonastic genitives (noun + τοῦ/τῆς/τῶν + same-root noun)
        if norm in ("του", "της", "των") and i > 0 and i + 1 < len(tokens):
            prev_norm = norms[i-1]
            next_norm = norms[i+1]
            # Check if prev and next share a root (first 3+ chars)
            if len(prev_norm) > 3 and len(next_norm) > 3:
                if prev_norm[:4] == next_norm[:4]:
                    issues.append({
                        "type": "pleonastic_genitive",
                        "severity": "warning",
                        "position": i,
                        "word": tokens[i],
                        "context": " ".join(tokens[max(0,i-2):i+3]),
                        "message": f"possible pleonastic genitive: '{tokens[i-1]} {tokens[i]} {tokens[i+1]}' — same root?",
                    })

        # Check 3: Missing * on likely transliterations
        if re.match(r'[ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]', tok) and not tok.startswith("*"):
            # Check if it looks like a transliterated name (non-Greek phonology)
            if any(cluster in norm for cluster in ["ντ", "μπ", "γκ", "τσ", "τζ"]):
                if len(tok) > 3 and not re.match(r'(και|ουτε|μητε|αντι|εντος|παντ|λεοντ|οντ|αντ|εντ|υντ|κοντ)', norm):
                    issues.append({
                        "type": "unmarked_transliteration",
                        "severity": "info",
                        "position": i,
                        "word": tok,
                        "context": " ".join(tokens[max(0,i-1):i+2]),
                        "message": f"'{tok}' contains non-Greek cluster — should it be *{tok}?",
                    })

        # Check 4: Accent on enclitic/proclitic (very rough)
        # Skip — too error-prone without morphological data

    # Check 5: Neuter plural subject + plural verb (should be singular in Attic)
    # Detect: neuter plural nouns/adjectives near plural verb forms.
    # Neuter plural endings: -α, -ια, -ματα, -η (2nd decl neut pl)
    # Plural verb endings: -ουσι(ν), -ονται, -οντο, -ησαν, -αντο, -οντες
    NEUT_PL_ENDINGS = ("ματα", "ια", "εα")  # strong neuter plural markers
    NEUT_PL_WORDS_NORM = {"τα", "ταυτα", "εκεινα", "παντα", "αλλα", "πολλα"}
    PL_VERB_ENDINGS = ("ουσι", "ουσιν", "ονται", "οντο", "ουντο",
                       "ησαν", "αντο", "οντες", "ουντες",
                       "ωσι", "ωσιν", "ωνται")
    # Singular equivalents that would be correct
    SG_VERB_ENDINGS = ("ει", "εται", "ετο", "ησε", "ησεν", "ατο", "ῃ")

    for i in range(len(norms)):
        is_neut_pl = False
        # Check if this token looks like a neuter plural
        if norms[i] in NEUT_PL_WORDS_NORM:
            is_neut_pl = True
        elif any(norms[i].endswith(e) for e in NEUT_PL_ENDINGS):
            # Only flag if preceded by τά or follows a τά-like pattern
            if i > 0 and norms[i-1] in ("τα",):
                is_neut_pl = True

        if not is_neut_pl:
            continue

        # Look for a nearby plural verb (within 4 tokens)
        for j in range(i+1, min(i+5, len(norms))):
            if any(norms[j].endswith(e) for e in PL_VERB_ENDINGS):
                # Check it's not a participle (those are fine in plural)
                if any(norms[j].endswith(pe) for pe in ("οντες", "ουντες", "ομενοι", "ομεναι")):
                    continue
                issues.append({
                    "type": "neuter_plural_verb",
                    "severity": "warning",
                    "position": j,
                    "word": tokens[j],
                    "context": " ".join(tokens[max(0,i-1):j+2]),
                    "message": f"neuter plural '{tokens[i]}' + plural verb '{tokens[j]}' — should the verb be singular? (Attic rule: τὰ ζῷα τρέχει not τρέχουσι)",
                })
                break

    # Check 5b: Specific neuter-plural-verb patterns that LLMs commonly get wrong
    # Scan for τά X ἐκαλοῦντο (should be ἐκαλεῖτο), τά X ἐγένοντο (should be ἐγένετο), etc.
    COMMON_NP_ERRORS = {
        "εκαλουντο": "ἐκαλεῖτο",
        "ωνομαζοντο": "ὠνομάζετο",
        "εγενοντο": "ἐγένετο",
        "ελεγοντο": "ἐλέγετο",
        "εφαινοντο": "ἐφαίνετο",
        "ησαν": None,  # ἦσαν can be correct or wrong depending on context
    }
    for i in range(len(norms)):
        if norms[i] in ("τα",) and i + 1 < len(norms):
            # Look for verb within next few words
            for j in range(i+1, min(i+5, len(norms))):
                correction = COMMON_NP_ERRORS.get(norms[j])
                if correction:
                    issues.append({
                        "type": "neuter_plural_verb",
                        "severity": "warning",
                        "position": j,
                        "word": tokens[j],
                        "context": " ".join(tokens[max(0,i-1):j+2]),
                        "message": f"τά + '{tokens[j]}' → should be {correction} (neuter plural takes singular verb)",
                    })
                    break

    # Check 6: Suspicious ellipses (LLM truncation)
    ellipsis_count = greek_text.count("...") + greek_text.count("…")
    if ellipsis_count > 0:
        issues.append({
            "type": "suspicious_ellipsis",
            "severity": "warning",
            "position": -1,
            "word": "...",
            "context": "",
            "message": f"found {ellipsis_count} ellipsis/ellipses — likely LLM truncation. Check and replace with full text.",
        })

    return issues


# ====================================================================
# PART 2b: IDF glossary consistency check
# ====================================================================

_glossary_cache = None

def _load_glossary() -> dict:
    global _glossary_cache
    if _glossary_cache is not None:
        return _glossary_cache

    glossary_path = ROOT / "glossary" / "idf_glossary.json"
    with open(glossary_path) as f:
        raw = json.load(f)

    # Build lookup: forbidden_form_norm -> (correct_form, english, category)
    forbidden = {}
    # Build required: for locked entries, map english concept -> (correct_greek_norm, correct_greek)
    required = {}

    for cat_name, cat in raw.items():
        if cat_name.startswith("_") or not isinstance(cat, dict):
            continue
        for key, entry in cat.items():
            if not isinstance(entry, dict):
                continue
            status = entry.get("status", "")
            ag = entry.get("ancient_greek", "")
            en = entry.get("english", key)

            if status != "locked" or not ag:
                continue

            # Extract the primary Greek form (before any slash or parenthetical)
            primary_ag = ag.split("/")[0].split("(")[0].strip()
            primary_norm = _strip_accents(primary_ag.replace("*", "").replace("ὁ ", "").replace("ἡ ", "").replace("τό ", "").replace("τὸ ", "").replace("αἱ ", "").replace("οἱ ", "").strip())

            required[en] = {
                "correct_greek": primary_ag,
                "correct_norm": primary_norm,
            }

    # Specific forbidden substitutions (wrong word for a locked concept)
    forbidden_pairs = [
        # (forbidden_norm_pattern, correct_term, what_it_replaces)
        ("δικαστ", "ὁ κριτής", "the judge"),
        ("ξιφ", "ἡ μάχαιρα", "knife (ξίφος is sword, too elevated)"),
        ("δολ[λα]αρ", "τὸ ἀργύριον", "dollars"),
        ("ξενοδοχ", "τὸ πανδοκεῖον", "hotel (ξενοδοχεῖον is modern)"),
        ("καφεν", "τὸ καπηλεῖον", "tavern/café (modern loan)"),
        ("σαλουν", "τὸ καπηλεῖον", "saloon (transliteration)"),
        ("ονοσ", "ὁ ἡμίονος", "mule (ὄνος is donkey, not mule)"),
    ]

    _glossary_cache = {
        "required": required,
        "forbidden_pairs": forbidden_pairs,
    }
    return _glossary_cache


def check_glossary_consistency(greek_text: str, english_text: str) -> list[dict]:
    """Check that locked IDF glossary terms are used consistently."""
    glossary = _load_glossary()
    issues = []
    text_norm = _strip_accents(greek_text)

    # Check forbidden substitutions
    for pattern, correct, concept in glossary["forbidden_pairs"]:
        if re.search(pattern, text_norm):
            # Find the actual word in the original text
            match = re.search(pattern, text_norm)
            if match:
                # Get approximate position in original
                start = match.start()
                context_start = max(0, start - 20)
                context_end = min(len(greek_text), start + 30)
                issues.append({
                    "type": "glossary_forbidden",
                    "severity": "warning",
                    "message": f"forbidden form for '{concept}' — use {correct}",
                    "context": greek_text[context_start:context_end].strip(),
                })

    # Check that key terms appear when the English source mentions them
    en_norm = english_text.lower() if english_text else ""
    # Spot checks: only flag when the English word appears as a standalone
    # noun (preceded by article/adjective or at sentence boundary), not
    # buried inside a larger word or irrelevant phrase.
    spot_checks = [
        (r"\bthe kid\b", "παιδ|μικρ", "ὁ παῖς / ὁ μικρός"),
        (r"\bthe judge\b", "κριτ", "ὁ κριτής"),
        (r"\bpistol\b", "πιστολ", "*πιστόλιον"),
        (r"\btavern\b", "καπηλ", "τὸ καπηλεῖον"),
        (r"\bsaloon\b", "καπηλ", "τὸ καπηλεῖον"),
        (r"\bmule\b", "ημιον", "ὁ ἡμίονος"),
        (r"\bhotel\b", "πανδοκ", "τὸ πανδοκεῖον"),
        (r"\bhat\b", "πετασ|πιλ", "ὁ πέτασος"),
        (r"\bboots\b", "εμβαδ", "αἱ ἐμβάδες"),
        (r"\bwhore", "πορν", "αἱ πόρναι"),
        (r"\bpilgrim\b", "προσκυνητ", "ὁ προσκυνητής"),
        (r"\bknife\b|\bknives\b", "μαχαιρ", "ἡ μάχαιρα"),
        (r"\bdollar", "αργυρ", "τὸ ἀργύριον"),
    ]

    for en_pattern, gr_pattern, correct in spot_checks:
        if re.search(en_pattern, en_norm):
            if not re.search(gr_pattern, text_norm):
                en_match = re.search(en_pattern, en_norm).group()
                issues.append({
                    "type": "glossary_missing",
                    "severity": "info",
                    "message": f"English has '{en_match}' but Greek lacks {correct} — intentional?",
                })

    return issues


# ====================================================================
# PART 3: Orchestration
# ====================================================================

def _clean_greek(text: str) -> str:
    """Strip non-Greek metadata from draft text before analysis."""
    lines = text.split("\n")
    clean = []
    for line in lines:
        line = line.strip()
        # Skip English metadata headers
        if re.match(r'^(PASSAGE|Source:|Register:|---)', line):
            continue
        # Strip line markers like [58]
        line = re.sub(r'\[\d+\]\s*', '', line)
        if line:
            clean.append(line)
    return "\n".join(clean).strip()


def review_passage(passage_id: str) -> dict:
    """Run full review pipeline on one passage."""
    primary_path = DRAFTS / passage_id / "primary.txt"
    passage_path = PASSAGES / f"{passage_id}.json"

    if not primary_path.exists():
        return {"passage_id": passage_id, "error": "no draft found"}

    raw_text = primary_path.read_text("utf-8").strip()
    greek_text = _clean_greek(raw_text)

    # If cleaning changed the file, write the clean version back
    if greek_text != raw_text:
        primary_path.write_text(greek_text + "\n", encoding="utf-8")

    # Get English text
    english_text = ""
    if passage_path.exists():
        with open(passage_path) as f:
            pdata = json.load(f)
        english_text = pdata.get("text", "")

    report = {
        "passage_id": passage_id,
        "greek_length": len(greek_text.split()),
    }

    # Grammar check (fast, always runs)
    grammar_issues = check_grammar(greek_text)

    # Glossary consistency check
    glossary_issues = check_glossary_consistency(greek_text, english_text)
    grammar_issues.extend(glossary_issues)
    report["grammar"] = {
        "issues": grammar_issues,
        "issue_count": len(grammar_issues),
        "warnings": sum(1 for i in grammar_issues if i["severity"] == "warning"),
    }

    # Retrieval review (slower, needs indices)
    try:
        retrieval = run_retrieval_review(passage_id, english_text, greek_text)
        report["retrieval"] = retrieval.get("retrieval", {})
        # Fold register issues into grammar issues for unified reporting
        reg_issues = report["retrieval"].get("register", {}).get("issues", [])
        grammar_issues.extend(reg_issues)
        report["grammar"]["issues"] = grammar_issues
        report["grammar"]["issue_count"] = len(grammar_issues)
        report["grammar"]["warnings"] = sum(1 for i in grammar_issues if i["severity"] == "warning")
    except Exception as e:
        report["retrieval"] = {"error": str(e)}

    return report


def review_all(passage_ids: list[str] | None = None) -> list[dict]:
    """Review all passages and save reports."""
    if passage_ids is None:
        passage_ids = sorted(
            d.name for d in DRAFTS.iterdir()
            if d.is_dir() and (d / "primary.txt").exists()
        )

    reports = []
    for pid in passage_ids:
        print(f"  Reviewing {pid}...")
        report = review_passage(pid)
        reports.append(report)

        # Save individual report
        report_path = REPORTS / f"{pid}_review.json"
        with open(report_path, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Review Summary: {len(reports)} passages")
    print(f"{'='*60}")
    total_warnings = 0
    for r in reports:
        pid = r["passage_id"]
        g = r.get("grammar", {})
        n_issues = g.get("issue_count", 0)
        n_warnings = g.get("warnings", 0)
        total_warnings += n_warnings
        reg = r.get("retrieval", {}).get("register", {})
        reg_assess = reg.get("assessment", "?")
        flag = " ⚠" if n_warnings > 0 else " ✓"
        print(f"  {pid:40s} grammar:{n_warnings:2d}w  register:{reg_assess}{flag}")

    print(f"\nTotal grammar warnings: {total_warnings}")
    print(f"Reports saved to: {REPORTS}/")
    return reports


def main():
    if len(sys.argv) > 1:
        passage_ids = sys.argv[1:]
    else:
        passage_ids = None
    review_all(passage_ids)


if __name__ == "__main__":
    main()
