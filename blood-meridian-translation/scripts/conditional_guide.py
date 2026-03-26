#!/usr/bin/env python3
"""
Identify conditional and modal constructions in English and recommend
the specific Greek equivalent by name.

English pattern → Greek named construction:

  CONDITIONALS:
  "if he does X"          → Future More Vivid:    ἐάν + subjunctive, future apodosis
  "if he did X"           → Present Contrafactual: εἰ + imperfect, ἄν + imperfect
  "if he had done X"      → Past Contrafactual:   εἰ + aorist, ἄν + aorist
  "if he should do X"     → Future Less Vivid:    εἰ + optative, ἄν + optative
  "if he does X" (general)→ Present General:      ἐάν + subjunctive, present apodosis

  OATHS / ASSEVERATIONS:
  "damn me if I don't"    → Oath + conditional:   ἦ μήν / νὴ Δία + εἰ μή + future/subj
  "I swear I will"        → Simple oath:          ἦ μήν + future indicative

  WISHES:
  "would that he were"    → Wish (unfulfilled):   εἴθε / εἰ γάρ + imperfect/aorist
  "may he prosper"        → Wish (fulfillable):   εἴθε + optative

  PURPOSE:
  "so that he might"      → Purpose:              ἵνα / ὅπως + subjunctive
  "lest he fall"          → Negative purpose:     μή + subjunctive

  RESULT:
  "so X that Y"           → Result:               ὥστε + infinitive (natural result)
                                                    ὥστε + indicative (actual result)

  TEMPORAL:
  "when he arrived"       → Past temporal:         ὅτε / ἐπεί + aorist indicative
  "whenever he arrives"   → General temporal:      ὅταν + subjunctive
  "before he could"       → Prior temporal:        πρίν + infinitive / πρίν ἄν + subj

  INDIRECT SPEECH:
  "he said that X"        → ὅτι/ὡς + indicative (retaining original tense)
                           → accusative + infinitive (Attic preference for some verbs)

  RELATIVE with modal force:
  "the man who would"     → ὅς + ἄν + subjunctive (indefinite)
  "the creature who would carry" → ὅς + future indicative (definite future)

Usage:
  python3 scripts/conditional_guide.py "Why damn my eyes if I wont shoot the son of a bitch"
  python3 scripts/conditional_guide.py --passage 010_reverend_dialogue
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASSAGES = ROOT / "passages"

_en_nlp = None


def _get_en_nlp():
    global _en_nlp
    if _en_nlp is None:
        import stanza
        _en_nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse',
                                   verbose=False)
    return _en_nlp


# ====================================================================
# Pattern detection
# ====================================================================

def identify_constructions(text: str) -> list[dict]:
    """Identify conditional/modal/temporal constructions in English text."""
    nlp = _get_en_nlp()
    doc = nlp(text)
    findings = []

    for sent in doc.sentences:
        words = sent.words
        word_texts = [w.text.lower() for w in words]
        sent_text = sent.text

        # --- CONDITIONALS ---
        for w in words:
            if w.deprel == "advcl":
                marker = next((c for c in words if c.head == w.id and c.deprel == "mark"), None)
                if not marker:
                    continue

                if marker.text.lower() in ("if", "unless"):
                    cond = _classify_conditional(w, marker, words, sent_text)
                    findings.append(cond)

        # --- OATHS / ASSEVERATIONS ---
        oath_patterns = [
            (r'\bdamn\b.*\bif\b', "oath_conditional"),
            (r'\bI\s+swear\b', "oath_simple"),
            (r'\bby\s+God\b.*\bif\b', "oath_conditional"),
            (r'\bso\s+help\s+me\b', "oath_simple"),
        ]
        for pat, otype in oath_patterns:
            if re.search(pat, sent_text, re.I):
                findings.append({
                    "type": otype,
                    "text": sent_text[:80],
                    "english_name": "oath with conditional" if "conditional" in otype else "oath/asseveration",
                    "greek_name": "ἦ μήν + εἰ μή + subjunctive/future" if "conditional" in otype else "ἦ μήν + future indicative",
                    "greek_pattern": "ἦ μήν εἰ μὴ [verb, aorist subjunctive]..." if "conditional" in otype else "ἦ μήν [verb, future]...",
                    "example": "ἦ μὴν εἰ μὴ κατατοξεύσω τὸν υἱὸν πόρνης (I swear if I don't shoot the son of a whore)",
                    "register_note": "Colloquial oath — use ἦ μήν for rough speech, νὴ τὸν Δία for more formal",
                })
                break

        # --- WISHES ---
        if re.search(r'\bwould\s+that\b|\bif\s+only\b|\bI\s+wish\b', sent_text, re.I):
            findings.append({
                "type": "wish",
                "text": sent_text[:80],
                "english_name": "unfulfilled wish",
                "greek_name": "εἴθε / εἰ γάρ + past tense",
                "greek_pattern": "εἴθε + imperfect (present wish) or aorist (past wish)",
                "example": "εἴθε παρῆν (would that he were here)",
            })

        # --- PURPOSE ---
        for w in words:
            if w.deprel == "advcl":
                marker = next((c for c in words if c.head == w.id and c.deprel == "mark"), None)
                if marker and marker.text.lower() in ("so", "that", "lest"):
                    findings.append({
                        "type": "purpose",
                        "text": sent_text[:80],
                        "english_name": "purpose clause",
                        "greek_name": "ἵνα / ὅπως + subjunctive",
                        "greek_pattern": "ἵνα + [verb, aorist subjunctive]",
                        "example": "ἵνα ἀποδιώξῃ (so that he might drive away)",
                    })
                    break
            if w.deprel == "xcomp" and w.upos == "VERB":
                # "in order to X" / "to X" as purpose
                feats = dict(p.split("=", 1) for p in (w.feats or "").split("|") if "=" in p)
                if feats.get("VerbForm") == "Inf":
                    # Check if this is genuinely purpose (after motion verbs, etc.)
                    head = next((h for h in words if h.id == w.head), None)
                    if head and head.upos == "VERB":
                        pass  # could be purpose, but too many false positives

        # --- RESULT ---
        if re.search(r'\bso\b.*\bthat\b|\bsuch\b.*\bthat\b', sent_text, re.I):
            findings.append({
                "type": "result",
                "text": sent_text[:80],
                "english_name": "result clause",
                "greek_name": "ὥστε + infinitive (natural) or indicative (actual)",
                "greek_pattern": "ὥστε + infinitive (more common, natural consequence)",
                "example": "ὥστε μὴ δύνασθαι ὁρᾶν (so as not to be able to see)",
            })

        # --- TEMPORAL ---
        for w in words:
            if w.deprel == "advcl":
                marker = next((c for c in words if c.head == w.id and c.deprel == "mark"), None)
                if marker and marker.text.lower() in ("when", "after", "before", "until", "while", "once"):
                    temporal = _classify_temporal(w, marker, words, sent_text)
                    findings.append(temporal)

        # --- RELATIVE with modal force ---
        for w in words:
            if w.deprel == "acl:relcl":
                # Check for modal auxiliaries (would, could, might)
                for c in words:
                    if c.head == w.id and c.upos == "AUX":
                        if c.text.lower() in ("would", "could", "might"):
                            head_noun = next((h for h in words if h.id == w.head), None)
                            findings.append({
                                "type": "modal_relative",
                                "text": sent_text[:80],
                                "english_name": f"relative clause with '{c.text}' (modal/future force)",
                                "greek_name": "ὅς + future indicative (definite) or ὅς + ἄν + subjunctive (indefinite)",
                                "greek_pattern": f"[head noun] ὅς + future indicative",
                                "example": "τὸ πλάσμα ὃ αὐτὴν ἀπάξει (the creature who would carry her off)",
                                "head_noun": head_noun.text if head_noun else "?",
                            })
                            break

        # --- INDIRECT SPEECH ---
        for w in words:
            if w.deprel == "ccomp":
                head = next((h for h in words if h.id == w.head), None)
                if head and head.lemma in ("say", "tell", "claim", "declare", "think", "believe",
                                            "know", "report", "announce", "inform"):
                    findings.append({
                        "type": "indirect_speech",
                        "text": sent_text[:80],
                        "english_name": f"indirect speech (after '{head.text}')",
                        "greek_name": "ὅτι/ὡς + indicative or accusative + infinitive",
                        "greek_pattern": "ὅτι + [original tense retained] or acc. + infinitive",
                        "example": "εἶπεν ὅτι πλάνος ἐστίν (he said that he is an imposter)",
                        "register_note": "Attic prefers acc+inf for φημί, λέγω; ὅτι for verbs of knowing/perceiving",
                    })

    return findings


def _classify_conditional(verb_word, marker, words, sent_text) -> dict:
    """Classify a conditional clause by its English tense pattern."""
    feats = dict(p.split("=", 1) for p in (verb_word.feats or "").split("|") if "=" in p)
    tense = feats.get("Tense", "")
    mood = feats.get("Mood", "")
    vform = feats.get("VerbForm", "")

    # Check for auxiliaries in the protasis
    aux = next((c for c in words if c.head == verb_word.id and c.upos == "AUX"), None)
    aux_text = aux.text.lower() if aux else ""

    # Check for past perfect (had + past participle)
    if aux_text == "had" or (tense == "Past" and vform == "Part"):
        return {
            "type": "conditional_past_contrafactual",
            "text": sent_text[:80],
            "english_name": "past contrafactual ('if he had done X, he would have done Y')",
            "greek_name": "Past Contrafactual: εἰ + aorist indicative ... ἄν + aorist indicative",
            "greek_pattern": "εἰ + [verb, aorist indic.] ... [verb, aorist indic.] + ἄν",
            "example": "εἰ ἔπραξε τοῦτο, ἔπαθεν ἄν (if he had done this, he would have suffered)",
        }

    # "if he should" / "if he were to"
    if aux_text in ("should", "were"):
        return {
            "type": "conditional_future_less_vivid",
            "text": sent_text[:80],
            "english_name": "future less vivid ('if he should do X, he would do Y')",
            "greek_name": "Future Less Vivid: εἰ + optative ... ἄν + optative",
            "greek_pattern": "εἰ + [verb, optative] ... [verb, optative] + ἄν",
            "example": "εἰ πράξειε τοῦτο, πάθοι ἄν (if he should do this, he would suffer)",
        }

    # "if he did X" (simple past — present contrafactual)
    if tense == "Past" and mood == "Ind":
        return {
            "type": "conditional_present_contrafactual",
            "text": sent_text[:80],
            "english_name": "present contrafactual ('if he did X, he would do Y')",
            "greek_name": "Present Contrafactual: εἰ + imperfect ... ἄν + imperfect",
            "greek_pattern": "εἰ + [verb, imperfect indic.] ... [verb, imperfect indic.] + ἄν",
            "example": "εἰ ἔπρασσε τοῦτο, ἔπασχεν ἄν (if he were doing this, he would be suffering)",
        }

    # "if he does X" — future more vivid or present general
    if tense == "Pres" or (not tense and mood == "Ind"):
        return {
            "type": "conditional_future_more_vivid",
            "text": sent_text[:80],
            "english_name": "future more vivid ('if he does X, he will do Y')",
            "greek_name": "Future More Vivid: ἐάν + subjunctive ... future indicative",
            "greek_pattern": "ἐάν + [verb, aorist subj.] ... [verb, future indic.]",
            "example": "ἐὰν πράξῃ τοῦτο, πείσεται (if he does this, he will suffer)",
        }

    # Default
    return {
        "type": "conditional_simple",
        "text": sent_text[:80],
        "english_name": "simple conditional",
        "greek_name": "εἰ + indicative (appropriate tense)",
        "greek_pattern": "εἰ + [verb, indicative]",
        "example": "εἰ ἔλαβον τὰς ἐμβάδας (if they got my boots)",
    }


def _classify_temporal(verb_word, marker, words, sent_text) -> dict:
    """Classify a temporal clause."""
    feats = dict(p.split("=", 1) for p in (verb_word.feats or "").split("|") if "=" in p)
    tense = feats.get("Tense", "")
    marker_text = marker.text.lower()

    if marker_text == "before":
        return {
            "type": "temporal_prior",
            "text": sent_text[:80],
            "english_name": f"temporal (before)",
            "greek_name": "πρίν + infinitive (normal) or πρίν ἄν + subjunctive (after negative main clause)",
            "greek_pattern": "πρίν + [verb, infinitive]",
            "example": "πρὶν ἐλθεῖν αὐτόν (before he came)",
        }
    elif marker_text == "until":
        return {
            "type": "temporal_until",
            "text": sent_text[:80],
            "english_name": "temporal (until)",
            "greek_name": "ἕως / μέχρι + subjunctive (future/indefinite) or indicative (past/definite)",
            "greek_pattern": "ἕως ἄν + [verb, subjunctive]",
            "example": "ἕως ἂν ἔλθῃ (until he comes)",
        }
    elif tense == "Past":
        return {
            "type": "temporal_past",
            "text": sent_text[:80],
            "english_name": f"temporal ({marker_text} + past)",
            "greek_name": f"ὅτε/ἐπεί + aorist indicative (single event) or imperfect (ongoing)",
            "greek_pattern": "ὅτε/ἐπεί + [verb, aorist indicative]",
            "example": "ὅτε ἦλθεν (when he came)",
        }
    else:
        return {
            "type": "temporal_general",
            "text": sent_text[:80],
            "english_name": f"temporal ({marker_text} — general/repeated)",
            "greek_name": "ὅταν + subjunctive (general) or ὅτε + indicative (specific)",
            "greek_pattern": "ὅταν + [verb, subjunctive]",
            "example": "ὅταν ἔλθῃ (whenever he comes)",
        }


def format_for_prompt(findings: list[dict]) -> str:
    """Format findings as prompt text."""
    if not findings:
        return ""

    lines = ["## Construction Guide\n"]
    lines.append("Named Greek constructions identified in the English source:\n")

    for f in findings:
        lines.append(f"**{f['english_name']}**: \"{f['text']}\"")
        lines.append(f"  Greek: {f['greek_name']}")
        lines.append(f"  Pattern: {f['greek_pattern']}")
        lines.append(f"  Example: {f['example']}")
        if f.get("register_note"):
            lines.append(f"  Register: {f['register_note']}")
        lines.append("")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?")
    parser.add_argument("--passage", type=str)
    args = parser.parse_args()

    if args.passage:
        p = PASSAGES / f"{args.passage}.json"
        if not p.exists():
            print(f"Not found: {p}")
            return
        text = json.load(open(p)).get("text", "")
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        return

    findings = identify_constructions(text)
    if findings:
        print(format_for_prompt(findings))
    else:
        print("No conditional/modal/temporal constructions found.")


if __name__ == "__main__":
    main()
