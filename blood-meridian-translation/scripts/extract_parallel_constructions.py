#!/usr/bin/env python3
"""
Step 1: Parse aligned EN↔GRC sentence pairs and extract construction patterns.

Reads 142K aligned pairs from the diachronic database, parses both sides
with stanza, and extracts construction features at sentence/clause/phrase
levels. Outputs construction_pairs.jsonl for the distribution model.

Usage:
  python3 scripts/extract_parallel_constructions.py              # full run
  python3 scripts/extract_parallel_constructions.py --limit 1000 # dev run
  python3 scripts/extract_parallel_constructions.py --resume     # continue from checkpoint
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT.parent / "db" / "diachronic.db"
OUTPUT_DIR = ROOT / "models" / "construction_model"
PAIRS_PATH = OUTPUT_DIR / "construction_pairs.jsonl"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.json"

# Lazy stanza pipelines
_en_nlp = None
_grc_nlp = None


def _get_en_nlp():
    global _en_nlp
    if _en_nlp is None:
        import stanza
        _en_nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse',
                                   verbose=False)
    return _en_nlp


def _get_grc_nlp():
    global _grc_nlp
    if _grc_nlp is None:
        import stanza
        _grc_nlp = stanza.Pipeline('grc', processors='tokenize,pos,lemma,depparse',
                                    verbose=False)
    return _grc_nlp


# ====================================================================
# Construction extraction
# ====================================================================

def extract_constructions(doc, lang: str) -> list[dict]:
    """Extract construction patterns at multiple scales from a stanza doc."""
    constructions = []

    for sent in doc.sentences:
        words = sent.words
        word_map = {w.id: w for w in words}

        # --- Sentence level ---
        has_root_verb = any(w.deprel == "root" and w.upos == "VERB" for w in words)
        n_conj = sum(1 for w in words if w.deprel == "conj")

        if not has_root_verb:
            constructions.append({
                "scale": "sentence",
                "type": "fragment",
                "text": sent.text[:100],
            })

        # --- Clause level ---
        for w in words:
            # Relative clauses
            if w.deprel in ("acl:relcl", "acl"):
                head = word_map.get(w.head)
                # Check for relative pronoun child
                has_rel_pron = any(
                    c.upos == "PRON" and c.head == w.id
                    for c in words
                )
                if lang == "en":
                    constructions.append({
                        "scale": "clause",
                        "type": "relative_clause",
                        "head_word": head.text if head else "?",
                        "verb": w.text,
                        "text": sent.text[:100],
                        "subtree_deprels": _get_subtree_deprels(words, w),
                    })
                else:  # grc
                    if has_rel_pron:
                        constructions.append({
                            "scale": "clause",
                            "type": "relative_clause",
                            "verb": w.text,
                            "text": sent.text[:100],
                            "subtree_deprels": _get_subtree_deprels(words, w),
                        })
                    else:
                        # Might be articular participle
                        w_feats = _parse_feats(w.feats)
                        if w_feats.get("VerbForm") == "Part":
                            constructions.append({
                                "scale": "clause",
                                "type": "articular_participle",
                                "verb": w.text,
                                "text": sent.text[:100],
                            })

            # Conditionals
            if w.deprel == "advcl":
                for child in words:
                    if child.head == w.id and child.deprel == "mark":
                        if lang == "en" and child.text.lower() == "if":
                            constructions.append({
                                "scale": "clause",
                                "type": "conditional",
                                "verb": w.text,
                                "text": sent.text[:100],
                            })
                        elif lang == "grc" and child.lemma in ("εἰ", "ἐάν", "ἤν"):
                            w_feats = _parse_feats(w.feats)
                            cond_type = "conditional_real"
                            mood = w_feats.get("Mood", "")
                            if mood == "Sub":
                                cond_type = "conditional_fv"
                            elif mood == "Opt":
                                cond_type = "conditional_opt"
                            constructions.append({
                                "scale": "clause",
                                "type": cond_type,
                                "marker": child.text,
                                "verb": w.text,
                                "text": sent.text[:100],
                            })
                        # Temporal clauses
                        elif lang == "en" and child.text.lower() in ("when", "while", "after", "before", "until"):
                            constructions.append({
                                "scale": "clause",
                                "type": "temporal",
                                "marker": child.text,
                                "text": sent.text[:100],
                            })
                        elif lang == "grc" and child.lemma in ("ὅτε", "ἐπεί", "ἐπειδή", "πρίν", "ἕως", "μέχρι"):
                            constructions.append({
                                "scale": "clause",
                                "type": "temporal",
                                "marker": child.text,
                                "text": sent.text[:100],
                            })
                        # Purpose clauses
                        elif lang == "grc" and child.lemma in ("ἵνα", "ὅπως"):
                            constructions.append({
                                "scale": "clause",
                                "type": "purpose",
                                "marker": child.text,
                                "text": sent.text[:100],
                            })
                        break

            # Genitive absolute (GRC only)
            if lang == "grc":
                w_feats = _parse_feats(w.feats)
                if w_feats.get("VerbForm") == "Part" and w_feats.get("Case") == "Gen":
                    for child in words:
                        if child.head == w.id and child.upos in ("NOUN", "PROPN", "PRON"):
                            c_feats = _parse_feats(child.feats)
                            if c_feats.get("Case") == "Gen":
                                constructions.append({
                                    "scale": "clause",
                                    "type": "genitive_absolute",
                                    "participle": w.text,
                                    "noun": child.text,
                                    "text": sent.text[:100],
                                })
                                break

        # --- Coordination ---
        if n_conj >= 2:
            constructions.append({
                "scale": "sentence",
                "type": "coordination_chain",
                "count": n_conj,
                "text": sent.text[:100],
            })

        # --- Phrase level ---
        for w in words:
            # Passive voice
            if w.deprel == "nsubj:pass" or (w.upos == "VERB" and "Pass" in (w.feats or "")):
                constructions.append({
                    "scale": "phrase",
                    "type": "passive",
                    "verb": w.text,
                })

            # Infinitive constructions
            w_feats = _parse_feats(w.feats)
            if w_feats.get("VerbForm") == "Inf":
                # Articular infinitive (GRC)
                has_det = any(c.head == w.id and c.upos == "DET" for c in words)
                if has_det and lang == "grc":
                    constructions.append({
                        "scale": "phrase",
                        "type": "articular_infinitive",
                        "verb": w.text,
                    })
                # Acc+Inf
                has_acc_subj = any(
                    c.head == w.id and c.deprel == "nsubj"
                    and "Acc" in (_parse_feats(c.feats).get("Case", ""))
                    for c in words
                )
                if has_acc_subj:
                    constructions.append({
                        "scale": "phrase",
                        "type": "acc_inf",
                        "verb": w.text,
                    })

            # Preposition + case (GRC phrase level)
            if lang == "grc" and w.upos == "ADP" and w.deprel == "case":
                head = word_map.get(w.head)
                if head:
                    h_feats = _parse_feats(head.feats)
                    case = h_feats.get("Case", "?")
                    constructions.append({
                        "scale": "phrase",
                        "type": "pp",
                        "prep": w.lemma,
                        "case": case,
                    })

            # Verb aspect (GRC)
            if lang == "grc" and w.upos == "VERB":
                tense = w_feats.get("Tense", "")
                mood = w_feats.get("Mood", "")
                voice = w_feats.get("Voice", "")
                if tense and mood:
                    constructions.append({
                        "scale": "phrase",
                        "type": "verb_form",
                        "tense": tense,
                        "mood": mood,
                        "voice": voice,
                        "lemma": w.lemma,
                    })

    return constructions


def _get_subtree_deprels(words, head_word) -> list[str]:
    """Get deprels of immediate children of a word."""
    return sorted(w.deprel for w in words if w.head == head_word.id)


def _parse_feats(feats_str: str | None) -> dict:
    if not feats_str:
        return {}
    result = {}
    for part in feats_str.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


# ====================================================================
# Pair alignment
# ====================================================================

def align_construction_pairs(en_constrs: list[dict], grc_constrs: list[dict]) -> list[dict]:
    """Align EN constructions to GRC constructions within a sentence pair.

    Strategy: for each EN clause-level construction, find the best matching
    GRC construction (same scale, compatible type). Record the mapping.
    """
    pairs = []

    # Build GRC construction pools by scale
    grc_by_scale = {}
    for c in grc_constrs:
        grc_by_scale.setdefault(c["scale"], []).append(c)

    # For each EN construction, find GRC match
    used_grc = set()

    for en_c in en_constrs:
        if en_c["scale"] not in ("clause", "sentence"):
            continue

        best_match = None
        best_idx = None

        grc_pool = grc_by_scale.get(en_c["scale"], [])
        for idx, grc_c in enumerate(grc_pool):
            if idx in used_grc:
                continue

            # Compatible types
            if en_c["type"] == "relative_clause" and grc_c["type"] in (
                "relative_clause", "articular_participle", "genitive_absolute"
            ):
                best_match = grc_c
                best_idx = idx
                break
            elif en_c["type"] == "conditional" and grc_c["type"].startswith("conditional"):
                best_match = grc_c
                best_idx = idx
                break
            elif en_c["type"] == "temporal" and grc_c["type"] == "temporal":
                best_match = grc_c
                best_idx = idx
                break
            elif en_c["type"] == "fragment" and grc_c["type"] == "fragment":
                best_match = grc_c
                best_idx = idx
                break
            elif en_c["type"] == "coordination_chain" and grc_c["type"] == "coordination_chain":
                best_match = grc_c
                best_idx = idx
                break

        if best_match and best_idx is not None:
            used_grc.add(best_idx)

        pairs.append({
            "en_type": en_c["type"],
            "en_scale": en_c["scale"],
            "grc_type": best_match["type"] if best_match else "none",
            "en_text": en_c.get("text", "")[:80],
        })

    # Also record GRC constructions that have no EN match (insertions)
    for idx, grc_c in enumerate(grc_constrs):
        if grc_c["scale"] in ("clause", "sentence") and idx not in used_grc:
            pairs.append({
                "en_type": "none",
                "en_scale": grc_c["scale"],
                "grc_type": grc_c["type"],
                "en_text": "",
            })

    return pairs


# ====================================================================
# Main extraction loop
# ====================================================================

def load_aligned_pairs(limit: int | None = None) -> list[tuple]:
    """Load (en_text, grc_text, doc_title, period) from database."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    query = """
        SELECT a.aligned_text, p.greek_text, d.title, d.period
        FROM alignments a
        JOIN passages p ON a.passage_id = p.passage_id
        JOIN documents d ON p.document_id = d.document_id
        WHERE a.aligned_text NOT LIKE '[awaiting%'
          AND a.aligned_text != ''
          AND p.greek_text != ''
          AND LENGTH(a.aligned_text) > 10
          AND LENGTH(p.greek_text) > 10
    """
    if limit:
        query += f" LIMIT {limit}"

    cur.execute(query)
    rows = cur.fetchall()
    conn.close()
    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    start_idx = 0
    if args.resume and CHECKPOINT_PATH.exists():
        ckpt = json.load(open(CHECKPOINT_PATH))
        start_idx = ckpt.get("processed", 0)
        print(f"Resuming from index {start_idx}")

    print("Loading aligned pairs from database...")
    pairs = load_aligned_pairs(limit=args.limit)
    print(f"  {len(pairs)} pairs loaded")

    if start_idx >= len(pairs):
        print("Already complete.")
        return

    en_nlp = _get_en_nlp()
    grc_nlp = _get_grc_nlp()

    mode = "a" if args.resume and start_idx > 0 else "w"
    outf = open(PAIRS_PATH, mode, encoding="utf-8")

    t0 = time.time()
    n_errors = 0

    for i in range(start_idx, len(pairs)):
        en_text, grc_text, doc_title, period = pairs[i]

        try:
            # Truncate very long texts to keep parsing fast
            en_text = en_text[:2000]
            grc_text = grc_text[:2000]

            en_doc = en_nlp(en_text)
            grc_doc = grc_nlp(grc_text)

            en_constrs = extract_constructions(en_doc, "en")
            grc_constrs = extract_constructions(grc_doc, "grc")

            aligned = align_construction_pairs(en_constrs, grc_constrs)

            record = {
                "idx": i,
                "source": doc_title,
                "period": period or "",
                "en_constructions": en_constrs,
                "grc_constructions": grc_constrs,
                "pairs": aligned,
            }

            outf.write(json.dumps(record, ensure_ascii=False) + "\n")

        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                print(f"  Error at {i}: {e}")

        if (i + 1) % args.batch_size == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - start_idx) / elapsed
            eta = (len(pairs) - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{len(pairs)} ({rate:.1f}/s, ETA {eta/60:.0f}m, {n_errors} errors)")

            # Checkpoint
            json.dump({"processed": i + 1}, open(CHECKPOINT_PATH, "w"))
            outf.flush()

    outf.close()
    json.dump({"processed": len(pairs)}, open(CHECKPOINT_PATH, "w"))

    elapsed = time.time() - t0
    print(f"\nDone: {len(pairs)} pairs in {elapsed/60:.1f}m ({n_errors} errors)")
    print(f"Output: {PAIRS_PATH}")


if __name__ == "__main__":
    main()
