#!/usr/bin/env python3
"""
Build a searchable fingerprint index of all paired English sentences
using full stanza dependency parsing for accurate structural features.

Stores: numpy feature array + JSON metadata for instant nearest-neighbor lookup.

Usage:
  python3 scripts/build_fingerprint_index.py           # build full index
  python3 scripts/build_fingerprint_index.py --query "See the child."
"""

import json
import sqlite3
import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT.parent / "db" / "diachronic.db"
INDEX_DIR = ROOT / "models" / "fingerprint_index"
FEATURES_PATH = INDEX_DIR / "features.npy"
META_PATH = INDEX_DIR / "metadata.jsonl"
CHECKPOINT_PATH = INDEX_DIR / "checkpoint.json"

_en_nlp = None

def _get_en_nlp():
    global _en_nlp
    if _en_nlp is None:
        import stanza
        _en_nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse',
                                   verbose=False)
    return _en_nlp


# Feature vector dimension
N_FEATURES = 16


def fingerprint_stanza(sent) -> np.ndarray:
    """Extract structural feature vector from a stanza sentence."""
    words = sent.words
    wc = len([w for w in words if w.upos != "PUNCT"])

    has_root_verb = 0.0
    n_relcl = 0
    n_advcl = 0
    n_cond = 0
    n_temp = 0
    n_purp = 0
    n_coord = 0
    n_passive = 0
    n_ccomp = 0
    n_xcomp = 0
    n_parataxis = 0

    root_id = None
    for w in words:
        if w.deprel == "root":
            root_id = w.id
            if w.upos in ("VERB", "AUX"):
                has_root_verb = 1.0

        if w.deprel == "acl:relcl":
            n_relcl += 1
        elif w.deprel == "advcl":
            n_advcl += 1
            for c in words:
                if c.head == w.id and c.deprel == "mark":
                    low = c.text.lower()
                    if low in ("if", "unless"):
                        n_cond += 1
                    elif low in ("when", "while", "after", "before", "until", "since", "once"):
                        n_temp += 1
                    elif low in ("to", "so", "that"):
                        n_purp += 1
                    break
        elif w.deprel == "conj":
            n_coord += 1
        elif w.deprel == "nsubj:pass":
            n_passive += 1
        elif w.deprel == "ccomp":
            n_ccomp += 1
        elif w.deprel == "xcomp":
            n_xcomp += 1
        elif w.deprel == "parataxis":
            n_parataxis += 1

    # Count unique deprels of root's children (argument structure)
    root_children = set()
    if root_id:
        root_children = set(w.deprel for w in words if w.head == root_id and w.upos != "PUNCT")
    n_root_args = len(root_children)

    # Sentence type
    is_fragment = 1.0 - has_root_verb
    n_subordinators = n_relcl + n_advcl
    if is_fragment:
        sent_type = 0.0
    elif n_subordinators == 0 and n_coord <= 1:
        sent_type = 1.0  # simple
    elif n_subordinators == 0:
        sent_type = 2.0  # compound
    elif n_coord <= 1:
        sent_type = 3.0  # complex
    else:
        sent_type = 4.0  # compound_complex

    return np.array([
        min(wc / 10.0, 5.0),            # 0: normalised word count
        sent_type / 4.0,                 # 1: sentence type
        min(n_relcl, 3) / 3.0,          # 2: relative clauses
        min(n_cond, 2) / 2.0,           # 3: conditionals
        min(n_temp, 2) / 2.0,           # 4: temporals
        min(n_purp, 2) / 2.0,           # 5: purpose clauses
        min(n_advcl, 4) / 4.0,          # 6: all adverbial clauses
        1.0 if n_passive > 0 else 0.0,  # 7: has passive
        min(n_coord, 6) / 6.0,          # 8: coordination count
        is_fragment,                      # 9: is fragment
        min(n_ccomp + n_xcomp, 3) / 3.0,# 10: complement clauses
        min(n_parataxis, 3) / 3.0,      # 11: parataxis
        min(n_root_args, 6) / 6.0,      # 12: root argument count
        has_root_verb,                    # 13: has finite root verb
        min(wc, 50) / 50.0,             # 14: raw word count (fine)
        min(n_subordinators, 4) / 4.0,  # 15: total subordination
    ], dtype=np.float32)


def fingerprint_label(sent) -> dict:
    """Human-readable fingerprint for display."""
    words = sent.words
    wc = len([w for w in words if w.upos != "PUNCT"])
    n_relcl = sum(1 for w in words if w.deprel == "acl:relcl")
    n_coord = sum(1 for w in words if w.deprel == "conj")
    n_cond = 0
    n_temp = 0
    has_passive = any(w.deprel == "nsubj:pass" for w in words)
    has_root_verb = any(w.deprel == "root" and w.upos in ("VERB", "AUX") for w in words)

    for w in words:
        if w.deprel == "advcl":
            for c in words:
                if c.head == w.id and c.deprel == "mark":
                    if c.text.lower() in ("if", "unless"):
                        n_cond += 1
                    elif c.text.lower() in ("when", "while", "after", "before", "until"):
                        n_temp += 1
                    break

    n_sub = n_relcl + n_cond + n_temp
    if not has_root_verb:
        stype = "fragment"
    elif n_sub == 0 and n_coord <= 1:
        stype = "simple"
    elif n_sub == 0:
        stype = "compound"
    elif n_coord <= 1:
        stype = "complex"
    else:
        stype = "compound_complex"

    return {
        "word_count": wc, "type": stype, "relative": n_relcl,
        "conditional": n_cond, "temporal": n_temp,
        "coordination": n_coord, "passive": has_passive,
    }


# ====================================================================
# Build index
# ====================================================================

def build_index():
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    print("Loading paired English sentences...")
    cur.execute("""
        SELECT p.passage_id, a.aligned_text, p.greek_text, d.title, d.period
        FROM alignments a
        JOIN passages p ON a.passage_id = p.passage_id
        JOIN documents d ON p.document_id = d.document_id
        WHERE a.aligned_text NOT LIKE '[awaiting%]' AND a.aligned_text != ''
          AND LENGTH(a.aligned_text) BETWEEN 10 AND 500
          AND LENGTH(p.greek_text) BETWEEN 10 AND 600
          AND a.aligned_text NOT LIKE '%[%'
    """)
    rows = cur.fetchall()
    conn.close()
    print(f"  {len(rows)} sentences loaded")

    # Check for checkpoint
    start_idx = 0
    if CHECKPOINT_PATH.exists():
        ckpt = json.load(open(CHECKPOINT_PATH))
        start_idx = ckpt.get("processed", 0)
        print(f"  Resuming from {start_idx}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    nlp = _get_en_nlp()

    # We'll collect features and metadata, then save at the end
    all_features = []
    all_meta = []

    # If resuming, load what we have
    if start_idx > 0 and FEATURES_PATH.exists():
        all_features = list(np.load(FEATURES_PATH))
        with open(META_PATH) as f:
            all_meta = [json.loads(l) for l in f if l.strip()]

    t0 = time.time()
    n_errors = 0
    batch_size = 500

    for i in range(start_idx, len(rows)):
        pid, en, grc, title, period = rows[i]
        en_clean = en.strip()

        try:
            doc = nlp(en_clean)
            # Take the first sentence (most pairs are single-sentence)
            if doc.sentences:
                fp = fingerprint_stanza(doc.sentences[0])
                label = fingerprint_label(doc.sentences[0])
            else:
                fp = np.zeros(N_FEATURES, dtype=np.float32)
                label = {}

            all_features.append(fp)
            all_meta.append({
                "passage_id": pid,
                "english": en_clean[:300],
                "greek": grc.replace("\n", " ").strip()[:300],
                "source": title,
                "period": period or "",
                "label": label,
            })
        except Exception as e:
            n_errors += 1
            # Append zeros so indices stay aligned
            all_features.append(np.zeros(N_FEATURES, dtype=np.float32))
            all_meta.append({
                "passage_id": pid, "english": en_clean[:300],
                "greek": grc.replace("\n", " ").strip()[:300],
                "source": title, "period": period or "",
                "label": {}, "error": str(e),
            })

        if (i + 1) % batch_size == 0:
            elapsed = time.time() - t0
            done = i + 1 - start_idx
            rate = done / elapsed if elapsed > 0 else 0
            remaining = len(rows) - i - 1
            eta = remaining / rate if rate > 0 else 0
            print(f"  {i+1}/{len(rows)} ({rate:.1f}/s, ETA {eta/60:.0f}m, {n_errors} errors)")

            # Checkpoint
            features_arr = np.array(all_features, dtype=np.float32)
            np.save(FEATURES_PATH, features_arr)
            with open(META_PATH, "w") as f:
                for m in all_meta:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            json.dump({"processed": i + 1}, open(CHECKPOINT_PATH, "w"))

    # Final save
    features_arr = np.array(all_features, dtype=np.float32)
    np.save(FEATURES_PATH, features_arr)
    with open(META_PATH, "w") as f:
        for m in all_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    json.dump({"processed": len(rows)}, open(CHECKPOINT_PATH, "w"))

    elapsed = time.time() - t0
    print(f"\nDone: {len(rows)} fingerprints in {elapsed/60:.1f}m "
          f"({len(rows)/(elapsed or 1):.1f}/s, {n_errors} errors)")
    print(f"  Features: {FEATURES_PATH} ({features_arr.nbytes / 1024:.0f} KB)")


# ====================================================================
# Query index
# ====================================================================

def load_index():
    features = np.load(FEATURES_PATH)
    metadata = []
    with open(META_PATH) as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))
    return features, metadata


def query(text: str, k: int = 5, features=None, metadata=None):
    """Find k nearest structural matches for an English sentence."""
    if features is None:
        features, metadata = load_index()

    nlp = _get_en_nlp()
    doc = nlp(text)
    if not doc.sentences:
        return []

    q = fingerprint_stanza(doc.sentences[0])
    q_label = fingerprint_label(doc.sentences[0])

    dists = np.sqrt(np.sum((features - q) ** 2, axis=1))
    top_idx = np.argpartition(dists, k)[:k]
    top_idx = top_idx[np.argsort(dists[top_idx])]

    results = []
    for idx in top_idx:
        m = metadata[idx].copy()
        m["distance"] = round(float(dists[idx]), 3)
        results.append(m)

    return q_label, results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query", type=str)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    if args.build:
        build_index()
    elif args.query:
        label, results = query(args.query, k=args.k)
        print(f'Query: "{args.query}"')
        print(f"  Fingerprint: {label}\n")
        for r in results:
            rl = r.get("label", {})
            print(f"  d={r['distance']:.3f} [{r['source']}] "
                  f"({rl.get('type','?')}, {rl.get('word_count','?')}w, "
                  f"rel={rl.get('relative',0)}, coord={rl.get('coordination',0)})")
            print(f"    EN: {r['english'][:70]}")
            print(f"    GR: {r['greek'][:70]}")
            print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
