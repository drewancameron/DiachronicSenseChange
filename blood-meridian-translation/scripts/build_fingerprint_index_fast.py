#!/usr/bin/env python3
"""
Fast fingerprint index using regex heuristics (no stanza).
Processes 142K sentences in seconds.

Usage:
  python3 scripts/build_fingerprint_index_fast.py --build
  python3 scripts/build_fingerprint_index_fast.py --query "See the child."
  python3 scripts/build_fingerprint_index_fast.py --query "He turned to the man who spoke."
  python3 scripts/build_fingerprint_index_fast.py --passage 003_the_mother_dead
"""

import json
import re
import sqlite3
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT.parent / "db" / "diachronic.db"
INDEX_DIR = ROOT / "models" / "fingerprint_index_fast"
FEATURES_PATH = INDEX_DIR / "features.npy"
META_PATH = INDEX_DIR / "metadata.jsonl"
PASSAGES = ROOT / "passages"

N_FEATURES = 16

# Patterns
RE_RELATIVE = re.compile(r'\b(?:who|whom|whose|which|that)\b', re.I)
RE_COND = re.compile(r'\b(?:if|unless)\b', re.I)
RE_TEMP = re.compile(r'\b(?:when|while|after|before|until|whenever|once)\b', re.I)
RE_CAUSAL = re.compile(r'\b(?:because)\b', re.I)
RE_PURPOSE = re.compile(r'\b(?:so\s+that|in\s+order\s+to|lest)\b', re.I)
RE_CONCESS = re.compile(r'\b(?:although|though|even\s+if)\b', re.I)
RE_RESULT = re.compile(r'\b(?:so\s+(?:that|\.\.\.)|such\s+that)\b', re.I)
RE_PASSIVE = re.compile(r'\b(?:was|were|been|being|is|are)\s+\w+(?:ed|en|t|wn)\b', re.I)
RE_COORD = re.compile(r'\b(?:and|or|nor)\b', re.I)
RE_BUT = re.compile(r'\b(?:but|yet|however)\b', re.I)
RE_FINITE = re.compile(
    r'\b(?:is|am|are|was|were|has|have|had|do|does|did|'
    r'will|would|shall|should|can|could|may|might|must|'
    r'goes|goes|says|said|came|went|took|gave|made|saw|knew|told|'
    r'\w+ed)\b', re.I
)
RE_SPEECH = re.compile(r'\b(?:said|cried|called|asked|replied|shouted|whispered)\b', re.I)


def fingerprint(text: str) -> np.ndarray:
    words = text.split()
    wc = len(words)

    n_rel = len(RE_RELATIVE.findall(text))
    n_cond = len(RE_COND.findall(text))
    n_temp = len(RE_TEMP.findall(text))
    n_causal = len(RE_CAUSAL.findall(text))
    n_purp = len(RE_PURPOSE.findall(text))
    n_concess = len(RE_CONCESS.findall(text))
    n_coord = len(RE_COORD.findall(text))
    n_but = len(RE_BUT.findall(text))
    has_passive = 1.0 if RE_PASSIVE.search(text) else 0.0
    has_finite = 1.0 if RE_FINITE.search(text) else 0.0
    has_speech = 1.0 if RE_SPEECH.search(text) else 0.0
    n_commas = text.count(",") + text.count(";")

    n_sub = n_rel + n_cond + n_temp + n_causal + n_purp + n_concess
    is_frag = 1.0 - has_finite

    if is_frag > 0.5:
        stype = 0.0
    elif n_sub == 0 and n_coord <= 1:
        stype = 1.0
    elif n_sub == 0:
        stype = 2.0
    elif n_coord <= 1:
        stype = 3.0
    else:
        stype = 4.0

    return np.array([
        min(wc / 10.0, 5.0),
        stype / 4.0,
        min(n_rel, 3) / 3.0,
        min(n_cond, 2) / 2.0,
        min(n_temp, 2) / 2.0,
        min(n_causal + n_purp, 2) / 2.0,
        min(n_concess, 2) / 2.0,
        has_passive,
        min(n_coord, 6) / 6.0,
        is_frag,
        min(n_but, 2) / 2.0,
        has_speech,
        min(n_commas, 5) / 5.0,
        min(n_sub, 4) / 4.0,
        has_finite,
        min(wc, 50) / 50.0,
    ], dtype=np.float32)


def label(text: str) -> dict:
    wc = len(text.split())
    n_rel = len(RE_RELATIVE.findall(text))
    n_cond = len(RE_COND.findall(text))
    n_temp = len(RE_TEMP.findall(text))
    n_coord = len(RE_COORD.findall(text))
    has_passive = bool(RE_PASSIVE.search(text))
    has_finite = bool(RE_FINITE.search(text))
    has_speech = bool(RE_SPEECH.search(text))
    n_sub = n_rel + n_cond + n_temp

    if not has_finite:
        stype = "fragment"
    elif n_sub == 0 and n_coord <= 1:
        stype = "simple"
    elif n_sub == 0:
        stype = "compound"
    elif n_coord <= 1:
        stype = "complex"
    else:
        stype = "compound_complex"

    return {"word_count": wc, "type": stype, "relative": n_rel,
            "conditional": n_cond, "temporal": n_temp,
            "coordination": n_coord, "passive": has_passive,
            "speech": has_speech}


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
    print(f"  {len(rows)} sentences")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    features = np.zeros((len(rows), N_FEATURES), dtype=np.float32)
    with open(META_PATH, "w") as mf:
        for i, (pid, en, grc, title, period) in enumerate(rows):
            en_clean = en.strip()
            features[i] = fingerprint(en_clean)
            mf.write(json.dumps({
                "passage_id": pid,
                "english": en_clean[:300],
                "greek": grc.replace("\n", " ").strip()[:300],
                "source": title,
                "period": period or "",
                "label": label(en_clean),
            }, ensure_ascii=False) + "\n")

    np.save(FEATURES_PATH, features)
    elapsed = time.time() - t0
    print(f"  Done: {len(rows)} fingerprints in {elapsed:.1f}s ({len(rows)/elapsed:.0f}/s)")
    print(f"  {FEATURES_PATH} ({features.nbytes / 1024:.0f} KB)")


def load_index():
    features = np.load(FEATURES_PATH)
    meta = []
    with open(META_PATH) as f:
        for line in f:
            if line.strip():
                meta.append(json.loads(line))
    return features, meta


def query(text: str, k: int = 5, features=None, meta=None):
    if features is None:
        features, meta = load_index()

    q = fingerprint(text)
    dists = np.sqrt(np.sum((features - q) ** 2, axis=1))
    top_idx = np.argpartition(dists, k)[:k]
    top_idx = top_idx[np.argsort(dists[top_idx])]

    q_label = label(text)
    results = []
    for idx in top_idx:
        m = meta[idx].copy()
        m["distance"] = round(float(dists[idx]), 3)
        results.append(m)
    return q_label, results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query", type=str)
    parser.add_argument("--passage", type=str, help="Query all sentences in a BM passage")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    if args.build:
        build_index()

    elif args.query:
        q_label, results = query(args.query, k=args.k)
        print(f'Query: "{args.query}"')
        print(f"  {q_label}\n")
        for r in results:
            rl = r.get("label", {})
            print(f"  d={r['distance']:.3f} [{r['source']}] "
                  f"({rl.get('type','?')}, {rl.get('word_count','?')}w, "
                  f"rel={rl.get('relative',0)}, coord={rl.get('coordination',0)})")
            print(f"    EN: {r['english'][:70]}")
            print(f"    GR: {r['greek'][:70]}")
            print()

    elif args.passage:
        p = PASSAGES / f"{args.passage}.json"
        if not p.exists():
            print(f"Not found: {p}")
            return
        en_text = json.load(open(p)).get("text", "")

        features, meta = load_index()
        # Split into sentences (rough)
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', en_text) if s.strip()]
        for sent in sents:
            q_label, results = query(sent, k=3, features=features, meta=meta)
            print(f'"{sent[:70]}{"..." if len(sent)>70 else ""}"')
            print(f"  {q_label['type']}, {q_label['word_count']}w, "
                  f"rel={q_label['relative']}, coord={q_label['coordination']}")
            for r in results[:2]:
                rl = r.get("label", {})
                print(f"    d={r['distance']:.3f} [{r['source'][:20]}] "
                      f"({rl.get('type','?')}, {rl.get('word_count','?')}w) "
                      f"EN: {r['english'][:50]}")
                print(f"      GR: {r['greek'][:50]}")
            print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
