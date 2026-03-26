#!/usr/bin/env python3
"""
Bulk evidence extraction using nano with established sense inventory.

Now that we have a clean 86-sense inventory from the pilot extraction,
we can use the cheaper/faster nano model to classify the remaining
occurrences. The task is simpler: pick from known senses, not discover
new ones.
"""

import json
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path

from openai import OpenAI

DB_PATH = Path(__file__).parent.parent / "db" / "diachronic.db"

TRANSLIT = {
    "κόσμος": "kosmos", "λόγος": "logos", "ψυχή": "psyche",
    "ἀρετή": "arete", "δίκη": "dike", "τέχνη": "techne",
    "νόμος": "nomos", "φύσις": "physis", "δαίμων": "daimon",
    "σῶμα": "soma", "θεός": "theos", "χάρις": "charis",
}

SYSTEM = """\
You are classifying Ancient Greek word occurrences by sense. You will be given
a Greek passage, an English translation (if available), and a fixed sense
inventory. Pick the best matching sense(s). Return JSON.

RULES:
1. Pick from the provided sense inventory — do NOT invent new senses.
2. If the evidence clearly supports one sense, assign it with high confidence.
3. If ambiguous, list 2-3 candidate senses with confidence scores.
4. If no translation is available and the Greek alone is insufficient, return "undetermined".
5. Also classify register: unmarked, poetic_elevated, archaizing, homericizing, technical, colloquial, uncertain.
6. Be concise — no explanations needed, just the JSON.
"""

USER_TEMPLATE = """\
Lemma: {lemma} ({translit})
Surface: {surface}
Period: {period}
Author: {author}

Greek: {greek}

Translation: {translation}

Senses:
{senses}

Return JSON: {{"primary_sense": "label", "confidence": 0.0-1.0, "alternatives": [{{"sense": "label", "confidence": 0.0-1.0}}], "register": "label"}}
"""


class BulkExtractor:
    def __init__(self):
        self.client = OpenAI(timeout=60.0)
        self.usage = {"input": 0, "output": 0}
        self.sense_cache = {}

    def _call(self, user: str) -> str:
        old = signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(45)
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            if resp.usage:
                self.usage["input"] += resp.usage.prompt_tokens
                self.usage["output"] += resp.usage.completion_tokens
            return resp.choices[0].message.content
        except Exception as e:
            return None
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)

    def get_senses(self, conn, lemma):
        if lemma not in self.sense_cache:
            rows = conn.execute(
                "SELECT sense_id, sense_label FROM sense_inventory WHERE lemma = ? ORDER BY sense_id",
                (lemma,),
            ).fetchall()
            self.sense_cache[lemma] = rows
        return self.sense_cache[lemma]

    def get_translation(self, conn, passage_id):
        row = conn.execute("""
            SELECT substr(al.aligned_text, 1, 300), t.translator
            FROM alignments al
            JOIN translations t ON al.translation_id = t.translation_id
            WHERE al.passage_id = ?
            AND al.aligned_text NOT LIKE '[awaiting%'
            AND length(al.aligned_text) > 20
            ORDER BY al.alignment_confidence DESC
            LIMIT 1
        """, (passage_id,)).fetchone()
        return row if row else (None, None)

    def extract_one(self, conn, occ_id, lemma, surface, period, author,
                    greek, passage_id):
        senses = self.get_senses(conn, lemma)
        sense_text = "\n".join(f"- {s[1]}" for s in senses if "undetermined" not in s[1].lower())

        trans_text, translator = self.get_translation(conn, passage_id)
        translation = f"{trans_text} (tr. {translator})" if trans_text else "(no translation available)"

        prompt = USER_TEMPLATE.format(
            lemma=lemma, translit=TRANSLIT.get(lemma, lemma),
            surface=surface, period=period or "?", author=author or "?",
            greek=greek[:200], translation=translation,
            senses=sense_text,
        )

        raw = self._call(prompt)
        if not raw:
            return False

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            return False

        primary = result.get("primary_sense", "")
        conf = result.get("confidence", 0.5)
        register = result.get("register", "uncertain")

        # Map to sense_id
        sense_id = None
        for sid, slabel in senses:
            if slabel == primary or primary.lower() in slabel.lower():
                sense_id = sid
                break

        if not sense_id:
            # Fuzzy match
            for sid, slabel in senses:
                words_p = set(primary.lower().split())
                words_s = set(slabel.lower().split())
                if len(words_p & words_s) >= 2:
                    sense_id = sid
                    break

        if not sense_id:
            return False

        # Robust confidence parsing
        def parse_conf(v, default=0.5):
            try:
                if isinstance(v, (int, float)):
                    return max(0.0, min(1.0, float(v)))
                s = str(v).strip().strip(':').strip()
                return max(0.0, min(1.0, float(s)))
            except (ValueError, TypeError):
                return default

        conf = parse_conf(conf)

        # Store
        conn.execute(
            "INSERT INTO candidate_labels (occurrence_id, sense_id, confidence, is_primary) VALUES (?, ?, ?, 1)",
            (occ_id, sense_id, conf),
        )

        # Alternatives
        for alt in result.get("alternatives", []):
            alt_sense = alt.get("sense", "")
            alt_id = None
            for sid, slabel in senses:
                if slabel == alt_sense or alt_sense.lower() in slabel.lower():
                    alt_id = sid
                    break
            if alt_id and alt_id != sense_id:
                alt_conf = parse_conf(alt.get("confidence", 0.3))
                conn.execute(
                    "INSERT INTO candidate_labels (occurrence_id, sense_id, confidence, is_primary) VALUES (?, ?, ?, 0)",
                    (occ_id, alt_id, alt_conf),
                )

        # Register
        VALID_REG = {"unmarked", "poetic_elevated", "archaizing", "homericizing",
                     "quoted_allusive", "scholastic", "colloquial", "technical", "uncertain"}
        reg = register if register in VALID_REG else "uncertain"
        conn.execute(
            "INSERT INTO register_labels (occurrence_id, register, confidence) VALUES (?, ?, ?)",
            (occ_id, reg, conf),
        )

        return True

    def run(self, target_count: int = 5171):
        conn = sqlite3.connect(DB_PATH)

        # Get occurrences that DON'T already have labels
        already = set(r[0] for r in conn.execute(
            "SELECT DISTINCT occurrence_id FROM candidate_labels"
        ).fetchall())

        # Pre-compute passages with translations (single query)
        print("Building translation index...", flush=True)
        passages_with_trans = set(
            r[0] for r in conn.execute("""
                SELECT DISTINCT passage_id FROM alignments
                WHERE aligned_text NOT LIKE '[awaiting%'
                AND length(aligned_text) > 20
            """).fetchall()
        )
        print(f"  {len(passages_with_trans):,} passages have translations", flush=True)

        # Get all candidate occurrences in one query
        rows = conn.execute("""
            SELECT o.occurrence_id, o.lemma, t.surface_form, o.period,
                   a.name, o.greek_context, o.passage_id
            FROM occurrences o
            JOIN tokens t ON o.token_id = t.token_id
            JOIN documents d ON o.document_id = d.document_id
            JOIN authors a ON d.author_id = a.author_id
            WHERE length(o.greek_context) > 20
            ORDER BY RANDOM()
        """).fetchall()

        # Split into with/without translation, skip already labelled
        with_trans = []
        without_trans = []
        for row in rows:
            if row[0] in already:
                continue
            if row[6] in passages_with_trans:
                with_trans.append(row)
            else:
                without_trans.append(row)

        # Prioritize occurrences with translations
        candidates = with_trans[:target_count]
        if len(candidates) < target_count:
            candidates.extend(without_trans[:target_count - len(candidates)])

        print(f"Extracting {len(candidates)} occurrences "
              f"({len(with_trans[:target_count])} with translations)...", flush=True)

        extracted = 0
        failed = 0

        for i, (occ_id, lemma, surface, period, author, greek, pid) in enumerate(candidates):
            ok = self.extract_one(conn, occ_id, lemma, surface, period, author, greek, pid)
            if ok:
                extracted += 1
            else:
                failed += 1

            if (i + 1) % 100 == 0:
                conn.commit()
                cost = (self.usage["input"] * 0.10 + self.usage["output"] * 0.40) / 1_000_000
                print(f"  {i+1}/{len(candidates)}: {extracted} ok, {failed} fail, ${cost:.2f}",
                      flush=True)

            time.sleep(0.05)  # gentle rate limiting

        conn.commit()

        # Final report
        cost = (self.usage["input"] * 0.10 + self.usage["output"] * 0.40) / 1_000_000
        total_labelled = conn.execute("SELECT COUNT(DISTINCT occurrence_id) FROM candidate_labels").fetchone()[0]
        total_occ = conn.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0]

        print(f"\n{'='*60}")
        print(f"Done: {extracted} extracted, {failed} failed")
        print(f"Total labelled: {total_labelled:,} / {total_occ:,} ({total_labelled*100//total_occ}%)")
        print(f"API cost: ${cost:.2f}")
        print(f"Tokens: {self.usage['input'] + self.usage['output']:,}")

        # Per-lemma coverage
        print(f"\nPer-lemma coverage:")
        for row in conn.execute("""
            SELECT o.lemma, COUNT(DISTINCT cl.occurrence_id), COUNT(DISTINCT o.occurrence_id)
            FROM occurrences o
            LEFT JOIN candidate_labels cl ON o.occurrence_id = cl.occurrence_id
            GROUP BY o.lemma ORDER BY o.lemma
        """).fetchall():
            pct = row[1] * 100 // row[2] if row[2] > 0 else 0
            print(f"  {row[0]}: {row[1]:,} / {row[2]:,} ({pct}%)")

        conn.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bulk sense extraction with nano")
    parser.add_argument("--target", type=int, default=5171)
    args = parser.parse_args()

    extractor = BulkExtractor()
    extractor.run(target_count=args.target)


if __name__ == "__main__":
    main()
