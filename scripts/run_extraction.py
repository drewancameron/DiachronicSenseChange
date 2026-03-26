#!/usr/bin/env python3
"""
WP2 Master: Evidence extraction pipeline.

Runs the full WP2 workflow:
  1. Align notes to passages/occurrences (free)
  2. Sample occurrences stratified by period/author
  3. Build extraction prompts with aligned evidence
  4. Call OpenAI to extract structured sense evidence
  5. Store results in evidence database

Usage:
  # Dry run (steps 1-3 only, no API):
  python3 scripts/run_extraction.py --dry-run

  # Full run:
  export OPENAI_API_KEY="sk-..."
  python3 scripts/run_extraction.py

  # Run on specific lemma:
  python3 scripts/run_extraction.py --lemma κόσμος

  # Limit sample size:
  python3 scripts/run_extraction.py --target 20
"""

import json
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

DB_PATH = Path(__file__).parent.parent / "db" / "diachronic.db"

TRANSLIT = {
    "κόσμος": "kosmos", "λόγος": "logos", "ψυχή": "psyche",
    "ἀρετή": "arete", "δίκη": "dike", "τέχνη": "techne",
    "νόμος": "nomos", "φύσις": "physis", "δαίμων": "daimon",
    "σῶμα": "soma", "θεός": "theos", "χάρις": "charis",
}

PILOT_LEMMATA = list(TRANSLIT.keys())

# ── Prompts ────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a research assistant for a historical semantics project on Ancient Greek.
Your role is to extract structured sense evidence from translations, translator's
notes, commentaries, and lexicographic sources. Return valid JSON.

CRITICAL RULES:
1. You are extracting evidence, not inventing interpretations.
2. Every sense claim must cite the exact wording from the source material.
3. If the evidence is ambiguous, record ALL plausible senses with confidence levels.
4. Distinguish DIRECT evidence (translation clearly renders the word with a specific
   meaning) from INFERENTIAL evidence (meaning inferred from context).
5. Flag archaizing, poetic, or register-marked usage when detectable.
6. If you cannot determine the sense, say so explicitly. Do not guess.
"""

USER_TEMPLATE = """\
## Target word
- **Lemma**: {lemma} ({translit})
- **Surface form**: {surface_form}

## Greek passage
**Reference**: {reference}
**Period**: {period}
**Author**: {author}
**Text**: {greek_text}

## Aligned English translation(s)
{translations}

## Scholarly notes (if any)
{notes}

## Known senses for {lemma} (if established)
{sense_inventory}

## Task
What sense(s) of **{lemma}** does the evidence support for this occurrence?

Return JSON:
{{
  "lemma": "{lemma}",
  "reference": "{reference}",
  "candidate_senses": [
    {{
      "sense_label": "short label (e.g. 'order/arrangement')",
      "confidence": 0.0-1.0,
      "is_primary": true/false,
      "evidence": [
        {{
          "text": "exact quoted span from translation or note",
          "type": "translation_rendering|translator_note|commentary_discussion",
          "source": "translator/commentator name",
          "directness": "direct|inferential|contextual"
        }}
      ]
    }}
  ],
  "register": {{
    "label": "unmarked|poetic_elevated|archaizing|homericizing|colloquial|technical|uncertain",
    "confidence": 0.0-1.0,
    "evidence": "brief justification or null"
  }},
  "notes": "any additional observations about ambiguity or context"
}}
"""


class ExtractionPipeline:

    def __init__(self, db_path: Path = DB_PATH, api_key: str | None = None):
        self.db_path = db_path
        self.client = None
        if OpenAI and (api_key or os.environ.get("OPENAI_API_KEY")):
            self.client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                timeout=60.0,
            )
        self.usage = {"input": 0, "output": 0}

    def _call_llm(self, system: str, user: str) -> str:
        """Call OpenAI with hard timeout."""
        def _timeout(signum, frame):
            raise TimeoutError("LLM call exceeded 60s")

        old = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(60)
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            if resp.usage:
                self.usage["input"] += resp.usage.prompt_tokens
                self.usage["output"] += resp.usage.completion_tokens
            return resp.choices[0].message.content
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    # ── Step 1: Note alignment ────────────────────────────────

    def align_notes(self):
        """Run note alignment (free, no API)."""
        import subprocess
        print("Step 1: Aligning notes to passages...")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "align_notes.py")],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"  Warning: {result.stderr[:200]}")

    # ── Step 2: Sample occurrences ────────────────────────────

    def sample_occurrences(self, target: int = 80, lemma_filter: str | None = None):
        """Run stratified sampling."""
        import subprocess
        print(f"\nStep 2: Sampling ~{target} occurrences per lemma...")
        cmd = [sys.executable, str(Path(__file__).parent / "sample_occurrences.py"),
               "--target", str(target)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)

    # ── Step 3: Build prompts ─────────────────────────────────

    def build_prompt(self, conn, occurrence_id: int) -> dict | None:
        """Build a complete extraction prompt for one occurrence."""

        # Get occurrence details
        row = conn.execute("""
            SELECT o.lemma, o.greek_context, o.period,
                   t.surface_form, p.reference, p.greek_text,
                   a.name as author
            FROM occurrences o
            JOIN tokens t ON o.token_id = t.token_id
            JOIN passages p ON o.passage_id = p.passage_id
            JOIN documents d ON o.document_id = d.document_id
            JOIN authors a ON d.author_id = a.author_id
            WHERE o.occurrence_id = ?
        """, (occurrence_id,)).fetchone()

        if not row:
            return None

        lemma, context, period, surface, ref, greek_text, author = row

        # Get aligned translations
        translations = conn.execute("""
            SELECT al.aligned_text, t.translator, al.alignment_confidence
            FROM alignments al
            JOIN translations t ON al.translation_id = t.translation_id
            WHERE al.passage_id = (SELECT passage_id FROM occurrences WHERE occurrence_id = ?)
            AND al.aligned_text NOT LIKE '[awaiting%'
            AND length(al.aligned_text) > 20
            ORDER BY al.alignment_confidence DESC
            LIMIT 3
        """, (occurrence_id,)).fetchall()

        trans_text = ""
        if translations:
            parts = []
            for text, translator, conf in translations:
                parts.append(f"**{translator}** (confidence: {conf:.2f}):\n{text[:500]}")
            trans_text = "\n\n".join(parts)
        else:
            trans_text = "(No aligned translation available)"

        # Get linked notes
        notes = conn.execute("""
            SELECT en.note_text, en.note_type
            FROM note_alignments na
            JOIN extracted_notes en ON na.note_id = en.note_id
            WHERE na.occurrence_id = ? OR na.passage_id = (
                SELECT passage_id FROM occurrences WHERE occurrence_id = ?
            )
            LIMIT 5
        """, (occurrence_id, occurrence_id)).fetchall()

        notes_text = ""
        if notes:
            parts = []
            for text, ntype in notes:
                parts.append(f"[{ntype}] {text[:300]}")
            notes_text = "\n\n".join(parts)
        else:
            notes_text = "(No scholarly notes available for this passage)"

        # Get current sense inventory for this lemma
        senses = conn.execute("""
            SELECT sense_label, sense_description
            FROM sense_inventory WHERE lemma = ?
        """, (lemma,)).fetchall()

        sense_text = ""
        if senses:
            sense_text = "\n".join(f"- **{s[0]}**: {s[1] or ''}" for s in senses)
        else:
            sense_text = "(No sense inventory yet — propose senses based on evidence)"

        user_msg = USER_TEMPLATE.format(
            lemma=lemma,
            translit=TRANSLIT.get(lemma, lemma),
            surface_form=surface,
            reference=ref or "?",
            period=period or "?",
            author=author or "?",
            greek_text=(greek_text or context)[:300],
            translations=trans_text,
            notes=notes_text,
            sense_inventory=sense_text,
        )

        return {
            "occurrence_id": occurrence_id,
            "lemma": lemma,
            "system": SYSTEM_PROMPT,
            "user": user_msg,
            "has_translation": len(translations) > 0,
            "has_notes": len(notes) > 0,
        }

    # ── Step 4: Run extraction ────────────────────────────────

    def extract_evidence(self, conn, occurrence_id: int, prompt: dict) -> dict | None:
        """Run LLM extraction for one occurrence."""
        try:
            raw = self._call_llm(prompt["system"], prompt["user"])
            result = json.loads(raw)
            return result
        except json.JSONDecodeError as e:
            print(f"    JSON parse error: {e}", flush=True)
            return None
        except Exception as e:
            print(f"    Extraction failed: {e}", flush=True)
            return None

    # ── Step 5: Store results ─────────────────────────────────

    def store_result(self, conn, occurrence_id: int, lemma: str, result: dict):
        """Store extraction result in the evidence database."""

        for candidate in result.get("candidate_senses", []):
            sense_label = candidate.get("sense_label", "unknown")

            # Find or create sense in inventory
            sense_row = conn.execute(
                "SELECT sense_id FROM sense_inventory WHERE lemma = ? AND sense_label = ?",
                (lemma, sense_label),
            ).fetchone()

            if sense_row:
                sense_id = sense_row[0]
            else:
                cur = conn.execute(
                    "INSERT INTO sense_inventory (lemma, sense_label) VALUES (?, ?)",
                    (lemma, sense_label),
                )
                sense_id = cur.lastrowid

            # Create candidate label
            cur = conn.execute(
                """INSERT INTO candidate_labels
                   (occurrence_id, sense_id, confidence, is_primary)
                   VALUES (?, ?, ?, ?)""",
                (occurrence_id, sense_id,
                 candidate.get("confidence", 0.5),
                 1 if candidate.get("is_primary") else 0),
            )
            label_id = cur.lastrowid

            # Store evidence spans
            VALID_EVIDENCE_TYPES = {
                "translation_rendering", "translator_note",
                "commentary_discussion", "lexicon_gloss",
                "scholion", "llm_inference",
            }
            VALID_DIRECTNESS = {"direct", "inferential", "contextual"}

            for ev in candidate.get("evidence", []):
                ev_type = ev.get("type", "translation_rendering")
                if ev_type not in VALID_EVIDENCE_TYPES:
                    ev_type = "translation_rendering"
                directness = ev.get("directness", "direct")
                if directness not in VALID_DIRECTNESS:
                    directness = "direct"

                conn.execute(
                    """INSERT INTO evidence_spans
                       (label_id, evidence_text, evidence_type,
                        source_identity, directness)
                       VALUES (?, ?, ?, ?, ?)""",
                    (label_id,
                     ev.get("text", ""),
                     ev_type,
                     ev.get("source"),
                     directness),
                )

        # Store register label
        VALID_REGISTERS = {
            "unmarked", "poetic_elevated", "archaizing",
            "homericizing", "quoted_allusive", "scholastic",
            "colloquial", "technical", "uncertain",
        }
        register = result.get("register", {})
        if register.get("label"):
            reg_label = register["label"]
            if reg_label not in VALID_REGISTERS:
                reg_label = "uncertain"
            conn.execute(
                """INSERT INTO register_labels
                   (occurrence_id, register, confidence, evidence_text)
                   VALUES (?, ?, ?, ?)""",
                (occurrence_id, reg_label,
                 register.get("confidence", 0.5),
                 register.get("evidence")),
            )

        # Log provenance
        conn.execute(
            """INSERT INTO provenance_events
               (entity_type, entity_id, action, agent, notes)
               VALUES ('occurrence', ?, 'evidence_extracted', 'llm:gpt-4.1-mini', ?)""",
            (occurrence_id, result.get("notes")),
        )

    # ── Main orchestrator ─────────────────────────────────────

    def run(self, target: int = 80, lemma_filter: str | None = None,
            dry_run: bool = False):
        """Run the full extraction pipeline."""

        # Step 1: Note alignment
        self.align_notes()

        # Step 2: Sampling
        self.sample_occurrences(target=target)

        conn = sqlite3.connect(self.db_path)

        # Get sampled occurrences
        query = """
            SELECT ms.occurrence_id, o.lemma
            FROM model_splits ms
            JOIN occurrences o ON ms.occurrence_id = o.occurrence_id
            WHERE ms.split_name = 'wp2_extraction_sample'
        """
        if lemma_filter:
            query += f" AND o.lemma = '{lemma_filter}'"
        query += " ORDER BY o.lemma, o.period"

        samples = conn.execute(query).fetchall()
        print(f"\nStep 3-5: Extracting evidence for {len(samples)} occurrences...")

        if dry_run:
            # Show sample prompts without calling API
            for occ_id, lemma in samples[:3]:
                prompt = self.build_prompt(conn, occ_id)
                if prompt:
                    print(f"\n{'─'*60}")
                    print(f"  {TRANSLIT.get(lemma, lemma)} (occ {occ_id})")
                    print(f"  Has translation: {prompt['has_translation']}")
                    print(f"  Has notes: {prompt['has_notes']}")
                    print(f"\n  --- User prompt (first 500 chars) ---")
                    print(f"  {prompt['user'][:500]}")
            print(f"\n[DRY RUN] Would extract {len(samples)} occurrences")
            est_cost = (len(samples) * 2000 * 0.15 + len(samples) * 500 * 0.60) / 1_000_000
            print(f"Estimated cost (mini): ${est_cost:.2f}")
            conn.close()
            return

        if not self.client:
            print("ERROR: OpenAI client not available. Set OPENAI_API_KEY.")
            conn.close()
            return

        # Run extraction
        extracted = 0
        failed = 0
        by_lemma = {}

        for i, (occ_id, lemma) in enumerate(samples):
            prompt = self.build_prompt(conn, occ_id)
            if not prompt:
                failed += 1
                continue

            result = self.extract_evidence(conn, occ_id, prompt)

            if result and result.get("candidate_senses"):
                self.store_result(conn, occ_id, lemma, result)
                extracted += 1
                by_lemma[lemma] = by_lemma.get(lemma, 0) + 1
            else:
                failed += 1

            if (i + 1) % 20 == 0:
                conn.commit()
                print(f"  {i+1}/{len(samples)}: {extracted} extracted, "
                      f"{failed} failed", flush=True)

            time.sleep(0.2)  # rate limiting

        conn.commit()

        # Summary
        print(f"\n{'='*60}")
        print(f"Extraction complete:")
        print(f"  Extracted: {extracted}")
        print(f"  Failed: {failed}")
        print(f"\n  By lemma:")
        for lemma, count in sorted(by_lemma.items()):
            print(f"    {TRANSLIT.get(lemma, lemma)}: {count}")

        senses = conn.execute("SELECT COUNT(*) FROM sense_inventory").fetchone()[0]
        evidence = conn.execute("SELECT COUNT(*) FROM evidence_spans").fetchone()[0]
        labels = conn.execute("SELECT COUNT(*) FROM candidate_labels").fetchone()[0]
        print(f"\n  Sense inventory: {senses} senses")
        print(f"  Candidate labels: {labels}")
        print(f"  Evidence spans: {evidence}")

        cost = (self.usage["input"] * 0.15 + self.usage["output"] * 0.60) / 1_000_000
        print(f"\n  API usage: {self.usage['input'] + self.usage['output']:,} tokens")
        print(f"  Cost: ${cost:.4f}")

        conn.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="WP2 evidence extraction pipeline")
    parser.add_argument("--target", type=int, default=80,
                        help="Occurrences per lemma to sample")
    parser.add_argument("--lemma", type=str, help="Extract for specific lemma only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-key", type=str)
    args = parser.parse_args()

    pipeline = ExtractionPipeline(api_key=args.api_key)
    pipeline.run(
        target=args.target,
        lemma_filter=args.lemma,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
