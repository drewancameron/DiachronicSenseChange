#!/usr/bin/env python3
"""
Document segmentation using OpenAI API (Option C hybrid, two-pass).

Pass 1 (nano, cheap): Coarse classification of each chunk as
  "mostly_greek", "mostly_english", "mixed", "commentary", "paratext"
Pass 2 (mini, only on mixed/commentary chunks): Full segmentation
  into typed segments.

Also handles note reclassification (nano) for TEI-extracted notes.
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("openai package not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)

import yaml

DB_PATH = Path(__file__).parent.parent / "db" / "diachronic.db"
CLEANED_DIR = Path(__file__).parent.parent / "corpus" / "cleaned" / "manual_text"
SOURCES_YAML = Path(__file__).parent.parent / "config"

# Models
NANO = "gpt-4.1-nano"
MINI = "gpt-4.1-mini"

# ── Segment types ──────────────────────────────────────────────

SEGMENT_TYPES = [
    "greek_text", "translation_text", "footnote",
    "commentary_note", "lexical_note", "apparatus",
    "paratext", "noise",
]

COARSE_TYPES = [
    "mostly_greek", "mostly_english", "mixed",
    "commentary", "paratext", "noise",
]

# ── Prompts ────────────────────────────────────────────────────

PASS1_SYSTEM = """\
You are a document triage assistant for a classical philology project.

You will be given a chunk of text extracted from a PDF. Classify the ENTIRE chunk
as one of these categories:

- **mostly_greek**: The chunk is predominantly Ancient Greek running text (may have line numbers or page headers)
- **mostly_english**: The chunk is predominantly English translation of a Greek text
- **mixed**: The chunk contains a significant MIX of Greek text and English text (e.g., facing-page editions, dual text, interlinear)
- **commentary**: The chunk contains scholarly commentary, footnotes, or lexical discussion (English text discussing Greek words/passages)
- **paratext**: Table of contents, preface, introduction, publisher info, title pages, bibliography, index
- **noise**: OCR artefacts, blank pages, unreadable content, watermarks

Return JSON: {"type": "<type>", "confidence": 0.0-1.0, "has_greek_chars": true/false}

Be quick and decisive. Look for Greek Unicode characters (α-ω, accented forms) to determine if Greek text is present.
"""

PASS2_SYSTEM = """\
You are a document segmentation assistant for a classical philology research project.

You will be given a chunk of text that contains a MIX of content types (Greek text,
English translation, scholarly commentary, footnotes, etc.).

Segment the text into typed sections. Each section should be classified as:

- **greek_text**: Running Ancient Greek text (primary source)
- **translation_text**: English translation of a Greek passage
- **footnote**: Numbered or marked translator's/editor's footnote
- **commentary_note**: Scholarly commentary discussing a passage
- **lexical_note**: Discussion of a specific word's meaning, etymology, or usage
- **apparatus**: Textual variants, manuscript readings, critical apparatus
- **paratext**: Page headers/footers, section titles
- **noise**: OCR artefacts, watermarks

Return a JSON object with a "segments" array. Each segment has:
- "type": one of the types above
- "text": the exact text (preserve original wording)
- "reference": any passage reference visible (e.g. "Il.1.1", "Rep.327a"), or null
- "notes": any relevant observation (optional)

IMPORTANT:
- Preserve original text exactly — do not paraphrase or translate
- Keep segments reasonably sized (don't create one segment per word)
- Greek text and English text should always be separate segments
"""

RECLASSIFY_SYSTEM = """\
You are a note classification assistant for a classical philology project.

Classify each note as one of:
- **footnote**: A translator's or editor's footnote (typically short, numbered)
- **commentary_note**: Scholarly discussion of a passage's meaning or context
- **lexical_note**: Discussion of a specific word's meaning, usage, or semantic history
- **apparatus**: Textual variant, manuscript reading, or critical apparatus note
- **paratext**: Editorial/publishing information

Return a JSON object with a "classifications" array, one entry per note,
each with "type" and "confidence" fields.
"""

# ── Pipeline ───────────────────────────────────────────────────


class SegmentationPipeline:
    def __init__(self, api_key: str | None = None, db_path: Path = DB_PATH):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=60.0,
        )
        self.db_path = db_path
        self.usage = {
            "nano_input": 0, "nano_output": 0,
            "mini_input": 0, "mini_output": 0,
        }

    def _call(self, model: str, system: str, user: str,
              max_tokens: int = 4096) -> str:
        """API call with retry and hard timeout."""
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("API call exceeded 60s hard timeout")

        for attempt in range(3):
            try:
                # Hard 60s timeout via signal
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(60)
                try:
                    resp = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=0.1,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

                prefix = "nano" if "nano" in model else "mini"
                if resp.usage:
                    self.usage[f"{prefix}_input"] += resp.usage.prompt_tokens
                    self.usage[f"{prefix}_output"] += resp.usage.completion_tokens
                return resp.choices[0].message.content
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"    API error: {e}, retry in {wait}s...",
                          flush=True)
                    time.sleep(wait)
                else:
                    print(f"    API failed after 3 attempts: {e}",
                          flush=True)
                    return '{"type": "noise", "confidence": 0.0}'

    # ── Phase 1: Note reclassification (nano) ──────────────────

    def reclassify_notes(self, limit: int | None = None):
        """Reclassify ambiguous TEI notes using nano."""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT note_id, note_text, note_type, reference, language
            FROM extracted_notes
            WHERE classification_method = 'rule_based'
            AND length(note_text) > 50
        """
        if limit:
            query += f" LIMIT {limit}"
        rows = conn.execute(query).fetchall()

        if not rows:
            print("No notes to reclassify")
            conn.close()
            return

        print(f"Reclassifying {len(rows)} notes with {NANO}...")
        batch_size = 20
        updated = 0

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            batch_text = "\n\n".join(
                f"--- Note {j+1} ---\n"
                f"Reference: {r[3] or '?'}\n"
                f"Current type: {r[2]}\n"
                f"Text: {r[1][:400]}"
                for j, r in enumerate(batch)
            )
            prompt = (
                f"Classify each of the following {len(batch)} notes.\n\n"
                f"{batch_text}"
            )
            try:
                raw = self._call(NANO, RECLASSIFY_SYSTEM, prompt)
                parsed = json.loads(raw)
                classifications = parsed.get("classifications",
                                  parsed.get("notes",
                                  parsed.get("results", [])))
                for j, r in enumerate(batch):
                    if j < len(classifications):
                        c = classifications[j]
                        new_type = c.get("type", r[2])
                        if new_type in ("footnote", "commentary_note",
                                        "lexical_note", "apparatus", "paratext"):
                            conn.execute(
                                """UPDATE extracted_notes
                                   SET note_type = ?, classification_method = 'openai_nano'
                                   WHERE note_id = ?""",
                                (new_type, r[0]),
                            )
                            updated += 1
            except Exception as e:
                print(f"    Batch {i//batch_size} failed: {e}")

            if (i // batch_size + 1) % 20 == 0:
                conn.commit()
                print(f"    Processed {min(i+batch_size, len(rows))}/{len(rows)} notes...")

        conn.commit()
        conn.close()
        print(f"  Updated {updated} notes")

    # ── Phase 2: Two-pass PDF segmentation ─────────────────────

    def _pass1_classify(self, chunk_text: str) -> dict:
        """Pass 1 (nano): Coarse classify a chunk."""
        # Truncate to ~2000 chars for fast classification
        sample = chunk_text[:2000]
        if len(chunk_text) > 2000:
            sample += f"\n\n[...truncated, {len(chunk_text)} total chars...]"

        raw = self._call(NANO, PASS1_SYSTEM, sample)
        result = json.loads(raw)
        return {
            "type": result.get("type", "noise"),
            "confidence": result.get("confidence", 0.5),
            "has_greek": result.get("has_greek_chars", False),
        }

    def _pass2_segment(self, chunk_text: str, meta: dict) -> list[dict]:
        """Pass 2 (mini): Full segmentation of a mixed/commentary chunk."""
        context = (
            f"Source: {meta.get('title', '?')}\n"
            f"Author: {meta.get('author', '?')}\n"
            f"Type: {meta.get('type', '?')}\n"
        )
        prompt = f"## Source\n{context}\n\n## Text to segment\n{chunk_text[:8000]}"

        try:
            raw = self._call(MINI, PASS2_SYSTEM, prompt)
            parsed = json.loads(raw)
            segments = parsed.get("segments", [])
            for seg in segments:
                if seg.get("type") not in SEGMENT_TYPES:
                    seg["type"] = "noise"
            return segments
        except Exception as e:
            print(f"    Segmentation failed: {e}")
            return [{"type": "noise", "text": chunk_text[:200],
                     "notes": f"failed: {e}"}]

    def process_pdf(self, json_path: Path, meta: dict) -> dict:
        """Two-pass segmentation of a single PDF."""
        with open(json_path) as f:
            doc = json.load(f)

        source_file = doc["source_file"]
        rights = doc.get("rights_status", "unknown")
        may_redist = 1 if doc.get("may_redistribute", False) else 0
        chunks = doc["chunks"]

        stats = {"total": len(chunks), "pass1": {}, "pass2_sent": 0, "segments": 0}

        conn = sqlite3.connect(self.db_path)
        self._ensure_tables(conn)

        # Skip if already >=90% segmented (handles timeout edge cases)
        existing = conn.execute(
            "SELECT COUNT(*) FROM segmented_content WHERE source_file = ?",
            (source_file,),
        ).fetchone()[0]
        threshold = int(len(chunks) * 0.90)
        if existing >= threshold:
            print(f"  Already segmented ({existing} segments, {existing*100//len(chunks)}%), skipping")
            conn.close()
            return {"total": len(chunks), "pass1": {"skipped": len(chunks)},
                    "pass2_sent": 0, "segments": existing}

        # Clear partial results for this source (re-do from scratch)
        if existing > 0:
            print(f"  Clearing {existing} partial segments, restarting...")
            conn.execute(
                "DELETE FROM segmented_content WHERE source_file = ?",
                (source_file,),
            )

        for i, chunk in enumerate(chunks):
            text = chunk["combined_text"]

            # ── Pass 1: coarse classification (nano) ──
            p1 = self._pass1_classify(text)
            ctype = p1["type"]
            stats["pass1"][ctype] = stats["pass1"].get(ctype, 0) + 1

            if ctype in ("mostly_greek", "mostly_english"):
                # Simple case: store as a single segment, no mini needed
                seg_type = "greek_text" if ctype == "mostly_greek" else "translation_text"
                # Strip page break markers for cleaner storage
                clean_text = text
                for marker in ["--- PAGE BREAK ---", "[Page "]:
                    # Keep the text but note it's multi-page
                    pass

                conn.execute(
                    """INSERT INTO segmented_content
                       (source_file, source_page_start, source_page_end,
                        segment_type, text_content, reference,
                        rights_status, may_redistribute,
                        classification_method, notes)
                       VALUES (?, ?, ?, ?, ?, NULL, ?, ?, 'pass1_nano', ?)""",
                    (source_file, chunk["start_page"], chunk["end_page"],
                     seg_type, text, rights, may_redist,
                     f"coarse: {ctype} (conf: {p1['confidence']:.2f})"),
                )
                stats["segments"] += 1

            elif ctype == "paratext":
                conn.execute(
                    """INSERT INTO segmented_content
                       (source_file, source_page_start, source_page_end,
                        segment_type, text_content, reference,
                        rights_status, may_redistribute,
                        classification_method, notes)
                       VALUES (?, ?, ?, 'paratext', ?, NULL, ?, ?, 'pass1_nano', NULL)""",
                    (source_file, chunk["start_page"], chunk["end_page"],
                     text, rights, may_redist),
                )
                stats["segments"] += 1

            elif ctype == "noise":
                # Skip noise chunks entirely
                pass

            else:
                # ── Pass 2: full segmentation (mini) ──
                stats["pass2_sent"] += 1
                segments = self._pass2_segment(text, meta)

                for seg in segments:
                    conn.execute(
                        """INSERT INTO segmented_content
                           (source_file, source_page_start, source_page_end,
                            segment_type, text_content, reference,
                            rights_status, may_redistribute,
                            classification_method, notes)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pass2_mini', ?)""",
                        (source_file, chunk["start_page"], chunk["end_page"],
                         seg["type"], seg.get("text", ""),
                         seg.get("reference"), rights, may_redist,
                         seg.get("notes")),
                    )
                    stats["segments"] += 1

                time.sleep(0.2)  # Rate limiting for mini

            # Progress
            if (i + 1) % 50 == 0:
                conn.commit()
                print(f"    {i+1}/{len(chunks)} chunks "
                      f"({stats['pass2_sent']} sent to mini)...",
                      flush=True)

        conn.commit()
        conn.close()
        return stats

    def process_all_pdfs(self, priority_files: list[str],
                         limit_files: int | None = None):
        """Process priority PDF files with two-pass segmentation."""
        if limit_files:
            priority_files = priority_files[:limit_files]

        # Load all source metadata
        all_meta = {}
        for yaml_file in SOURCES_YAML.glob("*sources*.yaml"):
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
            for s in config.get("sources", []):
                all_meta[s.get("filename", "")] = s

        conn = sqlite3.connect(self.db_path)
        self._ensure_tables(conn)
        conn.close()

        total_stats = {"files": 0, "chunks": 0, "pass2_sent": 0, "segments": 0}

        for stem in priority_files:
            # Find the JSON file (fuzzy match on stem)
            matches = list(CLEANED_DIR.glob(f"*{stem}*.json"))
            if not matches:
                print(f"\n  [skip] No extracted text for: {stem}")
                continue

            json_path = matches[0]
            with open(json_path) as f:
                doc = json.load(f)

            source_file = doc["source_file"]
            meta = all_meta.get(source_file, {"title": stem})
            est_tokens = doc.get("est_tokens", 0)

            print(f"\n{'─'*60}")
            print(f"  {meta.get('title', stem)}")
            print(f"  Chunks: {doc['total_chunks']}, Est tokens: {est_tokens:,}")

            stats = self.process_pdf(json_path, meta)

            print(f"  Pass 1 breakdown: {stats['pass1']}")
            print(f"  Chunks sent to mini (pass 2): {stats['pass2_sent']}")
            print(f"  Total segments stored: {stats['segments']}")

            total_stats["files"] += 1
            total_stats["chunks"] += stats["total"]
            total_stats["pass2_sent"] += stats["pass2_sent"]
            total_stats["segments"] += stats["segments"]

        print(f"\n{'='*60}")
        print(f"All files processed:")
        print(f"  Files: {total_stats['files']}")
        print(f"  Total chunks: {total_stats['chunks']}")
        print(f"  Chunks needing mini (pass 2): {total_stats['pass2_sent']}")
        print(f"  Total segments: {total_stats['segments']}")

    def _ensure_tables(self, conn):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS segmented_content (
                segment_id INTEGER PRIMARY KEY,
                source_file TEXT NOT NULL,
                source_page_start INTEGER,
                source_page_end INTEGER,
                segment_type TEXT CHECK(segment_type IN (
                    'greek_text', 'translation_text', 'footnote',
                    'commentary_note', 'lexical_note', 'apparatus',
                    'paratext', 'noise'
                )),
                text_content TEXT NOT NULL,
                reference TEXT,
                rights_status TEXT,
                may_redistribute INTEGER DEFAULT 0,
                classification_method TEXT,
                notes TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_seg_type
                ON segmented_content(segment_type)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_seg_source
                ON segmented_content(source_file)
        """)

    def print_usage(self):
        """Print API usage and cost."""
        ni, no = self.usage["nano_input"], self.usage["nano_output"]
        mi, mo = self.usage["mini_input"], self.usage["mini_output"]
        nc = (ni * 0.10 + no * 0.40) / 1_000_000
        mc = (mi * 0.15 + mo * 0.60) / 1_000_000

        print(f"\n{'='*60}")
        print(f"API Usage")
        print(f"{'='*60}")
        print(f"  Nano:  {ni+no:>10,} tokens  ${nc:.4f}")
        print(f"  Mini:  {mi+mo:>10,} tokens  ${mc:.4f}")
        print(f"  Total: {ni+no+mi+mo:>10,} tokens  ${nc+mc:.4f}")


# ── Priority files ─────────────────────────────────────────────

# Files to segment, ordered by value.
# Stems are matched fuzzily against JSON filenames.
PRIORITY_FILES = [
    # Delphi editions (Greek + English + dual)
    "Delphi_Complete_Works_of_Aristotle",
    "Delphi_Complete_Works_of_Thucydides",
    "Delphi_Complete_Works_of_Euripides",
    "Delphi_Complete_Works_of_Aristophanes",
    "Delphi_Complete_Works_of_Polybius",
    "Delphi_Complete_Works_of_Apollonius",
    "Delphi_Complete_Works_of_Dio_Chrysostom",
    "Delphi_Complete_Works_of_Appian",
    "Complete_Works_of_Diodorus_Siculus",
    "Works_of_Nonnus",
    "works_of_Hesiod",

    # Septuagint (PD, fills Koine gap)
    "Septuagint_-_Lancelot",

    # Commentaries with Greek text
    "Dodds-Euripides-Bacchae",
    "Aeschylus-Agamemnon-Aeschylus-Z-Library",
    "Himmelhoch",
    "Agamemnon_of_Aeschylu_-_David_Raeburn",
    "Basel_Commentary",
    "Plato_-_David_Sansone",
    "plato_the-republic",
    "Leaf-W-Commentary",

    # Dual language / Loeb
    "euripides_bacchae_-_a_dual_language",
    "If_Not_Winter",
    "Plutarchs_Lives_-_Plutarch",
    "Plutarch_-_J_L_Moles",
    "Odyssey_Books_13-24",
    "Lucian_Volume_VI",
    "Alcestis",

    # Key English translations (PD Jowett)
    "Sophocles_0561",
    "Charmides_-_Plato",
    "Crito_-_Plato",
    "Euthyphro_-_Plato",
    "Gorgias_-_Plato",
    "Phaedo_-_Plato",
    "Protagoras_-_Plato",
    "Menexenus_-_Plato",

    # Additional English translations
    "The_Republic_-_Plato",
    "Symposium_-_Plato",
    "Greek_Lives",
    "Fall_of_the_Roman_Republic",
    "Memories_of_Socrates",
    "Theogony__Works_and_Days",
    "Daphnis_and_Chloe",
]

# Skip: LSJ (too large), Sidonius (Latin), monographs, methodology papers


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Two-pass document segmentation with OpenAI"
    )
    parser.add_argument("--phase", choices=["notes", "pdfs", "all"],
                        default="all")
    parser.add_argument("--limit-notes", type=int)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--api-key", type=str)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        # Estimate costs
        conn = sqlite3.connect(DB_PATH)
        try:
            note_count = conn.execute(
                "SELECT COUNT(*) FROM extracted_notes "
                "WHERE classification_method = 'rule_based' AND length(note_text) > 50"
            ).fetchone()[0]
        except Exception:
            note_count = 0
        conn.close()

        note_tokens = note_count * 80
        note_cost = (note_tokens * 0.10 + note_count * 10 * 0.40) / 1_000_000

        # Estimate PDF costs with two-pass savings
        total_chunks = 0
        total_tokens = 0
        for stem in PRIORITY_FILES:
            matches = list(CLEANED_DIR.glob(f"*{stem}*.json"))
            if matches:
                with open(matches[0]) as f:
                    doc = json.load(f)
                total_chunks += doc["total_chunks"]
                total_tokens += doc.get("est_tokens", 0)

        # Pass 1: all chunks go through nano (~500 tokens each)
        p1_tokens = total_chunks * 600
        p1_cost = (p1_tokens * 0.10 + total_chunks * 20 * 0.40) / 1_000_000

        # Pass 2: estimate ~30% of chunks need mini (mixed/commentary)
        p2_fraction = 0.30
        p2_input = int(total_tokens * p2_fraction)
        p2_output = int(p2_input * 0.25)
        p2_cost = (p2_input * 0.15 + p2_output * 0.60) / 1_000_000

        print(f"=== DRY RUN COST ESTIMATE ===")
        print(f"")
        print(f"Phase 1: Note reclassification (nano)")
        print(f"  Notes to reclassify: {note_count:,}")
        print(f"  Est cost: ${note_cost:.4f}")
        print(f"")
        print(f"Phase 2: Two-pass PDF segmentation")
        print(f"  Priority files: {len(PRIORITY_FILES)}")
        print(f"  Total chunks: {total_chunks:,}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Pass 1 (nano, all chunks): ${p1_cost:.4f}")
        print(f"  Pass 2 (mini, ~{p2_fraction:.0%} of chunks): ${p2_cost:.4f}")
        print(f"")
        print(f"  TOTAL ESTIMATED COST: ${note_cost + p1_cost + p2_cost:.2f}")
        return

    pipeline = SegmentationPipeline(api_key=args.api_key)

    try:
        if args.phase in ("notes", "all"):
            print("=" * 60)
            print("PHASE 1: Note reclassification (nano)")
            print("=" * 60)
            pipeline.reclassify_notes(limit=args.limit_notes)

        if args.phase in ("pdfs", "all"):
            print("\n" + "=" * 60)
            print("PHASE 2: Two-pass PDF segmentation")
            print("=" * 60)
            pipeline.process_all_pdfs(
                PRIORITY_FILES,
                limit_files=args.limit_files,
            )
    finally:
        pipeline.print_usage()


if __name__ == "__main__":
    main()
