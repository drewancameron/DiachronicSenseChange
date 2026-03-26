#!/usr/bin/env python3
"""
LLM-based evidence extraction from translations and scholarly apparatus.

This is the core of the 'scholarship-distilled supervision' method.
Given a Greek passage containing a target lemma, and aligned English
translation(s) and notes, the LLM extracts structured sense evidence.

The LLM does NOT invent senses — it identifies which sense(s) existing
scholarship supports, cites the exact evidential wording, and records
uncertainty explicitly.
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "db" / "diachronic.db"
PROMPTS_DIR = Path(__file__).parent.parent / "config" / "prompts"

# ── Prompt templates ────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """\
You are a research assistant for a historical semantics project on Ancient Greek.
Your role is to extract structured sense evidence from translations, translator's
notes, commentaries, and lexicographic sources.

CRITICAL RULES:
1. You are extracting evidence, not inventing interpretations.
2. Every sense claim must cite the exact wording from the source material.
3. If the evidence is ambiguous, record ALL plausible senses with confidence levels.
4. Distinguish between DIRECT evidence (the translation clearly renders the word
   with a specific meaning) and INFERENTIAL evidence (the meaning can be inferred
   from context or notes).
5. Flag archaizing, poetic, or register-marked usage when detectable.
6. If you cannot determine the sense, say so. Do not guess.
7. Record the evidence type: translation_rendering, translator_note,
   commentary_discussion, lexicon_gloss, or scholion.
"""

EXTRACTION_USER_TEMPLATE = """\
## Target word
- **Lemma**: {lemma} ({lemma_transliteration})
- **Surface form in passage**: {surface_form}

## Greek passage
**Reference**: {reference}
**Text**: {greek_text}

## Aligned materials
{aligned_materials}

## Known sense inventory for {lemma}
{sense_inventory}

## Task
Analyze the aligned materials and determine which sense(s) of **{lemma}** are
supported by the evidence for this specific occurrence.

Return a JSON object with this structure:
```json
{{
  "occurrence_reference": "{reference}",
  "lemma": "{lemma}",
  "surface_form": "{surface_form}",
  "candidate_senses": [
    {{
      "sense_label": "the sense label from the inventory",
      "confidence": 0.0-1.0,
      "is_primary": true/false,
      "evidence": [
        {{
          "evidence_text": "exact quoted span from translation/note",
          "evidence_type": "translation_rendering|translator_note|commentary_discussion|lexicon_gloss",
          "source_identity": "translator or commentator name",
          "directness": "direct|inferential|contextual"
        }}
      ]
    }}
  ],
  "register": {{
    "label": "unmarked|poetic_elevated|archaizing|homericizing|quoted_allusive|scholastic|colloquial|technical|uncertain",
    "confidence": 0.0-1.0,
    "evidence_text": "if any register evidence exists"
  }},
  "notes": "any additional observations, especially about ambiguity or bridge cases",
  "extraction_confidence": 0.0-1.0
}}
```

Be conservative with confidence scores. A score of 0.9+ should mean the evidence
is unambiguous. A score of 0.5-0.7 means the evidence is suggestive but not decisive.
Below 0.5 means the evidence is weak or indirect.
"""

ALIGNMENT_QUERY = """\
SELECT a.alignment_id, a.aligned_text, a.alignment_type,
       a.alignment_confidence, t.translator, c.author_name as commentator
FROM alignments a
LEFT JOIN translations t ON a.translation_id = t.translation_id
LEFT JOIN commentaries c ON a.commentary_id = c.commentary_id
WHERE a.passage_id = ?
ORDER BY a.alignment_confidence DESC
"""


def format_aligned_materials(alignments: list[tuple]) -> str:
    """Format aligned translations and notes for the prompt."""
    if not alignments:
        return "(No aligned materials available for this passage)"

    parts = []
    for alignment_id, text, atype, confidence, translator, commentator in alignments:
        source = translator or commentator or "unknown"
        parts.append(
            f"### {atype} (by {source}, confidence: {confidence:.2f})\n{text}"
        )
    return "\n\n".join(parts)


def format_sense_inventory(conn: sqlite3.Connection, lemma: str) -> str:
    """Format the sense inventory for a lemma."""
    rows = conn.execute(
        "SELECT sense_label, sense_description FROM sense_inventory WHERE lemma = ?",
        (lemma,),
    ).fetchall()

    if not rows:
        return "(No sense inventory defined yet — propose senses based on the evidence)"

    parts = []
    for label, desc in rows:
        parts.append(f"- **{label}**: {desc}")
    return "\n".join(parts)


def build_extraction_prompt(
    conn: sqlite3.Connection,
    occurrence_id: int,
) -> dict:
    """Build the full extraction prompt for an occurrence."""

    # Get occurrence details
    row = conn.execute("""
        SELECT o.lemma, o.greek_context, o.period, o.genre,
               t.surface_form, p.reference, p.greek_text
        FROM occurrences o
        JOIN tokens t ON o.token_id = t.token_id
        JOIN passages p ON o.passage_id = p.passage_id
        WHERE o.occurrence_id = ?
    """, (occurrence_id,)).fetchone()

    if not row:
        return None

    lemma, context, period, genre, surface_form, reference, greek_text = row

    # Get aligned materials
    passage_id = conn.execute(
        "SELECT passage_id FROM occurrences WHERE occurrence_id = ?",
        (occurrence_id,),
    ).fetchone()[0]

    alignments = conn.execute(ALIGNMENT_QUERY, (passage_id,)).fetchall()
    aligned_text = format_aligned_materials(alignments)
    sense_inv = format_sense_inventory(conn, lemma)

    # Build transliteration lookup
    translit_map = {
        "κόσμος": "kosmos", "λόγος": "logos", "ψυχή": "psyche",
        "ἀρετή": "arete", "δίκη": "dike", "τέχνη": "techne",
        "νόμος": "nomos", "φύσις": "physis", "δαίμων": "daimon",
        "σῶμα": "soma", "θεός": "theos", "χάρις": "charis",
    }
    translit = translit_map.get(lemma, lemma)

    user_prompt = EXTRACTION_USER_TEMPLATE.format(
        lemma=lemma,
        lemma_transliteration=translit,
        surface_form=surface_form,
        reference=reference,
        greek_text=greek_text or context,
        aligned_materials=aligned_text,
        sense_inventory=sense_inv,
    )

    return {
        "system": EXTRACTION_SYSTEM_PROMPT,
        "user": user_prompt,
        "metadata": {
            "occurrence_id": occurrence_id,
            "lemma": lemma,
            "reference": reference,
            "period": period,
            "genre": genre,
        },
    }


def store_extraction_result(
    conn: sqlite3.Connection,
    occurrence_id: int,
    result: dict,
    model_version: str = "claude-sonnet-4-20250514",
) -> None:
    """Store an LLM extraction result in the database."""

    for candidate in result.get("candidate_senses", []):
        sense_label = candidate["sense_label"]

        # Find or create sense in inventory
        sense_row = conn.execute(
            "SELECT sense_id FROM sense_inventory WHERE lemma = ? AND sense_label = ?",
            (result["lemma"], sense_label),
        ).fetchone()

        if sense_row:
            sense_id = sense_row[0]
        else:
            cur = conn.execute(
                "INSERT INTO sense_inventory (lemma, sense_label) VALUES (?, ?)",
                (result["lemma"], sense_label),
            )
            sense_id = cur.lastrowid

        # Create candidate label
        cur = conn.execute(
            """INSERT INTO candidate_labels
               (occurrence_id, sense_id, confidence, is_primary)
               VALUES (?, ?, ?, ?)""",
            (occurrence_id, sense_id, candidate["confidence"],
             1 if candidate.get("is_primary") else 0),
        )
        label_id = cur.lastrowid

        # Store evidence spans
        for ev in candidate.get("evidence", []):
            conn.execute(
                """INSERT INTO evidence_spans
                   (label_id, evidence_text, evidence_type,
                    source_identity, directness)
                   VALUES (?, ?, ?, ?, ?)""",
                (label_id, ev["evidence_text"], ev["evidence_type"],
                 ev.get("source_identity"), ev.get("directness")),
            )

    # Store register label
    register = result.get("register", {})
    if register.get("label"):
        conn.execute(
            """INSERT INTO register_labels
               (occurrence_id, register, confidence, evidence_text)
               VALUES (?, ?, ?, ?)""",
            (occurrence_id, register["label"], register.get("confidence"),
             register.get("evidence_text")),
        )

    # Log provenance
    conn.execute(
        """INSERT INTO provenance_events
           (entity_type, entity_id, action, agent, model_version, notes)
           VALUES ('occurrence', ?, 'evidence_extracted', ?, ?, ?)""",
        (occurrence_id, f"llm:{model_version}", model_version,
         result.get("notes")),
    )

    conn.commit()


def main():
    """Generate extraction prompts for review (dry run mode)."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate LLM evidence extraction prompts"
    )
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--lemma", type=str, help="Generate for specific lemma")
    parser.add_argument("--limit", type=int, default=5,
                        help="Max prompts to generate")
    parser.add_argument("--output", type=Path, help="Write prompts to file")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    query = "SELECT occurrence_id, lemma FROM occurrences"
    params = []
    if args.lemma:
        query += " WHERE lemma = ?"
        params.append(args.lemma)
    query += f" LIMIT {args.limit}"

    occurrences = conn.execute(query, params).fetchall()

    if not occurrences:
        print("No occurrences found. Run find_occurrences.py first.")
        return

    prompts = []
    for occ_id, lemma in occurrences:
        prompt = build_extraction_prompt(conn, occ_id)
        if prompt:
            prompts.append(prompt)
            print(f"\n{'='*60}")
            print(f"Occurrence {occ_id}: {lemma} @ {prompt['metadata']['reference']}")
            print(f"Period: {prompt['metadata']['period']}")
            print(f"\n--- USER PROMPT (first 500 chars) ---")
            print(prompt["user"][:500])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {len(prompts)} prompts to {args.output}")

    conn.close()


if __name__ == "__main__":
    main()
