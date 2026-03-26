#!/usr/bin/env python3
"""
Generate Modern Greek bridge files for each passage using Google Translate,
then merge with manually transcribed Korto notes.

Outputs: bridge/<passage_id>_bridge.json
"""

import json
import time
from pathlib import Path

from deep_translator import GoogleTranslator

ROOT = Path(__file__).resolve().parent.parent
PASSAGES_DIR = ROOT / "passages"
BRIDGE_DIR = ROOT / "bridge"
BRIDGE_DIR.mkdir(exist_ok=True)

translator = GoogleTranslator(source="en", target="el")


def translate_sentences(sentences: list[dict]) -> list[dict]:
    """Translate each sentence via Google Translate with rate limiting."""
    results = []
    for s in sentences:
        try:
            gt = translator.translate(s["text"])
            time.sleep(0.3)  # gentle rate limit
        except Exception as e:
            gt = f"[GT ERROR: {e}]"
        results.append(
            {
                "index": s["index"],
                "english": s["text"],
                "google_translate_el": gt,
            }
        )
    return results


def build_bridge(passage_path: Path) -> dict:
    with open(passage_path, "r", encoding="utf-8") as f:
        passage = json.load(f)

    gt_sentences = translate_sentences(passage["sentences"])

    return {
        "passage_id": passage["id"],
        "title": passage["title"],
        "chapter": passage["chapter"],
        "korto_text": None,  # to be filled from PDF transcription
        "korto_notes": [],   # to be filled manually
        "google_translate": {
            "sentences": gt_sentences,
        },
        "idiom_notes": [],       # extracted clause-order / vocabulary observations
        "vocabulary_harvest": [], # MG vocabulary useful for AG composition
    }


def main():
    passage_files = sorted(PASSAGES_DIR.glob("0*.json"))
    print(f"Found {len(passage_files)} passage files\n")

    for pf in passage_files:
        print(f"Translating {pf.name}...")
        bridge = build_bridge(pf)
        out_path = BRIDGE_DIR / f"{bridge['passage_id']}_bridge.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bridge, f, ensure_ascii=False, indent=2)
        n = len(bridge["google_translate"]["sentences"])
        print(f"  → {out_path.name} ({n} sentences)\n")

    print("Done.")


if __name__ == "__main__":
    main()
