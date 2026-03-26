#!/usr/bin/env python3
"""
Merge Korto transcriptions and idiom notes into the GT bridge files.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRIDGE_DIR = ROOT / "bridge"
KORTO_FILE = BRIDGE_DIR / "korto_transcriptions.json"


def load_korto():
    with open(KORTO_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Remove metadata key
    return {k: v for k, v in data.items() if not k.startswith("_")}


def merge():
    korto = load_korto()

    for bridge_path in sorted(BRIDGE_DIR.glob("*_bridge.json")):
        with open(bridge_path, "r", encoding="utf-8") as f:
            bridge = json.load(f)

        pid = bridge["passage_id"]
        if pid in korto:
            k = korto[pid]
            bridge["korto_text"] = k.get("korto_text") or k.get("korto_text_fragment")
            bridge["korto_notes"] = k.get("korto_notes", [])

            # Generate idiom notes from Korto observations
            idiom_notes = []
            for note in bridge["korto_notes"]:
                if "note" in note:
                    idiom_notes.append(note["note"])
            bridge["idiom_notes"] = idiom_notes

            # Harvest vocabulary from Korto that maps directly AG
            vocab = []
            for note in bridge["korto_notes"]:
                n = note.get("note", "")
                if "AG" in n or "= AG" in n or "ancient" in n.lower():
                    vocab.append({
                        "english": note["english"],
                        "korto_mg": note["korto"],
                        "ag_hint": n,
                    })
            bridge["vocabulary_harvest"] = vocab

        with open(bridge_path, "w", encoding="utf-8") as f:
            json.dump(bridge, f, ensure_ascii=False, indent=2)

        has_korto = "✓" if pid in korto else "–"
        print(f"  {bridge_path.name}  korto={has_korto}")


if __name__ == "__main__":
    merge()
