"""
Stratified random passage sampler for prompt optimisation.

Draws ~100-word passages from Blood Meridian, one per stratum:
  1. Short asyndetic narration (parataxis, fragments, comma splices)
  2. Long subordinated sentences (relative clauses, temporal nesting)
  3. Modern vocabulary (guns, dollars, place names)
  4. Dialogue (speaker changes, colloquial)
  5. Philosophical/rhetorical (the Judge's speeches)
  6. Landscape/descriptive (visual, elemental, geological)
"""

import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BM_TEXT = ROOT / "passages" / "McCarthy_Blood_Meridian.txt"


def _load_paragraphs() -> list[str]:
    """Load BM text split into paragraphs."""
    text = BM_TEXT.read_text("utf-8")
    paras = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 50]
    return paras


def _extract_window(para: str, target_words: int = 100) -> str:
    """Extract a ~target_words window of complete sentences from a paragraph."""
    sents = re.split(r'(?<=[.!?])\s+', para)
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        return para[:500]

    # Pick a random starting sentence
    start = random.randint(0, max(0, len(sents) - 1))
    selected = []
    word_count = 0
    for i in range(start, len(sents)):
        selected.append(sents[i])
        word_count += len(sents[i].split())
        if word_count >= target_words:
            break
    # If we didn't get enough, prepend earlier sentences
    if word_count < target_words * 0.6:
        for i in range(start - 1, -1, -1):
            selected.insert(0, sents[i])
            word_count += len(sents[i].split())
            if word_count >= target_words:
                break

    return " ".join(selected)


# ====================================================================
# Stratum classifiers
# ====================================================================

def _is_short_asyndetic(para: str) -> bool:
    """Short sentences, comma splices, fragments."""
    sents = re.split(r'(?<=[.!?])\s+', para)
    if len(sents) < 3:
        return False
    avg_len = len(para.split()) / len(sents)
    comma_splices = para.count(", ") >= 3
    return avg_len < 12 and comma_splices


def _is_long_subordinated(para: str) -> bool:
    """Long sentences with relative/temporal clauses."""
    sents = re.split(r'(?<=[.!?])\s+', para)
    avg_len = len(para.split()) / max(1, len(sents))
    subordinators = len(re.findall(
        r'\b(where|when|while|which|who|whom|whose|that|although|because|since|until)\b',
        para, re.I
    ))
    return avg_len > 25 and subordinators >= 2


def _has_modern_vocab(para: str) -> bool:
    """Contains modern/anachronistic terms."""
    modern = ["dollar", "pistol", "rifle", "gun", "fort", "arkansas", "tennessee",
              "texas", "missouri", "kentucky", "saint louis", "new orleans",
              "mexican", "comanche", "delaware", "reverend", "whiskey", "hotel"]
    pl = para.lower()
    return sum(1 for m in modern if m in pl) >= 2


def _is_dialogue(para: str) -> bool:
    """Dialogue with speaker changes."""
    speech_verbs = len(re.findall(r'\b(said|cried|called|asked|replied|spat)\b', para, re.I))
    questions = para.count("?")
    short_sents = sum(1 for s in re.split(r'(?<=[.!?])\s+', para) if len(s.split()) < 8)
    return (speech_verbs >= 2 or questions >= 2) and short_sents >= 3


def _is_philosophical(para: str) -> bool:
    """The Judge's philosophical speeches."""
    markers = ["nature", "existence", "truth", "moral", "war", "god",
               "man's will", "creation", "suzerain", "autonomous",
               "the judge", "dispensation", "mystery", "reason"]
    pl = para.lower()
    hits = sum(1 for m in markers if m in pl)
    avg_len = len(para.split()) / max(1, len(re.split(r'(?<=[.!?])\s+', para)))
    return hits >= 2 and avg_len > 15


def _is_landscape(para: str) -> bool:
    """Visual, elemental, geological description."""
    markers = ["plain", "plains", "desert", "mountain", "ridge", "canyon",
               "mesa", "river", "sky", "sun", "moon", "stars", "wind",
               "rock", "sand", "dust", "horizon", "butte", "storm",
               "forest", "wood", "slope", "valley", "gorge", "torrent"]
    pl = para.lower()
    hits = sum(1 for m in markers if m in pl)
    # Landscape tends to be narration without dialogue
    has_speech = bool(re.search(r'\b(said|cried|asked)\b', para, re.I))
    return hits >= 3 and not has_speech


STRATA = [
    ("short_asyndetic", _is_short_asyndetic),
    ("long_subordinated", _is_long_subordinated),
    ("modern_vocabulary", _has_modern_vocab),
    ("dialogue", _is_dialogue),
    ("philosophical", _is_philosophical),
    ("landscape", _is_landscape),
]


def sample_passages(target_words: int = 100) -> dict[str, str]:
    """Draw one ~target_words passage per stratum.

    Returns dict of stratum_name → passage text.
    """
    paras = _load_paragraphs()
    random.shuffle(paras)

    # Classify all paragraphs
    classified = {name: [] for name, _ in STRATA}
    for para in paras:
        for name, classifier in STRATA:
            if classifier(para):
                classified[name].append(para)

    # Sample one from each
    result = {}
    for name, _ in STRATA:
        candidates = classified[name]
        if not candidates:
            # Fallback: pick any paragraph
            candidates = paras
        para = random.choice(candidates)
        result[name] = _extract_window(para, target_words)

    return result


if __name__ == "__main__":
    passages = sample_passages()
    for stratum, text in passages.items():
        wc = len(text.split())
        print(f"\n=== {stratum} ({wc} words) ===")
        print(text[:200] + "...")
