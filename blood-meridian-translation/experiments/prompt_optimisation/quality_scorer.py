"""
Dual-metric translation quality scorer.

Two components:
  1. Mechanical: Morpheus attestation + construction mismatch count
  2. LLM judge: Opus rates on a rubric designed for our translation project

Returns a combined score (higher = better).
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


# ====================================================================
# 1. Mechanical scoring
# ====================================================================

def mechanical_score(greek_text: str) -> dict:
    """Score using our existing checking pipeline.

    Returns dict with individual counts and a penalty score (lower = more errors).
    """
    import tempfile
    import shutil

    # Write to a temporary passage for the checkers
    tmp_id = "_scorer_tmp"
    tmp_dir = ROOT / "drafts" / tmp_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "primary.txt").write_text(greek_text + "\n", encoding="utf-8")

    results = {"unattested": 0, "neuter_plural": 0, "construction_mismatch": 0}

    try:
        from morpheus_check import check_passage, _save_cache
        issues = check_passage(tmp_id)
        _save_cache()
        results["unattested"] = sum(1 for i in issues if i["type"] == "unattested_word")
        results["neuter_plural"] = sum(1 for i in issues if i["type"] == "neuter_plural_verb")
    except Exception as e:
        pass

    # Clean up
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Penalty: each error costs points
    penalty = (
        results["unattested"] * 2 +      # invented words are bad
        results["neuter_plural"] * 5 +    # grammar violations are worse
        results["construction_mismatch"] * 3
    )

    results["penalty"] = penalty
    return results


# ====================================================================
# 2. LLM judge scoring
# ====================================================================

LLM_RUBRIC = """You are an expert evaluator of Ancient Greek prose translation. You are assessing a translation of Cormac McCarthy's Blood Meridian into Ancient Greek (Koine register with Attic vocabulary).

This is NOT a standard translation exercise. The goals of this project are:

1. **Preserve McCarthy's literary voice in Greek**: his parataxis, asyndeton, fragments, "and...and...and" coordination chains, and terse declarative style must survive. Over-subordination and over-participialisation are serious faults — they destroy the authorial voice.

2. **Produce genuinely readable classical Greek prose**: a competent reader of Thucydides, Xenophon, or the Septuagint should find this natural. Awkward calques from English, unattested vocabulary, and unidiomatic word order are faults.

3. **Appropriate register**: narrative prose should sound Thucydidean/Xenophontic; religious contexts should echo the Septuagint; philosophical speech should sound Platonic; dialogue should be natural and unforced. Register mismatches are faults.

4. **Correct grammar**: neuter plural + singular verb (Attic rule), correct preposition governance, correct relative pronoun agreement (gender/number matching the TRUE antecedent, not just the nearest noun), correct use of moods and tenses.

5. **Vocabulary quality**: prefer well-attested classical/Koine vocabulary. Loanwords and neologisms should be used only when no classical equivalent exists (and should be marked). Anachronisms like δολλάριον are serious errors. Polysemous words must match the correct sense in context (e.g. κῦμα for sea-swell, not οἴδημα).

6. **Preserve McCarthy's imagery**: his metaphors ("bones of trees assassinated", "silver filaments of cascades") should be rendered with equivalent force in Greek, not flattened into prosaic description.

## Scoring (0-10 per category):

Rate each of the following:

A. **Voice preservation** (0-10): Does this sound like McCarthy in Greek? Are his parataxis, fragments, and coordination chains preserved? Or has the translator over-subordinated, added unnecessary particles, expanded fragments into full sentences?

B. **Greek idiom** (0-10): Does this read as natural Greek prose? Would a classicist recognise this as competent Koine/Attic? Or does it feel like English wearing a Greek mask — calques, unidiomatic word order, awkward constructions?

C. **Register match** (0-10): Is the register appropriate to the content? Narrative, philosophical, religious, and colloquial passages should each find their natural Greek register.

D. **Grammar** (0-10): Correct agreement, case governance, mood/tense selection, relative pronoun antecedents?

E. **Vocabulary** (0-10): Well-attested, contextually appropriate, no anachronisms, correct sense selection for polysemous words?

F. **Imagery** (0-10): Are McCarthy's metaphors and vivid descriptions rendered with equivalent force, or flattened?

## Output format

Return ONLY a JSON object:
{
  "voice_preservation": <0-10>,
  "greek_idiom": <0-10>,
  "register_match": <0-10>,
  "grammar": <0-10>,
  "vocabulary": <0-10>,
  "imagery": <0-10>,
  "errors": ["<specific error 1>", "<specific error 2>", ...],
  "strengths": ["<specific strength 1>", ...],
  "total": <sum of all six scores, max 60>
}
"""


def llm_judge_score(en_text: str, greek_text: str) -> dict:
    """Have Opus judge the translation quality.

    Returns dict with per-category scores, errors list, and total.
    """
    import anthropic

    prompt = f"""{LLM_RUBRIC}

## English Source
{en_text}

## Greek Translation to Evaluate
{greek_text}

Return ONLY the JSON object. No other text."""

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    # Parse JSON
    try:
        clean = re.sub(r'^```json\s*', '', raw)
        clean = re.sub(r'\s*```$', '', clean)
        result = json.loads(clean)
        # Ensure total is computed
        categories = ["voice_preservation", "greek_idiom", "register_match",
                       "grammar", "vocabulary", "imagery"]
        result["total"] = sum(result.get(c, 0) for c in categories)
        return result
    except json.JSONDecodeError:
        return {"total": 30, "errors": ["could not parse judge response"], "raw": raw}


# ====================================================================
# 3. Combined scorer
# ====================================================================

def score_translation(en_text: str, greek_text: str) -> dict:
    """Combined mechanical + LLM judge score.

    Returns dict with:
      mechanical: {unattested, neuter_plural, penalty}
      llm_judge: {voice_preservation, ..., total}
      combined: single number (higher = better)
    """
    mech = mechanical_score(greek_text)
    judge = llm_judge_score(en_text, greek_text)

    # Combined: LLM total (0-60) minus mechanical penalty
    combined = judge.get("total", 30) - mech.get("penalty", 0)

    return {
        "mechanical": mech,
        "llm_judge": judge,
        "combined": combined,
    }
