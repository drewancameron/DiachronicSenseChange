"""
Prompt configuration state space for adaptive prompt optimisation.

Each configuration is a dict of dimension → level. The proposal mechanism
picks a random dimension and moves it to a random alternative level.
"""

import random
from copy import deepcopy

# ====================================================================
# State space: each dimension and its possible levels
# ====================================================================

DIMENSIONS = {
    "structural_mirroring": ["off", "soft", "hard"],
    "parallel_examples": ["off", "2_style_models", "4_style_models", "per_sentence"],
    "vocab_guidance": ["off", "polysemy_only", "polysemy_and_corpus"],
    "glossary": ["off", "soft", "hard"],
    "particle_suppression": ["off", "soft", "hard"],
    "construction_labels": ["off", "labels_only", "labels_and_taxonomy"],
    "rules_document": ["off", "core_only", "full"],
    "register_instruction": ["minimal", "moderate", "detailed"],
}

# Starting points
BARE_CONFIG = {
    "structural_mirroring": "off",
    "parallel_examples": "off",
    "vocab_guidance": "off",
    "glossary": "off",
    "particle_suppression": "off",
    "construction_labels": "off",
    "rules_document": "off",
    "register_instruction": "minimal",
}

CURRENT_BEST_CONFIG = {
    "structural_mirroring": "soft",
    "parallel_examples": "2_style_models",
    "vocab_guidance": "polysemy_only",
    "glossary": "soft",
    "particle_suppression": "soft",
    "construction_labels": "labels_only",
    "rules_document": "core_only",
    "register_instruction": "moderate",
}


def propose(config: dict) -> dict:
    """Propose a new config by changing one random dimension to a random other level."""
    new = deepcopy(config)
    dim = random.choice(list(DIMENSIONS.keys()))
    current_level = new[dim]
    alternatives = [l for l in DIMENSIONS[dim] if l != current_level]
    new[dim] = random.choice(alternatives)
    return new


def config_to_key(config: dict) -> str:
    """Hashable string representation of a config."""
    return "|".join(f"{k}={v}" for k, v in sorted(config.items()))


def key_to_config(key: str) -> dict:
    return dict(item.split("=") for item in key.split("|"))
