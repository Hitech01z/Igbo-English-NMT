"""
Dual Translation Model Registry

Loads both translation systems once when the API starts.

Supported directions:

    ig-en
    en-ig
"""

from inference.loader import load_translation_system
from inference.reverse_loader import load_reverse_translation_system


# ============================================================
# LOAD IGBO -> ENGLISH
# ============================================================

(
    ig_en_model,
    ig_tokenizer,
    en_tokenizer,
) = load_translation_system()


# ============================================================
# LOAD ENGLISH -> IGBO
# ============================================================

(
    en_ig_model,
    reverse_en_tokenizer,
    reverse_ig_tokenizer,
) = load_reverse_translation_system()


# ============================================================
# MODEL REGISTRY
# ============================================================

MODELS = {

    "ig-en": {

        "model": ig_en_model,

        "src": ig_tokenizer,

        "trg": en_tokenizer,

    },

    "en-ig": {

        "model": en_ig_model,

        "src": reverse_en_tokenizer,

        "trg": reverse_ig_tokenizer,

    },

}


# ============================================================
# GET MODEL
# ============================================================

def get_model(direction: str):

    if direction not in MODELS:

        raise ValueError(
            f"Unsupported translation direction: {direction}"
        )

    return MODELS[direction]