import torch

from inference.dual_loader import get_model
from inference.decoder import greedy_decode, decode_tokens

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# ENCODE INPUT SENTENCE
# ============================================================

def encode_sentence(text, tokenizer):

    text = text.lower().strip()

    if hasattr(tokenizer, "encode"):

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=True
        )

    else:
        raise RuntimeError(
            "Tokenizer does not implement encode()."
        )

    return torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )


import re


def clean_translation(text: str):

    text = text.strip()

    # Remove repeated spaces
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # Remove duplicated punctuation
    text = re.sub(r"([.!?]){2,}", r"\1", text)

    # Remove common hallucinated trailing words
    trailing_words = [
        "now",
        "today",
        "tomorrow",
        "later"
    ]

    words = text.split()

    while words and words[-1].lower().strip(".,!?") in trailing_words:
        words.pop()

    text = " ".join(words)

    # Ensure sentence ends properly
    if text and text[-1] not in ".!?":
        text += "."

    return text
    
# ============================================================
# TRANSLATION
# ============================================================

def translate(
    text,
    source_language,
    target_language,
):

    direction = f"{source_language}-{target_language}"

    system = get_model(direction)

    model = system["model"]
    src_tokenizer = system["src"]
    trg_tokenizer = system["trg"]

    src_tokens = encode_sentence(
        text,
        src_tokenizer
    )

    predicted_ids = greedy_decode(

    model=model,

    src_tokens=src_tokens,

    target_tokenizer=trg_tokenizer,

    max_length=100

)

    prediction = decode_tokens(
        predicted_ids,
        trg_tokenizer
    )

    prediction = clean_translation(prediction)

    return prediction