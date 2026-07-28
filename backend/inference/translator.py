import torch

from inference.loader import load_translation_system
from inference.decoder import (
    greedy_decode,
    decode_tokens
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model, igbo_tokenizer, english_tokenizer = (
    load_translation_system()
)

model.eval()


def encode_sentence(sentence):

    sentence = sentence.lower().strip()

    if hasattr(
        igbo_tokenizer,
        "encode"
    ):

        token_ids = igbo_tokenizer.encode(
            sentence
        )

    elif hasattr(
        igbo_tokenizer,
        "encode_text"
    ):

        token_ids = igbo_tokenizer.encode_text(
            sentence
        )

    else:

        raise AttributeError(
            "Igbo tokenizer has no encode() function."
        )

    return torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )


def translate_sentence(sentence):

    src_tokens = encode_sentence(
        sentence
    )

    predicted_ids = greedy_decode(
    model=model,
    src_tokens=src_tokens,
    target_tokenizer=trg_tokenizer,
    max_length=100
)

    prediction = decode_tokens(
        predicted_ids,
        english_tokenizer
    )

    return prediction