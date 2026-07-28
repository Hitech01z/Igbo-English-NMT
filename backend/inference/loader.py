from pathlib import Path

import torch

from nmt.transformer import TransformerNMT
from nmt.tokenizer import SimpleTokenizer


BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT = (
    BASE_DIR
    / "checkpoints"
    / "round2"
    / "round2_best.pt"
)

IGBO_TOKENIZER = (
    BASE_DIR
    / "processed"
    / "round2"
    / "tokenizers"
    / "igbo_tokenizer.json"
)

ENGLISH_TOKENIZER = (
    BASE_DIR
    / "processed"
    / "round2"
    / "tokenizers"
    / "english_tokenizer.json"
)


device = torch.device("cpu")


def get_vocab_size(tokenizer):

    for attr in [
        "vocab_size",
        "size",
        "num_tokens"
    ]:

        if hasattr(tokenizer, attr):
            value = getattr(tokenizer, attr)

            if callable(value):
                return value()

            return value

    if hasattr(tokenizer, "token_to_id"):

        return len(tokenizer.token_to_id)

    if hasattr(tokenizer, "stoi"):

        return len(tokenizer.stoi)

    raise RuntimeError(
        "Cannot determine tokenizer vocabulary size."
    )


def load_translation_system():

    igbo_tokenizer = SimpleTokenizer.load(
        str(IGBO_TOKENIZER)
    )

    english_tokenizer = SimpleTokenizer.load(
        str(ENGLISH_TOKENIZER)
    )

    model = TransformerNMT(
        src_vocab_size=get_vocab_size(
            igbo_tokenizer
        ),
        trg_vocab_size=get_vocab_size(
            english_tokenizer
        ),
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_length=100
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return (
        model,
        igbo_tokenizer,
        english_tokenizer
    )