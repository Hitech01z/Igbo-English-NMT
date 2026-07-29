from pathlib import Path

import torch

from nmt.tokenizer import SimpleTokenizer
from nmt.transformer import TransformerNMT


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


BASE_DIR = Path(__file__).resolve().parents[1]


CHECKPOINT = (
    BASE_DIR
    / "checkpoints"
    / "english_to_igbo"
    / "english_to_igbo_best.pt"
)


TOKENIZER_DIR = (
    BASE_DIR
    / "processed"
    / "english_to_igbo"
)


def load_reverse_translation_system():

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    english_tokenizer = SimpleTokenizer.load(
        TOKENIZER_DIR
        / "english_tokenizer.json"
    )

    igbo_tokenizer = SimpleTokenizer.load(
        TOKENIZER_DIR
        / "igbo_tokenizer.json"
    )

    model = TransformerNMT(

        src_vocab_size=checkpoint["src_vocab_size"],

        trg_vocab_size=checkpoint["trg_vocab_size"],

        d_model=checkpoint["d_model"],

        nhead=checkpoint["nhead"],

        num_encoder_layers=checkpoint["num_encoder_layers"],

        num_decoder_layers=checkpoint["num_decoder_layers"],

        dim_feedforward=checkpoint["dim_feedforward"],

        dropout=checkpoint["dropout"]

    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    return (

        model,

        english_tokenizer,

        igbo_tokenizer

    )