import math
from pathlib import Path

import pandas as pd
import torch

from nmt.tokenizer import SimpleTokenizer
from nmt.transformer import TransformerNMT


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MONOLINGUAL_FILE = (
    BASE_DIR
    / "dataset"
    / "monolingual"
    / "igbo.csv"
)

TOKENIZER_DIR = (
    BASE_DIR
    / "processed"
    / "baseline"
    / "tokenizers"
)

CHECKPOINT_FILE = (
    BASE_DIR
    / "checkpoints"
    / "baseline"
    / "baseline_best.pt"
)

OUTPUT_DIR = (
    BASE_DIR
    / "processed"
    / "back_translation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "round1_raw.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MAX_LENGTH = 50

D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1


# ============================================================
# MASK
# ============================================================

def create_padding_mask(
    sequence,
    pad_idx=0
):

    return sequence == pad_idx


# ============================================================
# TRANSLATION
# ============================================================

def translate_sentence(
    model,
    sentence,
    igbo_tokenizer,
    english_tokenizer,
    device
):

    model.eval()

    src_ids = igbo_tokenizer.encode(
        sentence
    )

    src = torch.tensor(
        src_ids,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    src_padding_mask = create_padding_mask(
        src
    )

    generated_ids = [
        english_tokenizer.token_to_id["<sos>"]
    ]

    with torch.no_grad():

        for _ in range(MAX_LENGTH):

            trg = torch.tensor(
                [generated_ids],
                dtype=torch.long,
                device=device
            )

            trg_padding_mask = create_padding_mask(
                trg
            )

            output = model(
                src,
                trg,
                src_padding_mask=src_padding_mask,
                trg_padding_mask=trg_padding_mask
            )

            next_token_logits = output[
                0,
                -1,
                :
            ]

            next_token_id = torch.argmax(
                next_token_logits
            ).item()

            generated_ids.append(
                next_token_id
            )

            if next_token_id == (
                english_tokenizer.token_to_id["<eos>"]
            ):

                break

    translation = english_tokenizer.decode(
        generated_ids
    )

    return translation


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "GENERATING ROUND 1 BACK-TRANSLATION DATA"
    )

    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load tokenizers
    # --------------------------------------------------------

    igbo_tokenizer = SimpleTokenizer.load(
        TOKENIZER_DIR
        / "igbo_tokenizer.json"
    )

    english_tokenizer = SimpleTokenizer.load(
        TOKENIZER_DIR
        / "english_tokenizer.json"
    )

    print(
        f"Igbo vocabulary: {len(igbo_tokenizer)}"
    )

    print(
        f"English vocabulary: {len(english_tokenizer)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = TransformerNMT(

        src_vocab_size=len(
            igbo_tokenizer
        ),

        trg_vocab_size=len(
            english_tokenizer
        ),

        d_model=D_MODEL,

        nhead=NHEAD,

        num_encoder_layers=NUM_ENCODER_LAYERS,

        num_decoder_layers=NUM_DECODER_LAYERS,

        dim_feedforward=DIM_FEEDFORWARD,

        dropout=DROPOUT
    )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "✓ Baseline model loaded"
    )

    # --------------------------------------------------------
    # Load monolingual Igbo data
    # --------------------------------------------------------

    df = pd.read_csv(
        MONOLINGUAL_FILE
    )

    print(
        f"Monolingual Igbo sentences: {len(df)}"
    )

    # --------------------------------------------------------
    # Generate synthetic English translations
    # --------------------------------------------------------

    generated_rows = []

    print()

    print(
        "Generating synthetic translations..."
    )

    for index, row in df.iterrows():

        igbo_sentence = str(
            row["igbo"]
        ).strip()

        english_translation = (
            translate_sentence(
                model=model,

                sentence=igbo_sentence,

                igbo_tokenizer=igbo_tokenizer,

                english_tokenizer=english_tokenizer,

                device=DEVICE
            )
        )

        generated_rows.append({

            "id": row["id"],

            "igbo": igbo_sentence,

            "english": english_translation,

            "domain": row["domain"],

            "source": (
                "baseline_model_round_1"
            )

        })

        if (
            (index + 1) % 10 == 0
            or index == len(df) - 1
        ):

            print(
                f"Translated "
                f"{index + 1}/{len(df)}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_df = pd.DataFrame(
        generated_rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print()

    print("=" * 60)

    print(
        "ROUND 1 BACK-TRANSLATION COMPLETE"
    )

    print("=" * 60)

    print(
        f"Generated pairs: {len(output_df)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print()

    print(
        output_df.head()
    )


if __name__ == "__main__":

    main()