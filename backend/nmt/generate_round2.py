import math
from pathlib import Path

import pandas as pd
import torch

from nmt.tokenizer import SimpleTokenizer
from nmt.transformer import TransformerNMT


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
    / "round1"
    / "tokenizers"
)


CHECKPOINT_FILE = (
    BASE_DIR
    / "checkpoints"
    / "round1"
    / "round1_best.pt"
)


OUTPUT_DIR = (
    BASE_DIR
    / "processed"
    / "back_translation"
)


OUTPUT_FILE = (
    OUTPUT_DIR
    / "round2_raw.csv"
)


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


def create_padding_mask(
    sequence,
    pad_idx=0
):

    return sequence == pad_idx


def translate_sentence(
    model,
    sentence,
    igbo_tokenizer,
    english_tokenizer,
    device
):

    model.eval()

    src_ids = (
        igbo_tokenizer.encode(
            sentence
        )
    )

    src = torch.tensor(
        src_ids,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():

        src_embedding = (
            model.src_embedding(src)
            * math.sqrt(model.d_model)
        )

        src_embedding = (
            model.src_positional_encoding(
                src_embedding
            )
        )

        src_padding_mask = (
            create_padding_mask(src)
        )

        memory = (
            model.transformer.encoder(
                src_embedding,
                src_key_padding_mask=
                src_padding_mask
            )
        )

    sos_idx = (
        english_tokenizer.token_to_id[
            "<sos>"
        ]
    )

    eos_idx = (
        english_tokenizer.token_to_id[
            "<eos>"
        ]
    )

    generated_ids = [
        sos_idx
    ]

    for _ in range(MAX_LENGTH):

        trg = torch.tensor(
            generated_ids,
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        with torch.no_grad():

            trg_embedding = (
                model.trg_embedding(trg)
                * math.sqrt(model.d_model)
            )

            trg_embedding = (
                model.trg_positional_encoding(
                    trg_embedding
                )
            )

            trg_mask = (
                model.generate_square_subsequent_mask(
                    trg.size(1),
                    device
                )
            )

            decoder_output = (
                model.transformer.decoder(
                    trg_embedding,
                    memory,
                    tgt_mask=trg_mask,
                    memory_key_padding_mask=
                    src_padding_mask
                )
            )

            output = (
                model.output_layer(
                    decoder_output
                )
            )

            next_token = (
                output[:, -1, :]
                .argmax(dim=-1)
                .item()
            )

        generated_ids.append(
            next_token
        )

        if next_token == eos_idx:

            break

    return (
        english_tokenizer.decode(
            generated_ids
        )
    )


def main():

    print("=" * 60)

    print(
        "GENERATING ROUND 2 "
        "BACK-TRANSLATION DATA"
    )

    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print()

    # -------------------------
    # Load tokenizers
    # -------------------------

    igbo_tokenizer = (
        SimpleTokenizer.load(
            TOKENIZER_DIR
            / "igbo_tokenizer.json"
        )
    )

    english_tokenizer = (
        SimpleTokenizer.load(
            TOKENIZER_DIR
            / "english_tokenizer.json"
        )
    )

    print(
        f"Igbo vocabulary: "
        f"{len(igbo_tokenizer)}"
    )

    print(
        f"English vocabulary: "
        f"{len(english_tokenizer)}"
    )

    # -------------------------
    # Build model
    # -------------------------

    model = TransformerNMT(

        src_vocab_size=
        len(igbo_tokenizer),

        trg_vocab_size=
        len(english_tokenizer),

        d_model=D_MODEL,

        nhead=NHEAD,

        num_encoder_layers=
        NUM_ENCODER_LAYERS,

        num_decoder_layers=
        NUM_DECODER_LAYERS,

        dim_feedforward=
        DIM_FEEDFORWARD,

        dropout=DROPOUT
    )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "✓ Round 1 model loaded"
    )

    # -------------------------
    # Load monolingual data
    # -------------------------

    mono_df = pd.read_csv(
        MONOLINGUAL_FILE
    )

    print()

    print(
        f"Monolingual Igbo sentences: "
        f"{len(mono_df)}"
    )

    # -------------------------
    # Generate translations
    # -------------------------

    results = []

    print()

    print(
        "Generating Round 2 "
        "synthetic translations..."
    )

    for index, row in mono_df.iterrows():

        igbo_sentence = (
            row["igbo"]
        )

        english_translation = (
            translate_sentence(
                model,
                igbo_sentence,
                igbo_tokenizer,
                english_tokenizer,
                DEVICE
            )
        )

        results.append({

            "id":
            row.get(
                "id",
                f"ROUND2_{index + 1:04d}"
            ),

            "igbo":
            igbo_sentence,

            "english":
            english_translation,

            "domain":
            row.get(
                "domain",
                "general"
            ),

            "source":
            "iterative_back_translation_round_2"

        })

        if (
            (index + 1) % 10 == 0
            or index + 1 == len(mono_df)
        ):

            print(
                f"Translated "
                f"{index + 1}/"
                f"{len(mono_df)}"
            )

    # -------------------------
    # Save results
    # -------------------------

    output_df = pd.DataFrame(
        results
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()

    print("=" * 60)

    print(
        "ROUND 2 BACK-TRANSLATION COMPLETE"
    )

    print("=" * 60)

    print(
        f"Generated pairs: "
        f"{len(output_df)}"
    )

    print()

    print(
        f"Saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        output_df.head()
    )


if __name__ == "__main__":

    main()