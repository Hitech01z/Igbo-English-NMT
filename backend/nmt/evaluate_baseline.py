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

TEST_FILE = (
    BASE_DIR
    / "processed"
    / "baseline"
    / "test.csv"
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

EVALUATION_DIR = (
    BASE_DIR
    / "evaluation"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

D_MODEL = 256

NHEAD = 8

NUM_ENCODER_LAYERS = 3

NUM_DECODER_LAYERS = 3

DIM_FEEDFORWARD = 512

DROPOUT = 0.1

MAX_SEQ_LENGTH = 100

MAX_GENERATION_LENGTH = 50


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    igbo_tokenizer,
    english_tokenizer
):

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

        dropout=DROPOUT,

        max_seq_length=MAX_SEQ_LENGTH

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


    return model


# ============================================================
# TRANSLATION
# ============================================================

def translate_sentence(

    sentence,

    model,

    igbo_tokenizer,

    english_tokenizer,

    device

):

    # --------------------------------------------------------
    # Encode source sentence
    # --------------------------------------------------------

    src_ids = igbo_tokenizer.encode(
        sentence
    )


    src = torch.tensor(

        src_ids,

        dtype=torch.long

    ).unsqueeze(

        0

    ).to(

        device

    )


    # --------------------------------------------------------
    # Create source embedding
    # --------------------------------------------------------

    with torch.no_grad():

        src_embedded = (

            model.src_embedding(
                src
            )

            *

            math.sqrt(
                model.d_model
            )

        )


        src_embedded = (

            model.src_positional_encoding(

                src_embedded

            )

        )


        # ----------------------------------------------------
        # Encode source
        # ----------------------------------------------------

        src_padding_mask = (

            src
            ==
            igbo_tokenizer.token_to_id[
                "<pad>"
            ]

        )


        memory = (

            model.transformer.encoder(

                src_embedded,

                src_key_padding_mask=
                src_padding_mask

            )

        )


    # --------------------------------------------------------
    # Start target sequence
    # --------------------------------------------------------

    sos_id = (

        english_tokenizer.token_to_id[
            "<sos>"
        ]

    )


    eos_id = (

        english_tokenizer.token_to_id[
            "<eos>"
        ]

    )


    trg_ids = [

        sos_id

    ]


    # --------------------------------------------------------
    # Autoregressive decoding
    # --------------------------------------------------------

    for _ in range(

        MAX_GENERATION_LENGTH

    ):

        trg = torch.tensor(

            trg_ids,

            dtype=torch.long

        ).unsqueeze(

            0

        ).to(

            device

        )


        trg_embedded = (

            model.trg_embedding(

                trg

            )

            *

            math.sqrt(

                model.d_model

            )

        )


        trg_embedded = (

            model.trg_positional_encoding(

                trg_embedded

            )

        )


        trg_mask = (

            model.generate_square_subsequent_mask(

                trg.size(1),

                device

            )

        )


        trg_padding_mask = (

            trg
            ==
            english_tokenizer.token_to_id[
                "<pad>"
            ]

        )


        decoder_output = (

            model.transformer.decoder(

                trg_embedded,

                memory,

                tgt_mask=trg_mask,

                tgt_key_padding_mask=
                trg_padding_mask,

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

            output[

                0,

                -1

            ]

            .argmax(

                dim=-1

            )

            .item()

        )


        trg_ids.append(

            next_token

        )


        if next_token == eos_id:

            break


    # --------------------------------------------------------
    # Decode output
    # --------------------------------------------------------

    translation = (

        english_tokenizer.decode(

            trg_ids,

            remove_special_tokens=True

        )

    )


    return translation


# ============================================================
# BLEU
# ============================================================

def calculate_bleu(

    references,

    predictions

):

    try:

        from nltk.translate.bleu_score import (

            corpus_bleu

        )


        tokenized_references = [

            [

                reference.lower().split()

            ]

            for reference in references

        ]


        tokenized_predictions = [

            prediction.lower().split()

            for prediction in predictions

        ]


        score = corpus_bleu(

            tokenized_references,

            tokenized_predictions,

            weights=(

                0.25,

                0.25,

                0.25,

                0.25

            )

        )


        return score * 100


    except Exception as error:

        print(

            f"BLEU calculation error: {error}"

        )

        return 0.0


# ============================================================
# chrF++
# ============================================================

def calculate_chrf(

    references,

    predictions

):

    try:

        from sacrebleu.metrics import (

            CHRF

        )


        metric = CHRF(

            word_order=2

        )


        score = metric.corpus_score(

            predictions,

            [

                references

            ]

        )


        return score.score


    except Exception as error:

        print(

            f"chrF++ calculation error: {error}"

        )

        return 0.0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(

        "IGBO-ENGLISH TRANSFORMER BASELINE EVALUATION"

    )

    print("=" * 60)


    print(

        f"Device: {DEVICE}"

    )


    # --------------------------------------------------------
    # Load tokenizers
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(

        igbo_tokenizer,

        english_tokenizer

    )


    # --------------------------------------------------------
    # Load fixed test set
    # --------------------------------------------------------

    test_df = pd.read_csv(

        TEST_FILE

    )


    print(

        f"Test samples: {len(test_df)}"

    )


    print()

    print(

        "Translating test set..."

    )


    predictions = []


    total = len(

        test_df

    )


    for index, row in test_df.iterrows():

        prediction = (

            translate_sentence(

                sentence=row["igbo"],

                model=model,

                igbo_tokenizer=
                igbo_tokenizer,

                english_tokenizer=
                english_tokenizer,

                device=DEVICE

            )

        )


        predictions.append(

            prediction

        )


        if (

            (index + 1) % 10 == 0

        ):

            print(

                f"Translated "
                f"{index + 1}/{total}"

            )


    # --------------------------------------------------------
    # References
    # --------------------------------------------------------

    references = (

        test_df["english"]

        .astype(str)

        .tolist()

    )


    # --------------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------------

    bleu = calculate_bleu(

        references,

        predictions

    )


    chrf = calculate_chrf(

        references,

        predictions

    )


    print()

    print("=" * 60)

    print(

        f"BLEU: {bleu:.2f}"

    )

    print(

        f"chrF++: {chrf:.2f}"

    )

    print("=" * 60)


    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    EVALUATION_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    predictions_df = pd.DataFrame({

        "igbo":

        test_df["igbo"],

        "reference_english":

        references,

        "predicted_english":

        predictions,

        "domain":

        test_df["domain"]

    })


    predictions_df.to_csv(

        EVALUATION_DIR
        / "baseline_predictions.csv",

        index=False,

        encoding="utf-8-sig"

    )


    scores_df = pd.DataFrame({

        "metric": [

            "BLEU",

            "chrF++"

        ],

        "score": [

            bleu,

            chrf

        ]

    })


    scores_df.to_csv(

        EVALUATION_DIR
        / "baseline_scores.csv",

        index=False,

        encoding="utf-8-sig"

    )


    print()

    print(

        "Saved:"

    )


    print(

        "evaluation/baseline_predictions.csv"

    )


    print(

        "evaluation/baseline_scores.csv"

    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()