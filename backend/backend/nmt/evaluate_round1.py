import math
from pathlib import Path

import pandas as pd
import torch
import sacrebleu

from nmt.tokenizer import SimpleTokenizer
from nmt.transformer import TransformerNMT


BASE_DIR = Path(__file__).resolve().parent.parent


TEST_FILE = (
    BASE_DIR
    / "processed"
    / "round1"
    / "splits"
    / "test.csv"
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
    / "evaluation"
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


def create_padding_mask(sequence, pad_idx=0):

    return sequence == pad_idx


def translate_sentence(
    model,
    sentence,
    src_tokenizer,
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

        memory = model.transformer.encoder(
            src_embedding,
            src_key_padding_mask=src_padding_mask
        )

    sos_idx = english_tokenizer.token_to_id["<sos>"]
    eos_idx = english_tokenizer.token_to_id["<eos>"]

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

            output = (
                model.transformer.decoder(
                    trg_embedding,
                    memory,
                    tgt_mask=trg_mask,
                    memory_key_padding_mask=src_padding_mask
                )
            )

            output = (
                model.output_layer(output)
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

    return english_tokenizer.decode(
        generated_ids
    )


def main():

    print("=" * 60)

    print(
        "IGBO-ENGLISH TRANSFORMER "
        "ITERATIVE BACK-TRANSLATION ROUND 1 EVALUATION"
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

    print()

    # -------------------------
    # Load test dataset
    # -------------------------

    test_df = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Test samples: "
        f"{len(test_df)}"
    )

    print()

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
        "Round 1 model loaded successfully."
    )

    print()

    # -------------------------
    # Translate test set
    # -------------------------

    predictions = []

    references = []

    source_sentences = []

    print(
        "Translating test set..."
    )

    for index, row in test_df.iterrows():

        source = row["igbo"]

        reference = row["english"]

        prediction = translate_sentence(

            model,

            source,

            igbo_tokenizer,

            english_tokenizer,

            DEVICE
        )

        source_sentences.append(
            source
        )

        references.append(
            reference
        )

        predictions.append(
            prediction
        )

        if (
            (index + 1) % 10 == 0
            or index + 1 == len(test_df)
        ):

            print(
                f"Translated "
                f"{index + 1}/"
                f"{len(test_df)}"
            )

    # -------------------------
    # Calculate BLEU
    # -------------------------

    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references]
    )

    # -------------------------
    # Calculate chrF++
    # -------------------------

    chrf = sacrebleu.corpus_chrf(
        predictions,
        [references],
        word_order=2
    )

    print()

    print("=" * 60)

    print(
        f"BLEU: "
        f"{bleu.score:.2f}"
    )

    print(
        f"chrF++: "
        f"{chrf.score:.2f}"
    )

    print("=" * 60)

    # -------------------------
    # Save predictions
    # -------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions_df = pd.DataFrame({

        "igbo":
        source_sentences,

        "reference":
        references,

        "prediction":
        predictions

    })

    predictions_file = (
        OUTPUT_DIR
        / "round1_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_file,
        index=False,
        encoding="utf-8-sig"
    )

    # -------------------------
    # Save scores
    # -------------------------

    scores_df = pd.DataFrame({

        "model": [
            "Round 1 Iterative Back-Translation"
        ],

        "test_samples": [
            len(test_df)
        ],

        "bleu": [
            bleu.score
        ],

        "chrf++": [
            chrf.score
        ]

    })

    scores_file = (
        OUTPUT_DIR
        / "round1_scores.csv"
    )

    scores_df.to_csv(
        scores_file,
        index=False
    )

    print()

    print(
        "Saved:"
    )

    print(
        predictions_file
    )

    print(
        scores_file
    )


if __name__ == "__main__":

    main()