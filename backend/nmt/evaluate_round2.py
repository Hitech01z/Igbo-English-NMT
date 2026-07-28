import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from nmt.tokenizer import SimpleTokenizer
from nmt.transformer import TransformerNMT


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = (
    BASE_DIR
    / "processed"
    / "round2"
    / "splits"
    / "test.csv"
)

IGBO_TOKENIZER_FILE = (
    BASE_DIR
    / "processed"
    / "round2"
    / "tokenizers"
    / "igbo_tokenizer.json"
)

ENGLISH_TOKENIZER_FILE = (
    BASE_DIR
    / "processed"
    / "round2"
    / "tokenizers"
    / "english_tokenizer.json"
)

CHECKPOINT_FILE = (
    BASE_DIR
    / "checkpoints"
    / "round2"
    / "round2_best.pt"
)

OUTPUT_DIR = (
    BASE_DIR
    / "evaluation"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MAX_SEQ_LENGTH = 100

D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1


# ============================================================
# TOKENIZER HELPERS
# ============================================================

def get_token_id(
    tokenizer,
    token_name,
    fallback
):
    """
    Safely obtain special-token IDs from SimpleTokenizer.

    This avoids assuming attributes such as:

        sos_token_id
        eos_token_id
        pad_token_id

    because the current tokenizer may expose them differently.
    """

    possible_attributes = {

        "pad": [
            "pad_token_id",
            "pad_id"
        ],

        "sos": [
            "sos_token_id",
            "sos_id",
            "start_token_id",
            "start_id"
        ],

        "eos": [
            "eos_token_id",
            "eos_id",
            "end_token_id",
            "end_id"
        ]
    }

    for attribute in possible_attributes[token_name]:

        if hasattr(tokenizer, attribute):

            value = getattr(
                tokenizer,
                attribute
            )

            if value is not None:
                return int(value)

    # Try vocabulary dictionary
    possible_vocab_attributes = [
        "stoi",
        "token_to_id",
        "vocab"
    ]

    for attribute in possible_vocab_attributes:

        if hasattr(tokenizer, attribute):

            vocab = getattr(
                tokenizer,
                attribute
            )

            if isinstance(vocab, dict):

                special_tokens = {

                    "pad": [
                        "<pad>",
                        "[PAD]"
                    ],

                    "sos": [
                        "<sos>",
                        "<bos>",
                        "[SOS]",
                        "[BOS]"
                    ],

                    "eos": [
                        "<eos>",
                        "</s>",
                        "[EOS]"
                    ]
                }

                for token in special_tokens[token_name]:

                    if token in vocab:

                        return int(
                            vocab[token]
                        )

    return fallback


def get_special_token_ids(
    tokenizer
):

    pad_id = get_token_id(
        tokenizer,
        "pad",
        0
    )

    sos_id = get_token_id(
        tokenizer,
        "sos",
        1
    )

    eos_id = get_token_id(
        tokenizer,
        "eos",
        2
    )

    return (
        pad_id,
        sos_id,
        eos_id
    )


# ============================================================
# TOKENIZATION
# ============================================================

def encode_sentence(
    tokenizer,
    sentence
):

    """
    Encode a sentence using the current SimpleTokenizer API.
    """

    encoded = tokenizer.encode(
        sentence
    )

    if isinstance(
        encoded,
        torch.Tensor
    ):

        encoded = encoded.tolist()

    return encoded


# ============================================================
# AUTOREGRESSIVE TRANSLATION
# ============================================================

@torch.no_grad()
def translate_sentence(
    model,
    sentence,
    src_tokenizer,
    trg_tokenizer,
    device,
    max_length=100
):

    """
    Translate one Igbo sentence into English.

    This function deliberately uses ONLY the model's actual
    forward() method:

        model(
            src,
            trg,
            src_padding_mask,
            trg_padding_mask
        )

    It does not use:

        model.encode()
        model.decode()
        model.positional_encoding

    because those methods do not exist in the current
    TransformerNMT implementation.
    """

    (
        src_pad_id,
        src_sos_id,
        src_eos_id
    ) = get_special_token_ids(
        src_tokenizer
    )

    (
        trg_pad_id,
        trg_sos_id,
        trg_eos_id
    ) = get_special_token_ids(
        trg_tokenizer
    )

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    src_tokens = encode_sentence(
        src_tokenizer,
        sentence
    )

    # Ensure source sequence has SOS and EOS
    if len(src_tokens) == 0:

        src_tokens = [
            src_sos_id,
            src_eos_id
        ]

    else:

        if src_tokens[0] != src_sos_id:

            src_tokens = [
                src_sos_id
            ] + src_tokens

        if src_tokens[-1] != src_eos_id:

            src_tokens = src_tokens + [
                src_eos_id
            ]

    # Limit source length
    src_tokens = src_tokens[
        :max_length
    ]

    if src_tokens[-1] != src_eos_id:

        src_tokens[-1] = src_eos_id

    src = torch.tensor(
        [src_tokens],
        dtype=torch.long,
        device=device
    )

    src_padding_mask = (
        src == src_pad_id
    )

    # --------------------------------------------------------
    # START TARGET WITH SOS
    # --------------------------------------------------------

    generated = [
        trg_sos_id
    ]

    # --------------------------------------------------------
    # AUTOREGRESSIVE DECODING
    # --------------------------------------------------------

    for _ in range(
        max_length - 1
    ):

        trg = torch.tensor(
            [generated],
            dtype=torch.long,
            device=device
        )

        trg_padding_mask = (
            trg == trg_pad_id
        )

        # ====================================================
        # USE THE ACTUAL MODEL FORWARD METHOD
        # ====================================================

        output = model(
            src=src,
            trg=trg,
            src_padding_mask=src_padding_mask,
            trg_padding_mask=trg_padding_mask
        )

        # ----------------------------------------------------
        # LAST TOKEN PREDICTION
        # ----------------------------------------------------

        next_token_logits = output[
            0,
            -1,
            :
        ]

        next_token = (
            next_token_logits
            .argmax()
            .item()
        )

        generated.append(
            next_token
        )

        # Stop at EOS
        if next_token == trg_eos_id:

            break

    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    # Remove SOS
    generated = generated[1:]

    if trg_eos_id in generated:

        generated = generated[
            :generated.index(
                trg_eos_id
            )
        ]

    # Use tokenizer decode
    try:

        translation = (
            trg_tokenizer.decode(
                generated
            )
        )

    except TypeError:

        translation = (
            trg_tokenizer.decode(
                generated,
                skip_special_tokens=True
            )
        )

    return translation


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def compute_bleu(
    references,
    predictions
):

    try:

        from nltk.translate.bleu_score import (
            corpus_bleu,
            SmoothingFunction
        )

        formatted_references = [
            [
                reference.split()
            ]
            for reference in references
        ]

        formatted_predictions = [
            prediction.split()
            for prediction in predictions
        ]

        smoothing = (
            SmoothingFunction()
            .method1
        )

        score = corpus_bleu(
            formatted_references,
            formatted_predictions,
            smoothing_function=smoothing
        )

        return score * 100

    except Exception as error:

        print(
            f"BLEU calculation warning: {error}"
        )

        return 0.0


def compute_chrf(
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

        result = metric.corpus_score(
            predictions,
            [
                references
            ]
        )

        return result.score

    except Exception as error:

        print(
            f"chrF++ calculation warning: {error}"
        )

        return 0.0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "IGBO-ENGLISH TRANSFORMER "
        "ITERATIVE BACK-TRANSLATION "
        "ROUND 2 EVALUATION"
    )

    print("=" * 60)

    print()

    print(
        f"Device: {DEVICE}"
    )

    print()

    # --------------------------------------------------------
    # LOAD TOKENIZERS
    # --------------------------------------------------------

    src_tokenizer = (
        SimpleTokenizer.load(
            IGBO_TOKENIZER_FILE
        )
    )

    trg_tokenizer = (
        SimpleTokenizer.load(
            ENGLISH_TOKENIZER_FILE
        )
    )

    print(
        f"Igbo vocabulary: "
        f"{len(src_tokenizer)}"
    )

    print(
        f"English vocabulary: "
        f"{len(trg_tokenizer)}"
    )

    print()

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    test_df = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Test samples: "
        f"{len(test_df)}"
    )

    print()

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    model = TransformerNMT(

        src_vocab_size=len(
            src_tokenizer
        ),

        trg_vocab_size=len(
            trg_tokenizer
        ),

        d_model=D_MODEL,

        nhead=NHEAD,

        num_encoder_layers=(
            NUM_ENCODER_LAYERS
        ),

        num_decoder_layers=(
            NUM_DECODER_LAYERS
        ),

        dim_feedforward=(
            DIM_FEEDFORWARD
        ),

        dropout=DROPOUT,

        max_seq_length=(
            MAX_SEQ_LENGTH
        )
    )

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            model.load_state_dict(
                checkpoint
            )

    else:

        model.load_state_dict(
            checkpoint
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "Round 2 model loaded successfully."
    )

    print()

    # --------------------------------------------------------
    # TRANSLATION
    # --------------------------------------------------------

    references = []

    predictions = []

    source_sentences = []

    print(
        "Translating test set..."
    )

    start_time = time.time()

    for index, row in test_df.iterrows():

        igbo_sentence = str(
            row["igbo"]
        )

        english_reference = str(
            row["english"]
        )

        prediction = translate_sentence(

            model=model,

            sentence=igbo_sentence,

            src_tokenizer=src_tokenizer,

            trg_tokenizer=trg_tokenizer,

            device=DEVICE,

            max_length=MAX_SEQ_LENGTH
        )

        source_sentences.append(
            igbo_sentence
        )

        references.append(
            english_reference
        )

        predictions.append(
            prediction
        )

        if (
            (index + 1) % 10 == 0
            or
            (index + 1) == len(test_df)
        ):

            print(
                f"Translated "
                f"{index + 1}/"
                f"{len(test_df)}"
            )

    elapsed = (
        time.time()
        -
        start_time
    )

    print()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    bleu_score = compute_bleu(
        references,
        predictions
    )

    chrf_score = compute_chrf(
        references,
        predictions
    )

    print("=" * 60)

    print(
        f"BLEU: {bleu_score:.2f}"
    )

    print(
        f"chrF++: {chrf_score:.2f}"
    )

    print("=" * 60)

    print()

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions_df = pd.DataFrame({

        "igbo": source_sentences,

        "reference_english": references,

        "predicted_english": predictions

    })

    predictions_file = (
        OUTPUT_DIR
        / "round2_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_file,
        index=False,
        encoding="utf-8"
    )

    scores_df = pd.DataFrame({

        "metric": [
            "BLEU",
            "chrF++"
        ],

        "score": [
            bleu_score,
            chrf_score
        ]

    })

    scores_file = (
        OUTPUT_DIR
        / "round2_scores.csv"
    )

    scores_df.to_csv(
        scores_file,
        index=False,
        encoding="utf-8"
    )

    print(
        "Saved:"
    )

    print(
        predictions_file
    )

    print(
        scores_file
    )

    print()

    print(
        f"Evaluation time: "
        f"{elapsed:.2f} seconds"
    )


if __name__ == "__main__":

    main()