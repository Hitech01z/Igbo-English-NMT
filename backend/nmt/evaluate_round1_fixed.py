import os
import sys
import math
import random
import warnings

import pandas as pd
import torch
import sacrebleu

from nmt.transformer import TransformerNMT
from nmt.tokenizer import SimpleTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_PATH = "processed/baseline/test.csv"

IGBO_TOKENIZER_PATH = (
    "processed/round1/tokenizers/igbo_tokenizer.json"
)

ENGLISH_TOKENIZER_PATH = (
    "processed/round1/tokenizers/english_tokenizer.json"
)

CHECKPOINT_PATH = (
    "checkpoints/round1/round1_best.pt"
)

OUTPUT_PREDICTIONS = (
    "evaluation/round1_fixed_predictions.csv"
)

OUTPUT_SCORES = (
    "evaluation/round1_fixed_scores.csv"
)

MAX_SEQ_LENGTH = 100

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TOKENIZER HELPERS
# ============================================================

def get_vocab_size(tokenizer):

    if hasattr(tokenizer, "token_to_id"):
        return len(tokenizer.token_to_id)

    if hasattr(tokenizer, "vocab"):
        return len(tokenizer.vocab)

    if hasattr(tokenizer, "word2idx"):
        return len(tokenizer.word2idx)

    if hasattr(tokenizer, "stoi"):
        return len(tokenizer.stoi)

    raise AttributeError(
        "Could not determine tokenizer vocabulary size."
    )


def get_token_id(tokenizer, name, default=None):

    # Direct attribute
    if hasattr(tokenizer, name):

        value = getattr(tokenizer, name)

        if isinstance(value, int):
            return value

    # Common alternative naming
    alternatives = {

        "pad_token_id": [
            "pad_id",
            "PAD_ID",
            "padding_idx"
        ],

        "sos_token_id": [
            "sos_id",
            "SOS_ID",
            "start_token_id",
            "bos_token_id",
            "bos_id"
        ],

        "eos_token_id": [
            "eos_id",
            "EOS_ID",
            "end_token_id"
        ]

    }

    for alternative in alternatives.get(name, []):

        if hasattr(tokenizer, alternative):

            value = getattr(tokenizer, alternative)

            if isinstance(value, int):
                return value

    # Search vocabulary
    if hasattr(tokenizer, "token_to_id"):

        token_to_id = tokenizer.token_to_id

        candidates = {

            "pad_token_id": [
                "<pad>",
                "[PAD]",
                "<PAD>"
            ],

            "sos_token_id": [
                "<sos>",
                "<SOS>",
                "<bos>",
                "<BOS>",
                "<start>"
            ],

            "eos_token_id": [
                "<eos>",
                "<EOS>",
                "</s>",
                "<end>"
            ]

        }

        for token in candidates.get(name, []):

            if token in token_to_id:

                return token_to_id[token]

    if default is not None:
        return default

    raise AttributeError(
        f"Could not determine {name} from tokenizer."
    )


def encode_sentence(tokenizer, sentence, max_length):

    encoded = tokenizer.encode(sentence)

    if isinstance(encoded, torch.Tensor):

        encoded = encoded.tolist()

    encoded = list(encoded)

    pad_id = get_token_id(
        tokenizer,
        "pad_token_id",
        default=0
    )

    if len(encoded) > max_length:

        encoded = encoded[:max_length]

    else:

        encoded = encoded + [
            pad_id
        ] * (
            max_length - len(encoded)
        )

    return torch.tensor(
        encoded,
        dtype=torch.long
    )


def decode_tokens(tokenizer, token_ids):

    try:

        decoded = tokenizer.decode(
            token_ids
        )

        if isinstance(decoded, str):

            return decoded.strip()

    except Exception:

        pass

    if hasattr(tokenizer, "id_to_token"):

        words = []

        for token_id in token_ids:

            token_id = int(token_id)

            token = tokenizer.id_to_token.get(
                token_id,
                ""
            )

            if token:

                words.append(token)

        return " ".join(words).strip()

    if hasattr(tokenizer, "idx2word"):

        words = []

        for token_id in token_ids:

            token_id = int(token_id)

            if token_id in tokenizer.idx2word:

                words.append(
                    tokenizer.idx2word[token_id]
                )

        return " ".join(words).strip()

    return ""


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(
    igbo_tokenizer,
    english_tokenizer,
    checkpoint_path,
    device
):

    igbo_vocab_size = get_vocab_size(
        igbo_tokenizer
    )

    english_vocab_size = get_vocab_size(
        english_tokenizer
    )

    print(
        f"Igbo vocabulary: {igbo_vocab_size}"
    )

    print(
        f"English vocabulary: {english_vocab_size}"
    )

    # IMPORTANT:
    #
    # Round 1 checkpoint was trained with:
    #
    # d_model = 256
    # nhead = 8
    # encoder layers = 3
    # decoder layers = 3
    # feedforward = 512
    # dropout = 0.1
    # max_seq_length = 100
    #
    # max_seq_length MUST remain 100 because
    # the checkpoint positional encodings have shape:
    #
    # [1, 100, 256]

    model = TransformerNMT(

        src_vocab_size=igbo_vocab_size,

        trg_vocab_size=english_vocab_size,

        d_model=256,

        nhead=8,

        num_encoder_layers=3,

        num_decoder_layers=3,

        dim_feedforward=512,

        dropout=0.1,

        max_seq_length=100

    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(device)

    model.eval()

    return model


# ============================================================
# TRANSLATION
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

    src_tokens = encode_sentence(

        src_tokenizer,

        sentence,

        max_length

    )

    src_tokens = src_tokens.unsqueeze(0)

    src_tokens = src_tokens.to(device)

    pad_id = get_token_id(

        src_tokenizer,

        "pad_token_id",

        default=0

    )

    src_padding_mask = (
        src_tokens == pad_id
    )

    sos_id = get_token_id(

        trg_tokenizer,

        "sos_token_id"

    )

    eos_id = get_token_id(

        trg_tokenizer,

        "eos_token_id"

    )

    generated = [

        sos_id

    ]

    for _ in range(max_length - 1):

        trg_tokens = torch.tensor(

            generated,

            dtype=torch.long,

            device=device

        ).unsqueeze(0)

        trg_padding_mask = (

            trg_tokens == get_token_id(

                trg_tokenizer,

                "pad_token_id",

                default=0

            )

        )

        output = model(

            src=src_tokens,

            trg=trg_tokens,

            src_padding_mask=src_padding_mask,

            trg_padding_mask=trg_padding_mask

        )

        next_token_logits = output[

            0,

            -1,

            :

        ]

        next_token = torch.argmax(

            next_token_logits

        ).item()

        generated.append(

            next_token

        )

        if next_token == eos_id:

            break

    translation = decode_tokens(

        trg_tokenizer,

        generated

    )

    return translation


# ============================================================
# CLEAN GENERATED TRANSLATION
# ============================================================

def clean_translation(
    translation,
    tokenizer
):

    if not translation:

        return ""

    special_tokens = [

        "<pad>",

        "<sos>",

        "<eos>",

        "<bos>",

        "</s>",

        "<unk>",

        "[PAD]",

        "[SOS]",

        "[EOS]",

        "[UNK]"

    ]

    words = translation.split()

    cleaned = []

    for word in words:

        if word in special_tokens:

            continue

        cleaned.append(word)

    return " ".join(cleaned).strip()


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(
    references,
    predictions
):

    bleu = sacrebleu.corpus_bleu(

        predictions,

        [references]

    )

    chrf = sacrebleu.corpus_chrf(

        predictions,

        [references]

    )

    return bleu.score, chrf.score


# ============================================================
# MAIN
# ============================================================

def main():

    warnings.filterwarnings(

        "ignore",

        message="The PyTorch API of nested tensors"

    )

    warnings.filterwarnings(

        "ignore",

        message="Support for mismatched key_padding_mask"

    )

    set_seed(SEED)

    print("=" * 60)

    print(
        "FIXED ROUND 1 EVALUATION"
    )

    print(
        "SAME TEST SET AS BASELINE AND ROUND 2"
    )

    print("=" * 60)

    print(

        f"Device: {DEVICE}"

    )

    print()

    # --------------------------------------------------------
    # LOAD SHARED TEST SET
    # --------------------------------------------------------

    if not os.path.exists(TEST_PATH):

        raise FileNotFoundError(

            f"Test dataset not found:\n{TEST_PATH}"

        )

    test_df = pd.read_csv(

        TEST_PATH

    )

    required_columns = [

        "igbo",

        "english"

    ]

    for column in required_columns:

        if column not in test_df.columns:

            raise ValueError(

                f"Missing required column: {column}"

            )

    print(

        f"Test samples: {len(test_df)}"

    )

    print()

    # --------------------------------------------------------
    # LOAD TOKENIZERS
    # --------------------------------------------------------

    if not os.path.exists(

        IGBO_TOKENIZER_PATH

    ):

        raise FileNotFoundError(

            "Igbo tokenizer not found:\n"

            + IGBO_TOKENIZER_PATH

        )

    if not os.path.exists(

        ENGLISH_TOKENIZER_PATH

    ):

        raise FileNotFoundError(

            "English tokenizer not found:\n"

            + ENGLISH_TOKENIZER_PATH

        )

    igbo_tokenizer = (

        SimpleTokenizer.load(

            IGBO_TOKENIZER_PATH

        )

    )

    english_tokenizer = (

        SimpleTokenizer.load(

            ENGLISH_TOKENIZER_PATH

        )

    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model(

        igbo_tokenizer,

        english_tokenizer,

        CHECKPOINT_PATH,

        DEVICE

    )

    print()

    print(

        "Round 1 model loaded successfully."

    )

    print()

    # --------------------------------------------------------
    # TRANSLATE TEST SET
    # --------------------------------------------------------

    predictions = []

    references = []

    source_sentences = []

    print(

        "Translating test set..."

    )

    for index, row in test_df.iterrows():

        igbo_sentence = str(

            row["igbo"]

        ).strip()

        english_reference = str(

            row["english"]

        ).strip()

        prediction = translate_sentence(

            model=model,

            sentence=igbo_sentence,

            src_tokenizer=igbo_tokenizer,

            trg_tokenizer=english_tokenizer,

            device=DEVICE,

            max_length=MAX_SEQ_LENGTH

        )

        prediction = clean_translation(

            prediction,

            english_tokenizer

        )

        source_sentences.append(

            igbo_sentence

        )

        predictions.append(

            prediction

        )

        references.append(

            english_reference

        )

        completed = index + 1

        if completed % 10 == 0:

            print(

                f"Translated {completed}/{len(test_df)}"

            )

    if len(test_df) % 10 != 0:

        print(

            f"Translated {len(test_df)}/{len(test_df)}"

        )

    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    bleu_score, chrf_score = (

        evaluate_predictions(

            references,

            predictions

        )

    )

    print()

    print("=" * 60)

    print(

        f"BLEU: {bleu_score:.2f}"

    )

    print(

        f"chrF++: {chrf_score:.2f}"

    )

    print("=" * 60)

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    os.makedirs(

        "evaluation",

        exist_ok=True

    )

    predictions_df = pd.DataFrame({

        "igbo": source_sentences,

        "reference_english": references,

        "predicted_english": predictions

    })

    predictions_df.to_csv(

        OUTPUT_PREDICTIONS,

        index=False,

        encoding="utf-8"

    )

    scores_df = pd.DataFrame({

        "model": [

            "round1_fixed"

        ],

        "test_samples": [

            len(test_df)

        ],

        "bleu": [

            bleu_score

        ],

        "chrf_plus_plus": [

            chrf_score

        ]

    })

    scores_df.to_csv(

        OUTPUT_SCORES,

        index=False

    )

    print()

    print(

        "Saved:"

    )

    print(

        OUTPUT_PREDICTIONS

    )

    print(

        OUTPUT_SCORES

    )

    print()

    # --------------------------------------------------------
    # SHOW SAMPLE TRANSLATIONS
    # --------------------------------------------------------

    print("=" * 60)

    print(

        "SAMPLE TRANSLATIONS"

    )

    print("=" * 60)

    sample_count = min(

        10,

        len(predictions_df)

    )

    for index in range(sample_count):

        print()

        print(

            f"Igbo: {predictions_df.iloc[index]['igbo']}"

        )

        print(

            "Reference: "

            + str(

                predictions_df.iloc[index][

                    "reference_english"

                ]

            )

        )

        print(

            "Prediction: "

            + str(

                predictions_df.iloc[index][

                    "predicted_english"

                ]

            )

        )


if __name__ == "__main__":

    main()