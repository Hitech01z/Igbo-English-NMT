import sys
from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(BASE_DIR)
)


from nmt.tokenizer import SimpleTokenizer


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    BASE_DIR
    / "processed"
    / "back_translation"
    / "round1_raw.csv"
)

TOKENIZER_FILE = (
    BASE_DIR
    / "processed"
    / "baseline"
    / "tokenizers"
    / "english_tokenizer.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "processed"
    / "back_translation"
)

FILTERED_FILE = (
    OUTPUT_DIR
    / "round1_quality_filtered.csv"
)

REJECTED_FILE = (
    OUTPUT_DIR
    / "round1_quality_rejected.csv"
)


# ============================================================
# FILTER CONFIGURATION
# ============================================================

MIN_SOURCE_WORDS = 3

MIN_TARGET_WORDS = 2

MAX_TARGET_WORDS = 40

MIN_LENGTH_RATIO = 0.25

MAX_LENGTH_RATIO = 3.0

MAX_REPETITION_RATIO = 0.50

MAX_UNKNOWN_RATIO = 0.30


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text):

    return str(text).strip().lower()


def get_words(text):

    return normalize_text(text).split()


def repetition_ratio(words):

    if not words:

        return 1.0

    counts = Counter(words)

    repeated_tokens = sum(

        count - 1

        for count in counts.values()

        if count > 1

    )

    return repeated_tokens / len(words)


def has_repeated_phrase(words):

    if len(words) < 4:

        return False

    for phrase_length in [2, 3]:

        for i in range(

            len(words)
            - phrase_length * 2
            + 1

        ):

            phrase_1 = words[
                i:
                i + phrase_length
            ]

            phrase_2 = words[
                i + phrase_length:
                i + phrase_length * 2
            ]

            if phrase_1 == phrase_2:

                return True

    return False


def contains_invalid_output(text):

    text = normalize_text(text)

    invalid_patterns = [

        "<unk>",

        "<pad>",

        "<sos>",

        "<eos>",

        "unknown",

        "undefined",

        "nan",

        "none",

    ]

    for pattern in invalid_patterns:

        if pattern in text:

            return True

    return False


# ============================================================
# QUALITY EVALUATION
# ============================================================

def evaluate_translation(

    igbo,

    english,

    tokenizer

):

    source_words = get_words(
        igbo
    )

    target_words = get_words(
        english
    )

    reasons = []

    score = 100

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if not source_words:

        return 0, ["empty_source"]

    if not target_words:

        return 0, ["empty_translation"]

    if len(source_words) < MIN_SOURCE_WORDS:

        reasons.append(
            "source_too_short"
        )

        score -= 20

    if len(target_words) < MIN_TARGET_WORDS:

        reasons.append(
            "translation_too_short"
        )

        score -= 40

    if len(target_words) > MAX_TARGET_WORDS:

        reasons.append(
            "translation_too_long"
        )

        score -= 30

    # --------------------------------------------------------
    # Invalid output
    # --------------------------------------------------------

    if contains_invalid_output(

        english

    ):

        reasons.append(
            "invalid_output"
        )

        score -= 100

    # --------------------------------------------------------
    # Length ratio
    # --------------------------------------------------------

    length_ratio = (

        len(target_words)

        /

        max(

            len(source_words),

            1

        )

    )

    if length_ratio < MIN_LENGTH_RATIO:

        reasons.append(

            "translation_too_short_relative_to_source"

        )

        score -= 40

    if length_ratio > MAX_LENGTH_RATIO:

        reasons.append(

            "translation_too_long_relative_to_source"

        )

        score -= 40

    # --------------------------------------------------------
    # Repetition
    # --------------------------------------------------------

    rep_ratio = repetition_ratio(

        target_words

    )

    if rep_ratio > MAX_REPETITION_RATIO:

        reasons.append(

            "excessive_word_repetition"

        )

        score -= 50

    if has_repeated_phrase(

        target_words

    ):

        reasons.append(

            "repeated_phrase"

        )

        score -= 40

    # --------------------------------------------------------
    # Unknown vocabulary
    # --------------------------------------------------------

    unknown_count = 0

    for word in target_words:

        if word not in tokenizer.token_to_id:

            unknown_count += 1

    unknown_ratio = (

        unknown_count

        /

        max(

            len(target_words),

            1

        )

    )

    if unknown_ratio > MAX_UNKNOWN_RATIO:

        reasons.append(

            "excessive_unknown_vocabulary"

        )

        score -= 30

    # --------------------------------------------------------
    # Suspicious output
    # --------------------------------------------------------

    if len(set(target_words)) == 1:

        reasons.append(

            "single_word_repetition"

        )

        score -= 60

    if (

        len(target_words) >= 3

        and

        target_words[0] == target_words[-1]

    ):

        reasons.append(

            "same_start_and_end_word"

        )

        score -= 20

    score = max(

        0,

        score

    )

    return score, reasons


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(

        "STRONG ROUND 1 QUALITY FILTER"

    )

    print("=" * 60)

    print()

    # --------------------------------------------------------
    # Load raw synthetic translations
    # --------------------------------------------------------

    df = pd.read_csv(

        INPUT_FILE

    )

    print(

        f"Raw synthetic pairs: {len(df)}"

    )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer = SimpleTokenizer.load(

        TOKENIZER_FILE

    )

    accepted_rows = []

    rejected_rows = []

    # --------------------------------------------------------
    # Evaluate translations
    # --------------------------------------------------------

    for _, row in df.iterrows():

        score, reasons = evaluate_translation(

            row["igbo"],

            row["english"],

            tokenizer

        )

        row_data = row.to_dict()

        row_data["quality_score"] = score

        row_data["rejection_reasons"] = (

            ";".join(reasons)

            if reasons

            else ""

        )

        # Strict acceptance rule
        if (

            score >= 70

            and

            len(

                get_words(

                    row["english"]

                )

            ) >= MIN_TARGET_WORDS

            and

            not reasons

        ):

            accepted_rows.append(

                row_data

            )

        else:

            rejected_rows.append(

                row_data

            )

    # --------------------------------------------------------
    # DataFrames
    # --------------------------------------------------------

    accepted_df = pd.DataFrame(

        accepted_rows

    )

    rejected_df = pd.DataFrame(

        rejected_rows

    )

    # --------------------------------------------------------
    # Remove duplicate translations
    # --------------------------------------------------------

    if not accepted_df.empty:

        duplicate_mask = (

            accepted_df[

                "english"

            ]

            .astype(str)

            .str.lower()

            .duplicated(

                keep="first"

            )

        )

        duplicate_rows = (

            accepted_df[

                duplicate_mask

            ]

            .copy()

        )

        duplicate_rows[

            "rejection_reasons"

        ] = "duplicate_translation"

        rejected_df = pd.concat(

            [

                rejected_df,

                duplicate_rows

            ],

            ignore_index=True

        )

        accepted_df = accepted_df[

            ~duplicate_mask

        ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )

    accepted_df.to_csv(

        FILTERED_FILE,

        index=False,

        encoding="utf-8"

    )

    rejected_df.to_csv(

        REJECTED_FILE,

        index=False,

        encoding="utf-8"

    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(

        "QUALITY FILTER COMPLETE"

    )

    print("=" * 60)

    print()

    print(

        f"Original pairs:     {len(df)}"

    )

    print(

        f"Accepted pairs:     {len(accepted_df)}"

    )

    print(

        f"Rejected pairs:     {len(rejected_df)}"

    )

    print()

    print(

        "Saved accepted translations:"

    )

    print(

        FILTERED_FILE

    )

    print()

    print(

        "Saved rejected translations:"

    )

    print(

        REJECTED_FILE

    )

    print()

    print(

        "Sample accepted translations:"

    )

    if not accepted_df.empty:

        print(

            accepted_df[

                [

                    "igbo",

                    "english",

                    "quality_score"

                ]

            ]

            .head(10)

            .to_string(

                index=False

            )

        )

    else:

        print(

            "No translations passed the filter."

        )


if __name__ == "__main__":

    main()