"""
DOMAIN-BASED MODEL PREDICTION ANALYSIS

This script:
1. Loads the common baseline test set containing the correct domains.
2. Loads prediction files from:
   - Baseline
   - Round 1
   - Round 2
3. Assigns the correct domain to Round 1 and Round 2 predictions
   by matching the Igbo sentence.
4. Compares model performance by domain.
5. Shows successful and failed translation examples.
6. Saves detailed CSV analysis files.

Run from the backend directory:

    python scripts/analyze_predictions_by_domain.py
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd
import re


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

BASELINE_TEST_FILE = (
    BASE_DIR
    / "processed"
    / "baseline"
    / "test.csv"
)

BASELINE_PREDICTIONS_FILE = (
    BASE_DIR
    / "evaluation"
    / "baseline_predictions.csv"
)

ROUND1_PREDICTIONS_FILE = (
    BASE_DIR
    / "evaluation"
    / "round1_fixed_predictions.csv"
)

ROUND2_PREDICTIONS_FILE = (
    BASE_DIR
    / "evaluation"
    / "round2_predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "evaluation"
    / "domain_analysis"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalizes text for reliable matching.

    This helps match the same Igbo sentence even when there
    are small differences in spacing or capitalization.
    """

    if pd.isna(text):
        return ""

    text = str(text).strip().lower()

    # Normalize repeated spaces
    text = re.sub(r"\s+", " ", text)

    # Normalize spaces before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text


# ============================================================
# LOAD BASELINE TEST SET
# ============================================================

def load_baseline_test_set():
    """
    Loads the common baseline test set.

    This file is treated as the authoritative source for
    domain labels.
    """

    if not BASELINE_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Baseline test file not found:\n{BASELINE_TEST_FILE}"
        )

    df = pd.read_csv(BASELINE_TEST_FILE)

    required_columns = {
        "igbo",
        "english",
        "domain"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Baseline test file is missing required columns:\n"
            f"{missing}\n\n"
            f"Available columns: {list(df.columns)}"
        )

    df["match_key"] = df["igbo"].apply(normalize_text)

    # Keep only the columns needed for domain mapping
    domain_map = (
        df[
            [
                "match_key",
                "igbo",
                "english",
                "domain"
            ]
        ]
        .drop_duplicates(
            subset=["match_key"]
        )
    )

    return df, domain_map


# ============================================================
# LOAD PREDICTION FILE
# ============================================================

def load_prediction_file(
    filepath,
    model_name
):
    """
    Loads a model prediction file.

    Supported prediction columns include:

        predicted_english
        prediction
        predicted
        translation

    Supported reference columns include:

        reference_english
        english
        reference
    """

    if not filepath.exists():
        raise FileNotFoundError(
            f"{model_name} prediction file not found:\n{filepath}"
        )

    df = pd.read_csv(filepath)

    print(f"\n{model_name} prediction columns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # FIND IGBO COLUMN
    # --------------------------------------------------------

    igbo_candidates = [
        "igbo",
        "source",
        "source_igbo",
        "input"
    ]

    igbo_column = None

    for column in igbo_candidates:

        if column in df.columns:
            igbo_column = column
            break

    if igbo_column is None:

        raise ValueError(
            f"\nCould not find Igbo/source column in:\n"
            f"{filepath}\n"
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # FIND REFERENCE COLUMN
    # --------------------------------------------------------

    reference_candidates = [
        "reference_english",
        "english",
        "reference",
        "target",
        "actual_english"
    ]

    reference_column = None

    for column in reference_candidates:

        if column in df.columns:
            reference_column = column
            break

    if reference_column is None:

        raise ValueError(
            f"\nCould not find reference English column in:\n"
            f"{filepath}\n"
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # FIND PREDICTION COLUMN
    # --------------------------------------------------------

    prediction_candidates = [
        "predicted_english",
        "prediction",
        "predicted",
        "translation",
        "generated_english"
    ]

    prediction_column = None

    for column in prediction_candidates:

        if column in df.columns:
            prediction_column = column
            break

    if prediction_column is None:

        raise ValueError(
            f"\nCould not find prediction column in:\n"
            f"{filepath}\n"
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # STANDARDIZE COLUMNS
    # --------------------------------------------------------

    result = pd.DataFrame()

    result["igbo"] = df[igbo_column]

    result["reference_english"] = (
        df[reference_column]
    )

    result["predicted_english"] = (
        df[prediction_column]
    )

    result["match_key"] = (
        result["igbo"]
        .apply(normalize_text)
    )

    result["model"] = model_name

    return result


# ============================================================
# ASSIGN REAL DOMAINS
# ============================================================

def assign_domains(
    predictions,
    domain_map
):
    """
    Assigns the correct domain to prediction rows by matching
    the normalized Igbo sentence against the baseline test set.
    """

    predictions = predictions.merge(
        domain_map[
            [
                "match_key",
                "domain"
            ]
        ],
        on="match_key",
        how="left"
    )

    unmatched = (
        predictions["domain"]
        .isna()
        .sum()
    )

    if unmatched > 0:

        print(
            f"\nWARNING:"
            f" {unmatched} prediction(s) "
            f"could not be matched to a domain."
        )

        predictions["domain"] = (
            predictions["domain"]
            .fillna("unknown")
        )

    return predictions


# ============================================================
# TEXT QUALITY HELPERS
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    return str(text).strip()


def exact_match(
    prediction,
    reference
):

    prediction = normalize_text(
        prediction
    )

    reference = normalize_text(
        reference
    )

    return prediction == reference


def word_overlap(
    prediction,
    reference
):

    prediction_words = set(
        normalize_text(
            prediction
        ).split()
    )

    reference_words = set(
        normalize_text(
            reference
        ).split()
    )

    if not reference_words:

        return 0.0

    overlap = (
        prediction_words
        & reference_words
    )

    return (
        len(overlap)
        /
        len(reference_words)
        *
        100
    )


# ============================================================
# ADD SUCCESS / FAILURE INFORMATION
# ============================================================

def add_quality_analysis(df):

    df = df.copy()

    df["exact_match"] = df.apply(
        lambda row: exact_match(
            row["predicted_english"],
            row["reference_english"]
        ),
        axis=1
    )

    df["word_overlap_percent"] = df.apply(
        lambda row: word_overlap(
            row["predicted_english"],
            row["reference_english"]
        ),
        axis=1
    )

    df["prediction_length"] = (
        df["predicted_english"]
        .fillna("")
        .astype(str)
        .apply(
            lambda x: len(x.split())
        )
    )

    df["reference_length"] = (
        df["reference_english"]
        .fillna("")
        .astype(str)
        .apply(
            lambda x: len(x.split())
        )
    )

    return df


# ============================================================
# DOMAIN SUMMARY
# ============================================================

def create_domain_summary(
    all_predictions
):

    summary_rows = []

    for (
        model,
        model_df
    ) in all_predictions.groupby(
        "model"
    ):

        for (
            domain,
            domain_df
        ) in model_df.groupby(
            "domain"
        ):

            total = len(
                domain_df
            )

            exact_matches = (
                domain_df[
                    "exact_match"
                ]
                .sum()
            )

            average_overlap = (
                domain_df[
                    "word_overlap_percent"
                ]
                .mean()
            )

            summary_rows.append(
                {
                    "model": model,
                    "domain": domain,
                    "test_samples": total,
                    "exact_matches": int(
                        exact_matches
                    ),
                    "exact_match_percent": round(
                        exact_matches
                        /
                        total
                        *
                        100,
                        2
                    ),
                    "average_word_overlap_percent": round(
                        average_overlap,
                        2
                    )
                }
            )

    return pd.DataFrame(
        summary_rows
    )


# ============================================================
# DOMAIN COMPARISON TABLE
# ============================================================

def create_comparison_table(
    summary
):

    comparison_rows = []

    domains = sorted(
        summary[
            "domain"
        ]
        .unique()
    )

    for domain in domains:

        row = {
            "domain": domain
        }

        domain_data = summary[
            summary["domain"]
            == domain
        ]

        for _, record in domain_data.iterrows():

            model = record[
                "model"
            ]

            model_short_name = (
                model
                .lower()
                .replace(
                    " ",
                    "_"
                )
            )

            row[
                f"{model_short_name}_exact_match_percent"
            ] = record[
                "exact_match_percent"
            ]

            row[
                f"{model_short_name}_word_overlap_percent"
            ] = record[
                "average_word_overlap_percent"
            ]

        comparison_rows.append(
            row
        )

    return pd.DataFrame(
        comparison_rows
    )


# ============================================================
# FIND BEST MODEL PER DOMAIN
# ============================================================

def find_best_models(
    summary
):

    rows = []

    for domain in sorted(
        summary[
            "domain"
        ]
        .unique()
    ):

        domain_data = summary[
            summary["domain"]
            == domain
        ]

        best_overlap = (
            domain_data
            .sort_values(
                "average_word_overlap_percent",
                ascending=False
            )
            .iloc[0]
        )

        rows.append(
            {
                "domain": domain,
                "best_model_by_word_overlap":
                    best_overlap[
                        "model"
                    ],
                "best_word_overlap_percent":
                    best_overlap[
                        "average_word_overlap_percent"
                    ]
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SHOW EXAMPLES
# ============================================================

def show_examples(
    all_predictions,
    number_of_examples=3
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SUCCESSFUL TRANSLATION EXAMPLES"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # BEST EXAMPLES
    # --------------------------------------------------------

    successful = (
        all_predictions
        .sort_values(
            [
                "word_overlap_percent",
                "exact_match"
            ],
            ascending=False
        )
    )

    for model in sorted(
        successful[
            "model"
        ]
        .unique()
    ):

        print(
            f"\n"
            f"MODEL: {model}"
        )

        model_data = successful[
            successful[
                "model"
            ]
            == model
        ]

        shown = 0

        for _, row in model_data.iterrows():

            if shown >= number_of_examples:

                break

            print(
                "\n"
                f"Domain: {row['domain']}"
            )

            print(
                f"Igbo: {row['igbo']}"
            )

            print(
                f"Reference: "
                f"{row['reference_english']}"
            )

            print(
                f"Prediction: "
                f"{row['predicted_english']}"
            )

            print(
                f"Word overlap: "
                f"{row['word_overlap_percent']:.2f}%"
            )

            shown += 1

    # --------------------------------------------------------
    # FAILURE EXAMPLES
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FAILED TRANSLATION EXAMPLES"
    )

    print(
        "=" * 70
    )

    failed = (
        all_predictions
        .sort_values(
            "word_overlap_percent",
            ascending=True
        )
    )

    for model in sorted(
        failed[
            "model"
        ]
        .unique()
    ):

        print(
            f"\n"
            f"MODEL: {model}"
        )

        model_data = failed[
            failed[
                "model"
            ]
            == model
        ]

        shown = 0

        for _, row in model_data.iterrows():

            if shown >= number_of_examples:

                break

            print(
                "\n"
                f"Domain: {row['domain']}"
            )

            print(
                f"Igbo: {row['igbo']}"
            )

            print(
                f"Reference: "
                f"{row['reference_english']}"
            )

            print(
                f"Prediction: "
                f"{row['predicted_english']}"
            )

            print(
                f"Word overlap: "
                f"{row['word_overlap_percent']:.2f}%"
            )

            shown += 1


# ============================================================
# FIND SENTENCE-LEVEL MODEL COMPARISONS
# ============================================================

def create_sentence_comparison(
    all_predictions
):

    pivot = (
        all_predictions
        .pivot_table(
            index=[
                "match_key",
                "igbo",
                "reference_english",
                "domain"
            ],
            columns="model",
            values=[
                "predicted_english",
                "word_overlap_percent",
                "exact_match"
            ],
            aggfunc="first"
        )
        .reset_index()
    )

    # Flatten multi-level column names
    flattened_columns = []

    for column in pivot.columns:

        if isinstance(
            column,
            tuple
        ):

            flattened_columns.append(
                "_".join(
                    [
                        str(
                            item
                        )
                        for item in column
                        if item
                    ]
                )
            )

        else:

            flattened_columns.append(
                str(
                    column
                )
            )

    pivot.columns = (
        flattened_columns
    )

    return pivot


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "CORRECTED DOMAIN-BASED MODEL PREDICTION ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "\nLoading common baseline test set..."
    )

    baseline_test, domain_map = (
        load_baseline_test_set()
    )

    print(
        f"Baseline test pairs: "
        f"{len(baseline_test)}"
    )

    print(
        f"Available domains: "
        f"{sorted(baseline_test['domain'].unique())}"
    )

    # --------------------------------------------------------
    # LOAD PREDICTIONS
    # --------------------------------------------------------

    baseline = load_prediction_file(
        BASELINE_PREDICTIONS_FILE,
        "Baseline"
    )

    round1 = load_prediction_file(
        ROUND1_PREDICTIONS_FILE,
        "Round 1 Iterative Back-Translation"
    )

    round2 = load_prediction_file(
        ROUND2_PREDICTIONS_FILE,
        "Round 2 Iterative Back-Translation"
    )

    # --------------------------------------------------------
    # ASSIGN DOMAINS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ASSIGNING REAL DOMAINS"
    )

    print(
        "=" * 70
    )

    baseline = assign_domains(
        baseline,
        domain_map
    )

    round1 = assign_domains(
        round1,
        domain_map
    )

    round2 = assign_domains(
        round2,
        domain_map
    )

    print(
        "\nBaseline domain distribution:"
    )

    print(
        baseline[
            "domain"
        ]
        .value_counts()
    )

    print(
        "\nRound 1 domain distribution:"
    )

    print(
        round1[
            "domain"
        ]
        .value_counts()
    )

    print(
        "\nRound 2 domain distribution:"
    )

    print(
        round2[
            "domain"
        ]
        .value_counts()
    )

    # --------------------------------------------------------
    # ADD QUALITY ANALYSIS
    # --------------------------------------------------------

    baseline = add_quality_analysis(
        baseline
    )

    round1 = add_quality_analysis(
        round1
    )

    round2 = add_quality_analysis(
        round2
    )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    all_predictions = pd.concat(
        [
            baseline,
            round1,
            round2
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SAVE FULL PREDICTIONS WITH DOMAINS
    # --------------------------------------------------------

    baseline.to_csv(
        OUTPUT_DIR
        / "baseline_predictions_with_domains.csv",
        index=False
    )

    round1.to_csv(
        OUTPUT_DIR
        / "round1_predictions_with_domains.csv",
        index=False
    )

    round2.to_csv(
        OUTPUT_DIR
        / "round2_predictions_with_domains.csv",
        index=False
    )

    all_predictions.to_csv(
        OUTPUT_DIR
        / "all_predictions_with_domains.csv",
        index=False
    )

    # --------------------------------------------------------
    # DOMAIN SUMMARY
    # --------------------------------------------------------

    summary = create_domain_summary(
        all_predictions
    )

    comparison = create_comparison_table(
        summary
    )

    best_models = find_best_models(
        summary
    )

    # --------------------------------------------------------
    # SENTENCE COMPARISON
    # --------------------------------------------------------

    sentence_comparison = (
        create_sentence_comparison(
            all_predictions
        )
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    summary.to_csv(
        OUTPUT_DIR
        / "domain_model_summary.csv",
        index=False
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "domain_comparison_table.csv",
        index=False
    )

    best_models.to_csv(
        OUTPUT_DIR
        / "best_model_by_domain.csv",
        index=False
    )

    sentence_comparison.to_csv(
        OUTPUT_DIR
        / "sentence_level_model_comparison.csv",
        index=False
    )

    # --------------------------------------------------------
    # DISPLAY DOMAIN SUMMARY
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DOMAIN PERFORMANCE SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # DISPLAY BEST MODEL BY DOMAIN
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BEST MODEL BY DOMAIN"
    )

    print(
        "=" * 70
    )

    print(
        best_models.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # DISPLAY EXAMPLES
    # --------------------------------------------------------

    show_examples(
        all_predictions
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nSaved files:"
    )

    print(
        f"  {OUTPUT_DIR / 'baseline_predictions_with_domains.csv'}"
    )

    print(
        f"  {OUTPUT_DIR / 'round1_predictions_with_domains.csv'}"
    )

    print(
        f"  {OUTPUT_DIR / 'round2_predictions_with_domains.csv'}"
    )

    print(
        f"  {OUTPUT_DIR / 'all_predictions_with_domains.csv'}"
    )

    print(
        f"  {OUTPUT_DIR / 'domain_model_summary.csv'}"
    )

    print(
        f"  {OUTPUT_DIR / 'domain_comparison_table.csv'}"
    )

    print(
        f"  {OUTPUT_DIR / 'best_model_by_domain.csv'}"
    )

    print(
        f"  {OUTPUT_DIR / 'sentence_level_model_comparison.csv'}"
    )

    print(
        "\n"
        + "=" * 70
    )


if __name__ == "__main__":

    main()