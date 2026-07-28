import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVALUATION_DIR = os.path.join(
    BASE_DIR,
    "evaluation"
)

DOMAIN_DIR = os.path.join(
    EVALUATION_DIR,
    "domain_analysis"
)

OUTPUT_DIR = os.path.join(
    EVALUATION_DIR,
    "final_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FILE PATHS
# ============================================================

BASELINE_SCORES = os.path.join(
    EVALUATION_DIR,
    "baseline_scores.csv"
)

ROUND1_SCORES = os.path.join(
    EVALUATION_DIR,
    "round1_fixed_scores.csv"
)

ROUND2_SCORES = os.path.join(
    EVALUATION_DIR,
    "round2_scores.csv"
)

DOMAIN_SUMMARY = os.path.join(
    DOMAIN_DIR,
    "domain_model_summary.csv"
)

BEST_DOMAIN = os.path.join(
    DOMAIN_DIR,
    "best_model_by_domain.csv"
)

SENTENCE_COMPARISON = os.path.join(
    DOMAIN_DIR,
    "sentence_level_model_comparison.csv"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_model_name(name):

    name = str(name).lower()

    if "baseline" in name:
        return "Baseline"

    if "round 1" in name or "round1" in name:
        return "Round 1 Iterative Back-Translation"

    if "round 2" in name or "round2" in name:
        return "Round 2 Iterative Back-Translation"

    return str(name)


def load_score_file(path, model_name):

    df = pd.read_csv(path)

    # Format 1:
    # metric, score
    if "metric" in df.columns and "score" in df.columns:

        result = {}

        for _, row in df.iterrows():

            metric = str(
                row["metric"]
            ).lower().strip()

            score = float(
                row["score"]
            )

            if "bleu" in metric:

                result["BLEU"] = score

            elif "chrf" in metric:

                result["chrF++"] = score

        return {
            "Model": model_name,
            "BLEU": result.get("BLEU", np.nan),
            "chrF++": result.get("chrF++", np.nan),
            "Test Samples": "N/A"
        }

    # Format 2:
    # model, test_samples, bleu, chrf_plus_plus
    if (
        "bleu" in df.columns
        and "chrf_plus_plus" in df.columns
    ):

        row = df.iloc[0]

        test_samples = (
            row["test_samples"]
            if "test_samples" in df.columns
            else "N/A"
        )

        return {
            "Model": model_name,
            "BLEU": float(
                row["bleu"]
            ),
            "chrF++": float(
                row["chrf_plus_plus"]
            ),
            "Test Samples": test_samples
        }

    raise ValueError(
        f"Unsupported score file format:\n"
        f"{path}\n"
        f"Columns: {list(df.columns)}"
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 70)

    print(
        "FINAL EXPERIMENT ANALYSIS"
    )

    print("=" * 70)

    print()

    # ========================================================
    # LOAD OVERALL RESULTS
    # ========================================================

    print(
        "Loading overall evaluation results..."
    )

    baseline = load_score_file(
        BASELINE_SCORES,
        "Baseline"
    )

    round1 = load_score_file(
        ROUND1_SCORES,
        "Round 1 Iterative Back-Translation"
    )

    round2 = load_score_file(
        ROUND2_SCORES,
        "Round 2 Iterative Back-Translation"
    )

    overall_df = pd.DataFrame(
        [
            baseline,
            round1,
            round2
        ]
    )

    print()

    print("=" * 70)

    print(
        "OVERALL PERFORMANCE"
    )

    print("=" * 70)

    print(
        overall_df.to_string(
            index=False
        )
    )

    # ========================================================
    # BASELINE COMPARISON
    # ========================================================

    baseline_bleu = baseline["BLEU"]
    baseline_chrf = baseline["chrF++"]

    comparison_rows = []

    for result in [
        round1,
        round2
    ]:

        bleu_change = (
            result["BLEU"]
            - baseline_bleu
        )

        chrf_change = (
            result["chrF++"]
            - baseline_chrf
        )

        bleu_percent = (
            bleu_change
            / baseline_bleu
            * 100
        )

        chrf_percent = (
            chrf_change
            / baseline_chrf
            * 100
        )

        comparison_rows.append({

            "Model": result["Model"],

            "BLEU": result["BLEU"],

            "chrF++": result["chrF++"],

            "BLEU Change": round(
                bleu_change,
                2
            ),

            "BLEU Change Percent": round(
                bleu_percent,
                2
            ),

            "chrF++ Change": round(
                chrf_change,
                2
            ),

            "chrF++ Change Percent": round(
                chrf_percent,
                2
            )

        })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    print()

    print("=" * 70)

    print(
        "CHANGE FROM BASELINE"
    )

    print("=" * 70)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    # ========================================================
    # BEST OVERALL MODELS
    # ========================================================

    best_bleu_row = overall_df.loc[
        overall_df["BLEU"].idxmax()
    ]

    best_chrf_row = overall_df.loc[
        overall_df["chrF++"].idxmax()
    ]

    print()

    print("=" * 70)

    print(
        "BEST OVERALL RESULTS"
    )

    print("=" * 70)

    print()

    print(
        "Best BLEU:"
    )

    print(
        f"  Model: {best_bleu_row['Model']}"
    )

    print(
        f"  Score: {best_bleu_row['BLEU']:.2f}"
    )

    print()

    print(
        "Best chrF++:"
    )

    print(
        f"  Model: {best_chrf_row['Model']}"
    )

    print(
        f"  Score: {best_chrf_row['chrF++']:.2f}"
    )

    # ========================================================
    # LOAD DOMAIN RESULTS
    # ========================================================

    print()

    print(
        "Loading domain analysis..."
    )

    domain_df = pd.read_csv(
        DOMAIN_SUMMARY
    )

    best_domain_df = pd.read_csv(
        BEST_DOMAIN
    )

    sentence_df = pd.read_csv(
        SENTENCE_COMPARISON
    )

    # ========================================================
    # DOMAIN PERFORMANCE
    # ========================================================

    print()

    print("=" * 70)

    print(
        "DOMAIN-LEVEL ANALYSIS"
    )

    print("=" * 70)

    print()

    print(
        domain_df.to_string(
            index=False
        )
    )

    # ========================================================
    # COUNT DOMAIN WINS
    # ========================================================

    print()

    print("=" * 70)

    print(
        "DOMAIN WIN COUNT"
    )

    print("=" * 70)

    domain_wins = (
        best_domain_df[
            "best_model_by_word_overlap"
        ]
        .value_counts()
    )

    for model, count in domain_wins.items():

        print(
            f"{model}: {count} domain(s)"
        )

    # ========================================================
    # SENTENCE-LEVEL ANALYSIS
    # ========================================================

    print()

    print("=" * 70)

    print(
        "SENTENCE-LEVEL COMPARISON"
    )

    print("=" * 70)

    # Detect likely model columns

    model_columns = [
        col
        for col in sentence_df.columns
        if "model" in col.lower()
        or "winner" in col.lower()
    ]

    print()

    print(
        "Sentence comparison columns:"
    )

    print(
        list(
            sentence_df.columns
        )
    )

    # ========================================================
    # SUCCESS COUNT
    # ========================================================

    success_summary = {}

    for model in [
        "Baseline",
        "Round 1 Iterative Back-Translation",
        "Round 2 Iterative Back-Translation"
    ]:

        success_summary[
            model
        ] = 0

    # Search winner columns

    for column in sentence_df.columns:

        if (
            "winner" in column.lower()
            or "best_model" in column.lower()
        ):

            values = (
                sentence_df[column]
                .astype(str)
            )

            for model in success_summary:

                success_summary[
                    model
                ] += values.str.contains(
                    model,
                    case=False,
                    na=False
                ).sum()

    print()

    print(
        "Model wins from sentence-level comparison:"
    )

    for model, count in success_summary.items():

        print(
            f"  {model}: {count}"
        )

    # ========================================================
    # RESEARCH INTERPRETATION
    # ========================================================

    print()

    print("=" * 70)

    print(
        "RESEARCH INTERPRETATION"
    )

    print("=" * 70)

    print()

    if (
        round1["BLEU"]
        < baseline["BLEU"]
        and
        round1["chrF++"]
        > baseline["chrF++"]
    ):

        print(
            "1. Round 1 produced a lower BLEU score "
            "than the baseline but a higher chrF++ score."
        )

        print()

        print(
            "   This indicates that Round 1 did not "
            "consistently reproduce complete reference "
            "translations, but it achieved improved "
            "character-level similarity in the test set."
        )

    if (
        round2["BLEU"]
        < round1["BLEU"]
        and
        round2["chrF++"]
        < round1["chrF++"]
    ):

        print()

        print(
            "2. Round 2 performed worse than Round 1 "
            "on both BLEU and chrF++."
        )

        print()

        print(
            "   This suggests that additional iterative "
            "back-translation introduced noise or error "
            "propagation into the training data."
        )

    print()

    print(
        "3. The results demonstrate that the effectiveness "
        "of iterative back-translation depends on the "
        "quality of the synthetic translations added to "
        "the training corpus."
    )

    print()

    print(
        "4. The strongest result should not be determined "
        "using only one metric. BLEU measures n-gram "
        "precision, while chrF++ is more sensitive to "
        "character-level similarity and can be useful "
        "for morphologically rich and low-resource languages."
    )

    print()

    print(
        "5. The domain analysis shows that the effect of "
        "back-translation varies across subject domains."
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    overall_path = os.path.join(
        OUTPUT_DIR,
        "overall_performance.csv"
    )

    comparison_path = os.path.join(
        OUTPUT_DIR,
        "baseline_comparison.csv"
    )

    domain_path = os.path.join(
        OUTPUT_DIR,
        "domain_performance.csv"
    )

    domain_wins_path = os.path.join(
        OUTPUT_DIR,
        "domain_win_counts.csv"
    )

    overall_df.to_csv(
        overall_path,
        index=False
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    domain_df.to_csv(
        domain_path,
        index=False
    )

    domain_wins_df = (
        domain_wins
        .reset_index()
    )

    domain_wins_df.columns = [
        "Model",
        "Domain Wins"
    ]

    domain_wins_df.to_csv(
        domain_wins_path,
        index=False
    )

    print()

    print("=" * 70)

    print(
        "FINAL EXPERIMENT ANALYSIS COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        "Saved files:"
    )

    print(
        f"  {overall_path}"
    )

    print(
        f"  {comparison_path}"
    )

    print(
        f"  {domain_path}"
    )

    print(
        f"  {domain_wins_path}"
    )

    print()

    print(
        "You can now use these results for:"
    )

    print(
        "  - Chapter Four results and analysis"
    )

    print(
        "  - Chapter Five discussion and conclusion"
    )

    print(
        "  - Project defense presentation"
    )


if __name__ == "__main__":

    main()