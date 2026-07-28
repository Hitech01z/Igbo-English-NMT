import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


ORIGINAL_DATASET = (
    BASE_DIR
    / "processed"
    / "baseline"
    / "train.csv"
)


ROUND1_REVIEW_FILE = (
    BASE_DIR
    / "processed"
    / "back_translation"
    / "round1_manual_review.csv"
)


OUTPUT_DIR = (
    BASE_DIR
    / "processed"
    / "round1"
)


OUTPUT_FILE = (
    OUTPUT_DIR
    / "round1_combined.csv"
)


def main():

    print("=" * 60)
    print("PREPARING ROUND 1 TRAINING DATASET")
    print("=" * 60)

    # -------------------------
    # Load original training data
    # -------------------------

    original = pd.read_csv(
        ORIGINAL_DATASET
    )

    print(
        f"Original training pairs: {len(original)}"
    )

    # -------------------------
    # Load reviewed synthetic data
    # -------------------------

    reviewed = pd.read_csv(
        ROUND1_REVIEW_FILE
    )

    print(
        f"Reviewed synthetic pairs: {len(reviewed)}"
    )

    # -------------------------
    # Keep only verified pairs
    # -------------------------

    if "verified" in reviewed.columns:

        reviewed["verified"] = (
            reviewed["verified"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        reviewed = reviewed[
            reviewed["verified"] == "yes"
        ]

    elif "english_corrected" in reviewed.columns:

        reviewed = reviewed[
            reviewed["english_corrected"]
            .notna()
        ]

    else:

        raise ValueError(
            "No verified column or corrected translation column found."
        )

    # -------------------------
    # Select required columns
    # -------------------------

    original = original[
        ["igbo", "english", "domain"]
    ].copy()

    reviewed = reviewed[
        ["igbo", "english", "domain"]
    ].copy()

    # -------------------------
    # Add source labels
    # -------------------------

    original["source"] = "original_parallel"

    reviewed["source"] = (
        "iterative_back_translation_round_1"
    )

    # -------------------------
    # Combine datasets
    # -------------------------

    combined = pd.concat(
        [
            original,
            reviewed
        ],
        ignore_index=True
    )

    # -------------------------
    # Remove duplicates
    # -------------------------

    combined = combined.drop_duplicates(
        subset=["igbo", "english"]
    )

    # -------------------------
    # Shuffle dataset
    # -------------------------

    combined = combined.sample(
        frac=1,
        random_state=42
    ).reset_index(
        drop=True
    )

    # -------------------------
    # Add IDs
    # -------------------------

    combined.insert(
        0,
        "id",
        [
            f"R1_{i:05d}"
            for i in range(
                1,
                len(combined) + 1
            )
        ]
    )

    # -------------------------
    # Save
    # -------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("ROUND 1 DATASET CREATED")
    print("=" * 60)

    print(
        f"Original pairs:  {len(original)}"
    )

    print(
        f"Accepted synthetic pairs: {len(reviewed)}"
    )

    print(
        f"Final pairs:     {len(combined)}"
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print()
    print("Source distribution:")

    print(
        combined["source"]
        .value_counts()
    )


if __name__ == "__main__":

    main()