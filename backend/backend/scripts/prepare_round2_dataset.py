import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


ORIGINAL_TRAIN_FILE = (
    BASE_DIR
    / "processed"
    / "round1"
    / "splits"
    / "train.csv"
)


REVIEWED_FILE = (
    BASE_DIR
    / "processed"
    / "back_translation"
    / "round2_manual_review.csv"
)


OUTPUT_FILE = (
    BASE_DIR
    / "processed"
    / "round2"
    / "round2_combined.csv"
)


def main():

    print("=" * 60)
    print("PREPARING ROUND 2 TRAINING DATASET")
    print("=" * 60)

    original = pd.read_csv(
        ORIGINAL_TRAIN_FILE
    )

    print()
    print(
        f"Round 1 training pairs: {len(original)}"
    )

    reviewed = pd.read_csv(
        REVIEWED_FILE
    )

    print(
        f"Reviewed Round 2 pairs: {len(reviewed)}"
    )

    accepted = reviewed[
        reviewed["verified"]
        .astype(str)
        .str.lower()
        .str.strip()
        == "yes"
    ].copy()

    accepted["english"] = accepted.apply(

        lambda row:

        row["corrected_english"]

        if (
            pd.notna(
                row["corrected_english"]
            )
            and str(
                row["corrected_english"]
            ).strip()
            != ""
        )

        else row["english"],

        axis=1

    )

    accepted = accepted[
        [
            "igbo",
            "english"
        ]
    ].copy()

    accepted = accepted.drop_duplicates(
        subset=[
            "igbo",
            "english"
        ]
    )

    original = original[
        [
            "igbo",
            "english"
        ]
    ].copy()

    original["source"] = (
        "round1_training_data"
    )

    accepted["source"] = (
        "iterative_back_translation_round_2"
    )

    combined = pd.concat(
        [
            original,
            accepted
        ],
        ignore_index=True
    )

    combined = combined.drop_duplicates(
        subset=[
            "igbo",
            "english"
        ]
    ).reset_index(
        drop=True
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 60)
    print("ROUND 2 DATASET CREATED")
    print("=" * 60)

    print()
    print(
        f"Round 1 training pairs: {len(original)}"
    )

    print(
        f"Accepted Round 2 pairs: {len(accepted)}"
    )

    print(
        f"Final Round 2 pairs: {len(combined)}"
    )

    print()
    print("Saved to:")
    print(OUTPUT_FILE)

    print()
    print("Source distribution:")
    print(
        combined["source"].value_counts()
    )


if __name__ == "__main__":
    main()