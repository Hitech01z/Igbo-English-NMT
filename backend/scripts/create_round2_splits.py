import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


INPUT_FILE = (
    BASE_DIR
    / "processed"
    / "round2"
    / "round2_combined.csv"
)


OUTPUT_DIR = (
    BASE_DIR
    / "processed"
    / "round2"
    / "splits"
)


def main():

    print("=" * 60)
    print("CREATING ROUND 2 DATASET SPLITS")
    print("=" * 60)

    df = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(
        f"Total Round 2 pairs: {len(df)}"
    )

    # Keep the test set fixed from the original dataset.
    # Round 2 training data is added only to the training split.

    round1_train = df[
        df["source"]
        == "round1_training_data"
    ].copy()

    round2_data = df[
        df["source"]
        == "iterative_back_translation_round_2"
    ].copy()

    # Load the original fixed validation and test sets
    valid_file = (
        BASE_DIR
        / "processed"
        / "baseline"
        / "valid.csv"
    )

    test_file = (
        BASE_DIR
        / "processed"
        / "baseline"
        / "test.csv"
    )

    valid = pd.read_csv(
        valid_file
    )

    test = pd.read_csv(
        test_file
    )

    # Combine original Round 1 training data
    # with accepted Round 2 synthetic data.

    train = pd.concat(
        [
            round1_train,
            round2_data
        ],
        ignore_index=True
    )

    # Remove metadata columns if present
    # so the model receives only translation columns.

    train = train[
        [
            "igbo",
            "english"
        ]
    ]

    valid = valid[
        [
            "igbo",
            "english"
        ]
    ]

    test = test[
        [
            "igbo",
            "english"
        ]
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
        encoding="utf-8-sig"
    )

    valid.to_csv(
        OUTPUT_DIR / "valid.csv",
        index=False,
        encoding="utf-8-sig"
    )

    test.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 60)
    print("ROUND 2 DATASET SPLITS CREATED")
    print("=" * 60)

    print()
    print(
        f"Training:   {len(train)}"
    )

    print(
        f"Validation: {len(valid)}"
    )

    print(
        f"Testing:    {len(test)}"
    )

    print()
    print(
        "The validation and test sets remain fixed."
    )

    print()
    print("Saved to:")
    print(OUTPUT_DIR)

    print()
    print("Files created:")
    print("  train.csv")
    print("  valid.csv")
    print("  test.csv")


if __name__ == "__main__":
    main()