import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "processed"
    / "round1"
    / "round1_combined.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "processed"
    / "round1"
    / "splits"
)

TRAIN_FILE = OUTPUT_DIR / "train.csv"
VALID_FILE = OUTPUT_DIR / "valid.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"


# Fixed split proportions
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10


# Fixed random seed ensures reproducible splits
RANDOM_STATE = 42


def main():

    print("=" * 60)
    print("CREATING ROUND 1 DATASET SPLITS")
    print("=" * 60)

    # -------------------------
    # Load Round 1 dataset
    # -------------------------

    df = pd.read_csv(INPUT_FILE)

    print(f"Total Round 1 pairs: {len(df)}")

    # -------------------------
    # Shuffle dataset
    # -------------------------

    df = df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # -------------------------
    # Calculate split sizes
    # -------------------------

    total = len(df)

    train_size = int(
        total * TRAIN_RATIO
    )

    valid_size = int(
        total * VALID_RATIO
    )

    # Remaining records go to test
    # This prevents loss of rows due to rounding

    train_df = df.iloc[
        :train_size
    ]

    valid_df = df.iloc[
        train_size:
        train_size + valid_size
    ]

    test_df = df.iloc[
        train_size + valid_size:
    ]

    # -------------------------
    # Create output directory
    # -------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------
    # Save splits
    # -------------------------

    train_df.to_csv(
        TRAIN_FILE,
        index=False,
        encoding="utf-8"
    )

    valid_df.to_csv(
        VALID_FILE,
        index=False,
        encoding="utf-8"
    )

    test_df.to_csv(
        TEST_FILE,
        index=False,
        encoding="utf-8"
    )

    # -------------------------
    # Display results
    # -------------------------

    print()
    print("=" * 60)
    print("ROUND 1 DATASET SPLITS CREATED")
    print("=" * 60)

    print(f"Training:   {len(train_df)}")
    print(f"Validation: {len(valid_df)}")
    print(f"Testing:    {len(test_df)}")

    print()
    print("Saved to:")
    print(
        "processed/round1/splits/"
    )

    print()
    print("Files created:")
    print(
        "  train.csv"
    )
    print(
        "  valid.csv"
    )
    print(
        "  test.csv"
    )


if __name__ == "__main__":

    main()