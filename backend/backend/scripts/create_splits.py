from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "processed"
    / "baseline"
    / "all_parallel.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "processed"
    / "baseline"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("CREATING FIXED DATASET SPLITS")
print("=" * 60)

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Total pairs: {len(df)}"
)


# ============================================================
# FIRST SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    shuffle=True
)


# ============================================================
# SECOND SPLIT
# ============================================================

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    shuffle=True
)


# ============================================================
# SAVE
# ============================================================

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False,
    encoding="utf-8-sig"
)

valid_df.to_csv(
    OUTPUT_DIR / "valid.csv",
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# REPORT
# ============================================================

print()

print("=" * 60)
print("DATASET SPLITS CREATED")
print("=" * 60)

print(
    f"Training:   {len(train_df)}"
)

print(
    f"Validation: {len(valid_df)}"
)

print(
    f"Testing:    {len(test_df)}"
)

print()

print(
    "The test set is now FIXED and must not change."
)