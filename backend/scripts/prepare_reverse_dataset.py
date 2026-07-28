from pathlib import Path
import random
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

random.seed(42)

# ============================================================
# PATHS
# ============================================================

DATASET_DIR = BASE_DIR / "dataset" / "parallel_reverse"

OUTPUT_DIR = (
    BASE_DIR
    / "processed"
    / "english_to_igbo"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("PREPARING ENGLISH → IGBO DATASET")
print("=" * 60)

rows = []

files = sorted(
    DATASET_DIR.glob("*.csv")
)

for file in files:

    print(f"Reading {file.name}")

    df = pd.read_csv(file)

    required_columns = {
        "english",
        "igbo"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{file.name} must contain "
            f"'english' and 'igbo' columns."
        )

    for _, row in df.iterrows():

        english = str(
            row["english"]
        ).strip()

        igbo = str(
            row["igbo"]
        ).strip()

        if english and igbo:

            rows.append(
                {
                    "english": english,
                    "igbo": igbo
                }
            )

print()

print(
    f"Total sentence pairs: {len(rows)}"
)

# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(rows)

# ============================================================
# SPLIT
# ============================================================

total = len(rows)

train_end = int(total * 0.80)

valid_end = int(total * 0.90)

train_rows = rows[:train_end]

valid_rows = rows[train_end:valid_end]

test_rows = rows[valid_end:]

# ============================================================
# DATAFRAMES
# ============================================================

train_df = pd.DataFrame(train_rows)

valid_df = pd.DataFrame(valid_rows)

test_df = pd.DataFrame(test_rows)

# ============================================================
# SAVE CSV FILES
# ============================================================

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

valid_df.to_csv(
    OUTPUT_DIR / "valid.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print()

print(f"Train      : {len(train_df)}")
print(f"Validation : {len(valid_df)}")
print(f"Test       : {len(test_df)}")

print()

print("Saved:")

print(OUTPUT_DIR / "train.csv")

print(OUTPUT_DIR / "valid.csv")

print(OUTPUT_DIR / "test.csv")

print("=" * 60)
print("DATASET READY")
print("=" * 60)