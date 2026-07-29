from pathlib import Path
import random
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

random.seed(42)

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "processed"
    / "round1"
    / "dataset_augmented.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "processed"
    / "round2"
    / "splits"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("SPLITTING DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} sentence pairs")

# ============================================================
# CLEAN
# ============================================================

df = df.dropna()

df["english"] = df["english"].astype(str).str.strip()
df["igbo"] = df["igbo"].astype(str).str.strip()

df = df[
    (df["english"] != "")
    &
    (df["igbo"] != "")
]

df = df.drop_duplicates()

print(f"After cleaning: {len(df)}")

# ============================================================
# SHUFFLE
# ============================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ============================================================
# SPLIT
# ============================================================

n = len(df)

train_end = int(n * TRAIN_RATIO)
valid_end = train_end + int(n * VALID_RATIO)

train = df.iloc[:train_end]
valid = df.iloc[train_end:valid_end]
test = df.iloc[valid_end:]

# ============================================================
# SAVE
# ============================================================

train.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

valid.to_csv(
    OUTPUT_DIR / "valid.csv",
    index=False
)

test.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("DATASET SPLIT COMPLETE")
print("=" * 60)

print(f"Total      : {len(df)}")
print(f"Train      : {len(train)}")
print(f"Validation : {len(valid)}")
print(f"Test       : {len(test)}")

print()
print("Saved files:")

print(OUTPUT_DIR / "train.csv")
print(OUTPUT_DIR / "valid.csv")
print(OUTPUT_DIR / "test.csv")

print("=" * 60)