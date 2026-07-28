from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASELINE_TRAIN = Path(
    "processed/splits/train.csv"
)

BASELINE_VALID = Path(
    "processed/splits/valid.csv"
)

BASELINE_TEST = Path(
    "processed/splits/test.csv"
)

SYNTHETIC_DATA = Path(
    "processed/back_translation/round_1_corrected.csv"
)

OUTPUT_DIR = Path(
    "processed/round1/splits"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("PREPARING ITERATIVE BACK-TRANSLATION ROUND 1 SPLITS")
print("=" * 60)


train = pd.read_csv(
    BASELINE_TRAIN
)

valid = pd.read_csv(
    BASELINE_VALID
)

test = pd.read_csv(
    BASELINE_TEST
)

synthetic = pd.read_csv(
    SYNTHETIC_DATA
)


print(
    f"Original training pairs: {len(train)}"
)

print(
    f"Synthetic pairs: {len(synthetic)}"
)

print(
    f"Validation pairs: {len(valid)}"
)

print(
    f"Test pairs: {len(test)}"
)


# ============================================================
# STANDARDIZE COLUMNS
# ============================================================

required_columns = [
    "igbo",
    "english",
    "domain"
]


train = train[
    required_columns
].copy()

valid = valid[
    required_columns
].copy()

test = test[
    required_columns
].copy()

synthetic = synthetic[
    required_columns
].copy()


# ============================================================
# COMBINE TRAINING DATA
# ============================================================

round1_train = pd.concat(

    [
        train,
        synthetic
    ],

    ignore_index=True

)


# Remove duplicate pairs
round1_train = round1_train.drop_duplicates(

    subset=[
        "igbo",
        "english"
    ]

)


# Shuffle only the training set
round1_train = round1_train.sample(

    frac=1,

    random_state=42

).reset_index(

    drop=True

)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True

)


round1_train.to_csv(

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


# ============================================================
# REPORT
# ============================================================

print()

print("=" * 60)

print("ROUND 1 SPLITS CREATED")

print("=" * 60)

print(
    f"Round 1 training: {len(round1_train)}"
)

print(
    f"Validation:       {len(valid)}"
)

print(
    f"Test:              {len(test)}"
)

print()

print(
    f"Saved to: {OUTPUT_DIR}"
)