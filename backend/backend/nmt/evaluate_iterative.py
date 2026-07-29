from pathlib import Path

import pandas as pd

from nmt.tokenizer import SimpleTokenizer


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = Path(
    "processed/round1/splits/train.csv"
)

OUTPUT_DIR = Path(
    "processed/round1/tokenizers"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("BUILDING ROUND 1 TOKENIZERS")
print("=" * 60)


df = pd.read_csv(
    TRAIN_FILE
)


print(
    f"Training pairs: {len(df)}"
)


# ============================================================
# BUILD IGBO TOKENIZER
# ============================================================

igbo_tokenizer = SimpleTokenizer(
    min_freq=1
)

igbo_tokenizer.build_vocab(
    df["igbo"].tolist()
)


# ============================================================
# BUILD ENGLISH TOKENIZER
# ============================================================

english_tokenizer = SimpleTokenizer(
    min_freq=1
)

english_tokenizer.build_vocab(
    df["english"].tolist()
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True

)


igbo_tokenizer.save(

    OUTPUT_DIR
    /
    "igbo_tokenizer.json"

)


english_tokenizer.save(

    OUTPUT_DIR
    /
    "english_tokenizer.json"

)


# ============================================================
# REPORT
# ============================================================

print()

print(
    f"Igbo vocabulary: "
    f"{len(igbo_tokenizer)}"
)

print(
    f"English vocabulary: "
    f"{len(english_tokenizer)}"
)

print()

print(
    "Saved:"
)

print(
    OUTPUT_DIR
    /
    "igbo_tokenizer.json"
)

print(
    OUTPUT_DIR
    /
    "english_tokenizer.json"
)