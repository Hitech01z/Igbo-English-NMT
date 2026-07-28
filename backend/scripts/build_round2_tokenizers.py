import sys
from pathlib import Path

import pandas as pd


# Add backend directory to Python path
BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(BASE_DIR)
)

from nmt.tokenizer import SimpleTokenizer


TRAIN_FILE = (
    BASE_DIR
    / "processed"
    / "round2"
    / "splits"
    / "train.csv"
)


OUTPUT_DIR = (
    BASE_DIR
    / "processed"
    / "round2"
    / "tokenizers"
)


def main():

    print("=" * 60)
    print("BUILDING ROUND 2 TOKENIZERS")
    print("=" * 60)

    df = pd.read_csv(
        TRAIN_FILE
    )

    print()
    print(
        f"Training sentences: {len(df)}"
    )

    igbo_tokenizer = SimpleTokenizer(
        min_freq=1
    )

    english_tokenizer = SimpleTokenizer(
        min_freq=1
    )

    igbo_tokenizer.build_vocab(
        df["igbo"].tolist()
    )

    english_tokenizer.build_vocab(
        df["english"].tolist()
    )

    print(
        f"Igbo vocabulary: "
        f"{len(igbo_tokenizer)}"
    )

    print(
        f"English vocabulary: "
        f"{len(english_tokenizer)}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    igbo_tokenizer.save(
        OUTPUT_DIR
        / "igbo_tokenizer.json"
    )

    english_tokenizer.save(
        OUTPUT_DIR
        / "english_tokenizer.json"
    )

    print()
    print("Saved:")
    print(
        OUTPUT_DIR
        / "igbo_tokenizer.json"
    )

    print(
        OUTPUT_DIR
        / "english_tokenizer.json"
    )


if __name__ == "__main__":
    main()