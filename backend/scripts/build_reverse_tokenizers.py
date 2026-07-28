from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

import pandas as pd

from nmt.tokenizer import SimpleTokenizer

DATASET_DIR = BASE_DIR / "dataset" / "parallel_reverse"

OUTPUT_DIR = BASE_DIR / "processed" / "english_to_igbo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("BUILDING ENGLISH → IGBO TOKENIZERS")
print("=" * 60)

english_sentences = []
igbo_sentences = []

files = sorted(DATASET_DIR.glob("*.csv"))

print(f"Found {len(files)} dataset files.\n")

for file in files:

    print(f"Reading {file.name}")

    df = pd.read_csv(file)

    english_sentences.extend(
        df["english"].astype(str).tolist()
    )

    igbo_sentences.extend(
        df["igbo"].astype(str).tolist()
    )

print("\nTotal English sentences:", len(english_sentences))
print("Total Igbo sentences:", len(igbo_sentences))

print("\nBuilding English tokenizer...")

english_tokenizer = SimpleTokenizer()

english_tokenizer.build_vocab(
    english_sentences
)

print(
    f"English vocabulary size: {len(english_tokenizer)}"
)

print("\nBuilding Igbo tokenizer...")

igbo_tokenizer = SimpleTokenizer()

igbo_tokenizer.build_vocab(
    igbo_sentences
)

print(
    f"Igbo vocabulary size: {len(igbo_tokenizer)}"
)

english_tokenizer.save(
    OUTPUT_DIR / "english_tokenizer.json"
)

igbo_tokenizer.save(
    OUTPUT_DIR / "igbo_tokenizer.json"
)

print("\nSaved:")

print(
    OUTPUT_DIR / "english_tokenizer.json"
)

print(
    OUTPUT_DIR / "igbo_tokenizer.json"
)

print("=" * 60)
print("TOKENIZER BUILD COMPLETE")
print("=" * 60)