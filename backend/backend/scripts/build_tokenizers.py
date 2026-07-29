from pathlib import Path
import sys
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))

from nmt.tokenizer import SimpleTokenizer

DATA = pd.read_csv(
    BASE / "processed" / "round1" / "dataset_augmented.csv"
)

OUTPUT = (
    BASE
    / "processed"
    / "round2"
    / "tokenizers"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# English Tokenizer
# --------------------------------------------------

english_tokenizer = SimpleTokenizer(min_freq=1)

english_tokenizer.build_vocab(
    DATA["english"].astype(str).tolist()
)

english_tokenizer.save(
    OUTPUT / "english_tokenizer.json"
)

# --------------------------------------------------
# Igbo Tokenizer
# --------------------------------------------------

igbo_tokenizer = SimpleTokenizer(min_freq=1)

igbo_tokenizer.build_vocab(
    DATA["igbo"].astype(str).tolist()
)

igbo_tokenizer.save(
    OUTPUT / "igbo_tokenizer.json"
)

print("=" * 60)
print("TOKENIZERS BUILT")
print("=" * 60)

print()

print("English Vocabulary :", len(english_tokenizer))
print("Igbo Vocabulary    :", len(igbo_tokenizer))

print()

print("Saved to")

print(OUTPUT)