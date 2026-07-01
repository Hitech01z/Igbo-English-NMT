import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

FILE = BASE_DIR / "dataset" / "final_dataset.csv"

df = pd.read_csv(FILE)

print()

print("============================")

print("DATASET STATISTICS")

print("============================")

print()

print("Total Sentences")

print(len(df))

print()

if "domain" in df.columns:

    print("Domain Distribution")

    print(df["domain"].value_counts())

print()

if "source" in df.columns:

    print("Source Distribution")

    print(df["source"].value_counts())

print()

if "verified" in df.columns:

    print("Verification Status")

    print(df["verified"].value_counts())