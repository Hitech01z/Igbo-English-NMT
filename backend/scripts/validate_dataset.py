import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = BASE_DIR / "dataset"

print("\n==============================")
print("DATASET VALIDATION")
print("==============================\n")

total_rows = 0

for csv_file in DATASET_DIR.rglob("*.csv"):

    if csv_file.name == "final_dataset.csv":
        continue

    try:

        df = pd.read_csv(csv_file)

        print(f"File: {csv_file.relative_to(DATASET_DIR)}")
        print(f"Rows: {len(df)}")
        print(f"Duplicates: {df.duplicated().sum()}")
        print(f"Missing Values: {df.isnull().sum().sum()}")

        print("-----------------------------")

        total_rows += len(df)

    except Exception as e:

        print(f"Error reading {csv_file.name}")

        print(e)

print()

print("Total Dataset Rows:", total_rows)