from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "dataset" / "parallel"
OUTPUT_DIR = BASE_DIR / "processed" / "round1"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

frames = []

for file in sorted(DATASET_DIR.glob("*.csv")):
    print(f"Loading {file.name}")

    df = pd.read_csv(file)

    # keep only required columns
    df = df[["english", "igbo"]]

    # convert to string
    df["english"] = df["english"].astype(str)
    df["igbo"] = df["igbo"].astype(str)

    # trim spaces
    df["english"] = df["english"].str.strip()
    df["igbo"] = df["igbo"].str.strip()

    # lowercase english
    df["english"] = df["english"].str.lower()

    frames.append(df)

dataset = pd.concat(frames, ignore_index=True)

print("Before cleaning:", len(dataset))

dataset.drop_duplicates(inplace=True)

dataset = dataset[
    (dataset["english"] != "")
    & (dataset["igbo"] != "")
]

dataset.reset_index(drop=True, inplace=True)

print("After cleaning:", len(dataset))

dataset.to_csv(
    OUTPUT_DIR / "dataset.csv",
    index=False
)

print("\nSaved:")
print(OUTPUT_DIR / "dataset.csv")