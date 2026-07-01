import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET = BASE_DIR / "dataset"

frames = []

for csv in DATASET.rglob("*.csv"):

    if csv.name == "final_dataset.csv":

        continue

    try:

        df = pd.read_csv(csv)

        frames.append(df)

    except:

        pass

merged = pd.concat(

    frames,

    ignore_index=True

)

merged = merged.drop_duplicates()

merged.to_csv(

    DATASET / "final_dataset.csv",

    index=False

)

print("Merged Successfully")

print("Rows:", len(merged))