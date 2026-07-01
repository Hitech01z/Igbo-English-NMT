from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATASET_DIR = BASE_DIR / "dataset"

OUTPUT_FILE = (
    BASE_DIR
    / "backend"
    / "data"
    / "final_dataset_clean.csv"
)

all_data = []


for csv_file in DATASET_DIR.rglob("*.csv"):

    if "contributions" in str(csv_file):
        continue

    try:

        df = pd.read_csv(csv_file)

        required = ["igbo", "english"]

        if not all(
            col in df.columns
            for col in required
        ):
            print(
                f"Skipping {csv_file.name}"
            )
            continue

        df = df[["igbo", "english"]]

        df = df.dropna()

        df["igbo"] = (
            df["igbo"]
            .astype(str)
            .str.strip()
        )

        df["english"] = (
            df["english"]
            .astype(str)
            .str.strip()
        )

        df = df[
            (df["igbo"] != "")
            &
            (df["english"] != "")
        ]

        all_data.append(df)

        print(
            f"Loaded {csv_file.name}: "
            f"{len(df)} rows"
        )

    except Exception as e:

        print(
            f"Error in {csv_file}: {e}"
        )


final_df = pd.concat(
    all_data,
    ignore_index=True
)

final_df = final_df.drop_duplicates()

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n===================")
print(
    f"TOTAL: {len(final_df)} rows"
)
print("===================")