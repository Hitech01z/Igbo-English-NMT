import pandas as pd


def clean_dataset(input_file, output_file):

    df = pd.read_csv(input_file)

    df = df.dropna()

    df = df.drop_duplicates()

    df["igbo"] = df["igbo"].astype(str).str.strip()

    df["english"] = df["english"].astype(str).str.strip()

    df = df[
        (df["igbo"] != "")
        & (df["english"] != "")
    ]

    df.to_csv(output_file, index=False)

    print(f"Final rows: {len(df)}")