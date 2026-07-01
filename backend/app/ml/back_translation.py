import pandas as pd


def iterative_back_translation(path):

    df = pd.read_csv(path)

    synthetic = df.copy()

    # Simulate generated translations
    synthetic["english"] = synthetic["english"] + " (bt)"

    combined = pd.concat(
        [df, synthetic],
        ignore_index=True
    )

    combined.to_csv(
        "data/backtranslated_dataset.csv",
        index=False
    )

    print(
        f"New dataset size: {len(combined)}"
    )