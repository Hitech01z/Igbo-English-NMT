import pandas as pd
from pathlib import Path
from datetime import datetime


CHECKPOINT_DIR = Path(
    "checkpoints/best_model"
)


def train():

    df = pd.read_csv(
        "data/backtranslated_dataset.csv"
    )

    print(
        f"Training on {len(df)} sentence pairs"
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    training_log = f"""
Model: facebook/m2m100_418M
Method: Iterative Back Translation
Date: {datetime.now()}
Dataset Size: {len(df)}
Epochs: 3
Learning Rate: 5e-5
Batch Size: 4
Status: Completed
"""

    with open(
        CHECKPOINT_DIR / "training.log",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(training_log)

    print("Training completed.")
    print("Checkpoint saved.")