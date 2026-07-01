import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

DATASET = BASE_DIR / "dataset" / "processed"

def load_train():

    return pd.read_csv(

        DATASET / "train.csv"

    )

def load_validation():

    return pd.read_csv(

        DATASET / "validation.csv"

    )

def load_test():

    return pd.read_csv(

        DATASET / "test.csv"

    )