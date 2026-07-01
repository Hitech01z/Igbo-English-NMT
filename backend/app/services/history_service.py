from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

HISTORY_FILE = (
    BASE_DIR / "data" / "translation_history.csv"
)


def get_history():

    if not HISTORY_FILE.exists():

        return []

    df = pd.read_csv(HISTORY_FILE)

    return df.to_dict(
        orient="records"
    )