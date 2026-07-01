from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

FINAL_DATASET = BASE_DIR / "data" / "final_dataset.csv"


def get_dataset():

    if not FINAL_DATASET.exists():
        return []

    df = pd.read_csv(FINAL_DATASET)

    df = df.fillna("")

    return df.to_dict(orient="records")


def dataset_statistics():

    if not FINAL_DATASET.exists():

        return {
            "total_sentences": 0,
            "education": 0,
            "health": 0,
            "technology": 0,
            "agriculture": 0,
            "government": 0,
            "culture": 0,
            "ai_ml": 0,
            "nlp": 0,
            "translation": 0,
        }

    df = pd.read_csv(FINAL_DATASET)

    df["domain"] = (
        df["domain"]
        .fillna("")
        .str.strip()
        .str.lower()
    )

    return {
        "total_sentences": len(df),
        "education": len(df[df["domain"] == "education"]),
        "health": len(df[df["domain"] == "health"]),
        "technology": len(df[df["domain"] == "technology"]),
        "agriculture": len(df[df["domain"] == "agriculture"]),
        "government": len(df[df["domain"] == "government"]),
        "culture": len(df[df["domain"] == "culture"]),
        "ai_ml": len(df[df["domain"] == "ai_ml"]),
        "nlp": len(df[df["domain"] == "nlp"]),
        "translation": len(df[df["domain"] == "translation"]),
    }