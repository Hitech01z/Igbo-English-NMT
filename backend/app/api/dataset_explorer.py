from fastapi import APIRouter
import pandas as pd
from pathlib import Path

router = APIRouter()

DATASET = Path("data/final_dataset.csv")


@router.get("/")
def get_dataset():

    if not DATASET.exists():
        return []

    df = pd.read_csv(DATASET)

    return df.head(50).to_dict(
        orient="records"
    )