from fastapi import APIRouter
import pandas as pd
from pathlib import Path


router = APIRouter()

FILE = Path(
    "dataset/contributions/user_contributions.csv"
)


@router.get("/pending")
def pending():

    if not FILE.exists():
        return []

    df = pd.read_csv(FILE)

    return df[
        df["verified"] == "No"
    ].to_dict(
        orient="records"
    )