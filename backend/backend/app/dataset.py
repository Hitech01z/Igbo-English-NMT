from fastapi import APIRouter
import csv
from pathlib import Path

router = APIRouter()

DATASET = (
    Path(__file__).resolve().parents[1]
    / "processed"
    / "round2"
    / "test.csv"
)


@router.get("/dataset")
def get_dataset():

    rows = []

    with open(DATASET, encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            rows.append(row)

    return rows