from fastapi import APIRouter
from pydantic import BaseModel
from pathlib import Path
import csv

router = APIRouter()

FILE = (
    Path(__file__).resolve().parents[1]
    / "contributions.csv"
)


class Contribution(BaseModel):
    igbo: str
    english: str


@router.post("/contribute")
def contribute(data: Contribution):

    file_exists = FILE.exists()

    with open(FILE, "a", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["igbo", "english"])

        writer.writerow([
            data.igbo,
            data.english
        ])

    return {
        "message": "Contribution submitted successfully."
    }