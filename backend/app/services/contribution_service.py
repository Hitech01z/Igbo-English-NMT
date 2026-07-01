import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[3]

FILE = BASE_DIR / "dataset" / "contributions" / "user_contributions.csv"


def save_contribution(data):

    row = pd.DataFrame([{

        "igbo": data.igbo,

        "english": data.english,

        "domain": data.domain,

        "contributor": data.contributor,

        "verified": "No",

        "date_added": datetime.now().strftime("%Y-%m-%d")

    }])

    FILE.parent.mkdir(parents=True, exist_ok=True)

    file_exists = FILE.exists()

    row.to_csv(

        FILE,

        mode="a",

        header=not file_exists,

        index=False

    )

    return {

        "success": True,

        "message": "Contribution saved successfully."

    }