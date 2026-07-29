import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(BASE_DIR)
)


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = (
    BASE_DIR
    / "processed"
    / "back_translation"
    / "round1_raw.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "processed"
    / "back_translation"
    / "round1_manual_review.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "CREATING ROUND 1 MANUAL REVIEW FILE"
    )

    print("=" * 60)

    print()

    # --------------------------------------------------------
    # Load raw synthetic translations
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Raw synthetic pairs: {len(df)}"
    )

    # --------------------------------------------------------
    # Add verification columns
    # --------------------------------------------------------

    df["verified"] = ""

    df["review_notes"] = ""

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    preferred_columns = [

        "id",

        "igbo",

        "english",

        "domain",

        "source",

        "verified",

        "review_notes"

    ]

    existing_columns = [

        column

        for column in preferred_columns

        if column in df.columns

    ]

    df = df[existing_columns]

    # --------------------------------------------------------
    # Save manual review file
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    df.to_csv(

        OUTPUT_FILE,

        index=False,

        encoding="utf-8-sig"

    )

    print()

    print("=" * 60)

    print(
        "MANUAL REVIEW FILE CREATED"
    )

    print("=" * 60)

    print()

    print(
        f"Total translations to review: {len(df)}"
    )

    print()

    print(
        "Saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "Review instructions:"
    )

    print(
        "verified = yes  -> translation is correct"
    )

    print(
        "verified = no   -> translation is incorrect"
    )

    print(
        "review_notes     -> optional explanation"
    )


if __name__ == "__main__":

    main()