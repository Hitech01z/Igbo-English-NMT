import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


INPUT_FILE = (
    BASE_DIR
    / "processed"
    / "back_translation"
    / "round2_raw.csv"
)


OUTPUT_FILE = (
    BASE_DIR
    / "processed"
    / "back_translation"
    / "round2_manual_review.csv"
)


def main():

    print("=" * 60)

    print(
        "CREATING ROUND 2 MANUAL REVIEW FILE"
    )

    print("=" * 60)

    df = pd.read_csv(
        INPUT_FILE
    )

    print()

    print(
        f"Raw synthetic pairs: "
        f"{len(df)}"
    )

    # Add verification columns

    df["verified"] = ""

    df["corrected_english"] = ""

    df["review_notes"] = ""

    # Keep only the required columns

    columns = [

        "id",

        "igbo",

        "english",

        "corrected_english",

        "domain",

        "source",

        "verified",

        "review_notes"

    ]

    df = df[columns]

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
        "ROUND 2 MANUAL REVIEW FILE CREATED"
    )

    print("=" * 60)

    print()

    print(
        f"Total translations to review: "
        f"{len(df)}"
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
        "Columns:"
    )

    print(
        "verified          -> yes or no"
    )

    print(
        "corrected_english -> corrected translation"
    )

    print(
        "review_notes      -> optional explanation"
    )


if __name__ == "__main__":

    main()