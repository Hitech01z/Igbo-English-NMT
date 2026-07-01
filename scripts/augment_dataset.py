import pandas as pd
from pathlib import Path

INPUT_FILE = Path("../dataset/final_dataset.csv")
OUTPUT_FILE = Path("../dataset/augmented_dataset.csv")


def augment_sentence(text):

    replacements = {
        "student": "learner",
        "teacher": "instructor",
        "school": "academy",
        "book": "textbook",
        "study": "learn",
        "classroom": "lecture room"
    }

    new_text = text

    for old, new in replacements.items():
        new_text = new_text.replace(old, new)

    return new_text


df = pd.read_csv(INPUT_FILE)

new_rows = []

for _, row in df.iterrows():

    english = str(row["english"])

    augmented = augment_sentence(english)

    if augmented != english:

        new_rows.append({
            "igbo": row["igbo"],
            "english": augmented,
            "domain": row["domain"]
        })


augmented_df = pd.DataFrame(new_rows)

combined = pd.concat(
    [df, augmented_df],
    ignore_index=True
)

combined.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Original:", len(df))
print("Augmented:", len(augmented_df))
print("Total:", len(combined))