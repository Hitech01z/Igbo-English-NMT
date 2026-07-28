from pathlib import Path
import random
import pandas as pd

random.seed(42)

BASE = Path(__file__).resolve().parents[1]

DATA = pd.read_csv(
    BASE / "processed/round1/dataset.csv"
)

OUTPUT = BASE / "processed/round1/dataset_augmented.csv"

pronouns = {
    "i":"we",
    "we":"they",
    "he":"she",
    "she":"he",
    "they":"we",
}

verbs = {
    "go":"travel",
    "come":"arrive",
    "buy":"purchase",
    "eat":"consume",
    "see":"meet",
    "want":"need",
    "need":"require",
}

time_words = [
    "today",
    "now",
    "tomorrow",
    "later"
]

new_rows = []

for _, row in DATA.iterrows():

    en = row["english"]
    ig = row["igbo"]

    new_rows.append((en, ig))

    words = en.split()

    # Pronoun variation
    if len(words) > 0:

        first = words[0].lower()

        if first in pronouns:

            temp = words.copy()
            temp[0] = pronouns[first]

            new_rows.append(
                (
                    " ".join(temp),
                    ig
                )
            )

    # Verb variation

    temp = []

    changed = False

    for w in words:

        lw = w.lower()

        if lw in verbs:

            temp.append(verbs[lw])

            changed = True

        else:

            temp.append(w)

    if changed:

        new_rows.append(
            (
                " ".join(temp),
                ig
            )
        )

    # Time variation

    new_rows.append(
        (
            en + " " + random.choice(time_words),
            ig
        )
    )

augmented = pd.DataFrame(
    new_rows,
    columns=[
        "english",
        "igbo"
    ]
)

augmented.drop_duplicates(inplace=True)

augmented.reset_index(drop=True, inplace=True)

print()

print("Original :", len(DATA))

print("Augmented:", len(augmented))

augmented.to_csv(
    OUTPUT,
    index=False
)

print()

print("Saved")

print(OUTPUT)