import pandas as pd

from app.ml.inference import (
    translate_english_to_igbo
)


def generate_synthetic_dataset(
    input_file,
    output_file
):

    df = pd.read_csv(input_file)

    results = []

    for sentence in df["english"]:

        try:

            igbo = translate_english_to_igbo(
                sentence
            )

            results.append(
                {
                    "english": sentence,
                    "igbo": igbo
                }
            )

        except Exception:

            continue

    pd.DataFrame(results).to_csv(
        output_file,
        index=False
    )

    print("Synthetic dataset created.")