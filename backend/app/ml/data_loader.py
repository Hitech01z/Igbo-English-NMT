import pandas as pd


def load_parallel_dataset(path):

    df = pd.read_csv(path)

    return list(
        zip(
            df["igbo"],
            df["english"]
        )
    )