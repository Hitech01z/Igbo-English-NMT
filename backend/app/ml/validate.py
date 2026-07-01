import pandas as pd


def validate_dataset(path):

    df = pd.read_csv(path)

    print("\nRows:", len(df))

    print("\nMissing values:")

    print(df.isnull().sum())

    print("\nDuplicate rows:")

    print(df.duplicated().sum())

    print("\nColumns:")

    print(df.columns.tolist())