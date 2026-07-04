import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "backend/data/final_dataset.csv"
)

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42
)

train_df.to_csv(
    "train/data/train.csv",
    index=False
)

valid_df.to_csv(
    "train/data/valid.csv",
    index=False
)

test_df.to_csv(
    "train/data/test.csv",
    index=False
)

print("Dataset prepared successfully.")