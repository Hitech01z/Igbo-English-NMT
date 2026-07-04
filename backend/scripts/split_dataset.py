import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(__file__)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "final_dataset_clean.csv"
)

df = pd.read_csv(INPUT_FILE)

train, temp = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
)

valid, test = train_test_split(
    temp,
    test_size=0.5,
    random_state=42,
)

os.makedirs("train/data", exist_ok=True)

train.to_csv("train/data/train.csv", index=False)
valid.to_csv("train/data/valid.csv", index=False)
test.to_csv("train/data/test.csv", index=False)

print("Dataset split successfully!")
print(f"Train: {len(train)}")
print(f"Validation: {len(valid)}")
print(f"Test: {len(test)}")
