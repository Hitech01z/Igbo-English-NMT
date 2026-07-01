import pandas as pd

df = pd.read_csv("../dataset/final_dataset.csv")

before = len(df)

df = df.drop_duplicates()

after = len(df)

df.to_csv("../dataset/final_dataset.csv", index=False)

print("Before:", before)

print("After:", after)

print("Removed:", before-after)