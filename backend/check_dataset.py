import pandas as pd


df = pd.read_csv("data/final_dataset.csv")

print("Rows:", len(df))

print(df.head())

print(df.isnull().sum())