import pandas as pd

df = pd.read_csv("../dataset/final_dataset.csv")

df = df[["igbo","english"]]

df = df.dropna()

df.to_csv(

    "../dataset/processed/training_dataset.csv",

    index=False

)

print(

    "Training Dataset Ready"

)