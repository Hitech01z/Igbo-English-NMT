import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv("../dataset/final_dataset.csv")

train, temp = train_test_split(

    df,

    test_size=0.2,

    random_state=42

)

validation, test = train_test_split(

    temp,

    test_size=0.5,

    random_state=42

)

train.to_csv(

    "../dataset/train/train.csv",

    index=False

)

validation.to_csv(

    "../dataset/validation/validation.csv",

    index=False

)

test.to_csv(

    "../dataset/test/test.csv",

    index=False

)

print("Dataset Split Completed")