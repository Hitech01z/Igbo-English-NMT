import pandas as pd

df = pd.read_csv("../dataset/final_dataset.csv")

report = f"""

DATASET REPORT

=====================

Total Rows:

{len(df)}

Duplicate Rows:

{df.duplicated().sum()}

Missing Values:

{df.isnull().sum().sum()}

"""

with open(

    "../dataset/reports/report.txt",

    "w",

    encoding="utf8"

) as f:

    f.write(report)

print(report)