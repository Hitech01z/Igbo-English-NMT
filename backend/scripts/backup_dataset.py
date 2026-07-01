import shutil

from datetime import datetime

timestamp = datetime.now().strftime(

    "%Y%m%d_%H%M"

)

shutil.copy(

    "../dataset/final_dataset.csv",

    f"../dataset/reports/backup_{timestamp}.csv"

)

print("Backup Created")