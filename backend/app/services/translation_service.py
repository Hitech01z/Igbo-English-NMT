from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
HISTORY_FILE = BASE_DIR / "data" / "translation_history.csv"


def save_history(text, translation):

    row = pd.DataFrame([
        {
            "input": text,
            "output": translation
        }
    ])

    if HISTORY_FILE.exists():
        row.to_csv(
            HISTORY_FILE,
            mode="a",
            header=False,
            index=False
        )
    else:
        row.to_csv(
            HISTORY_FILE,
            index=False
        )   

def translate(
    text,
    source_language="igbo",
    target_language="english"
):

    demo = {

        "Ndewo": "Hello",
        "Ụtụtụ ọma": "Good morning",
        "Kedu ka ị mere?": "How are you?",
        "Aha m bụ Chinedu": "My name is Chinedu",
        "Daalụ": "Thank you"

    }

    if source_language == "igbo":

        translation = demo.get(
            text,
            f"[Demo EN] {text}"
        )

    else:

        reverse_demo = {
            v: k
            for k, v in demo.items()
        }

        translation = reverse_demo.get(
            text,
            f"[Demo IG] {text}"
        )

    save_history(
        text,
        translation,
        
    )

    return translation