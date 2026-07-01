import pandas as pd
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer
)

MODEL = "facebook/m2m100_418M"

tokenizer = M2M100Tokenizer.from_pretrained(MODEL)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL)

df = pd.read_csv("dataset/final_dataset.csv")

new_rows = []

for _, row in df.iterrows():

    english = row["english"]

    tokenizer.src_lang = "en"

    encoded = tokenizer(
        english,
        return_tensors="pt"
    )

    generated = model.generate(
        **encoded,
        forced_bos_token_id=
        tokenizer.get_lang_id("ig")
    )

    igbo = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True
    )[0]

    new_rows.append({
        "igbo": igbo,
        "english": english,
        "domain": row["domain"]
    })

pd.DataFrame(new_rows).to_csv(
    "dataset/backtranslated_dataset.csv",
    index=False
)