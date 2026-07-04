from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
from sacrebleu import corpus_bleu
import pandas as pd

MODEL_PATH = "models/m2m100-igbo-en"

tokenizer = M2M100Tokenizer.from_pretrained(MODEL_PATH)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_PATH)

test_df = pd.read_csv("train/data/test.csv")

predictions = []
references = []

for _, row in test_df.iterrows():

    tokenizer.src_lang = "en"

    inputs = tokenizer(
        row["english"],
        return_tensors="pt"
    )

    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.get_lang_id("ig")
    )

    pred = tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    )

    predictions.append(pred)
    references.append([row["igbo"]])

bleu = corpus_bleu(predictions, references)

print("BLEU Score:", bleu.score)