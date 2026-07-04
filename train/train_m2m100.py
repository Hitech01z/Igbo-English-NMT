from datasets import Dataset
import pandas as pd

from transformers import (
    M2M100Tokenizer,
    M2M100ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_NAME = "facebook/m2m100_418M"

tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)

tokenizer.src_lang = "en"
tokenizer.tgt_lang = "ig"

print("Loading datasets...")

train_df = pd.read_csv("train/data/train.csv")
valid_df = pd.read_csv("train/data/valid.csv")

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)


def preprocess(examples):

    tokenizer.src_lang = "ig"

    inputs = tokenizer(
        examples["igbo"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    targets = tokenizer(
        text_target=examples["english"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = targets["input_ids"]

    return inputs


print("Tokenizing...")

train_dataset = train_dataset.map(preprocess)
valid_dataset = valid_dataset.map(preprocess)

training_args = Seq2SeqTrainingArguments(
    output_dir="trained_model",

    evaluation_strategy="epoch",

    learning_rate=5e-5,

    per_device_train_batch_size=2,

    per_device_eval_batch_size=2,

    num_train_epochs=3,

    weight_decay=0.01,

    save_total_limit=2,

    predict_with_generate=True,

    fp16=False,
)

trainer = Seq2SeqTrainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=valid_dataset,
)

print("Training started...")

trainer.train()

trainer.save_model("trained_model")
tokenizer.save_pretrained("trained_model")

print("Training complete.")