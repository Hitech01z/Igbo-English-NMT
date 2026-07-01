from transformers import (
    M2M100Tokenizer,
    M2M100ForConditionalGeneration
)

MODEL_NAME = "facebook/m2m100_418M"

tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)

print("Model loaded successfully!")