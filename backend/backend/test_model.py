from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer
)

print("Loading tokenizer...")

tokenizer = M2M100Tokenizer.from_pretrained(
    "facebook/m2m100_418M"
)

print("Loading model...")

model = M2M100ForConditionalGeneration.from_pretrained(
    "facebook/m2m100_418M"
)

print("MODEL LOADED SUCCESSFULLY")