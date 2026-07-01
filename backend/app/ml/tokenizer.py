from app.ml.model_loader import load_model


def tokenize_text(text, language="en"):

    _, tokenizer = load_model()

    tokenizer.src_lang = language

    return tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )