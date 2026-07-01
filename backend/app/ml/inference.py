from app.ml.model_loader import load_model


model, tokenizer = load_model()


def translate_text(text):

    demo = {

        "Ndewo": "Hello",

        "Ụtụtụ ọma": "Good morning",

        "Kedu ka ị mere?": "How are you?",

        "Aha m bụ Chinedu": "My name is Chinedu",

        "Daalụ": "Thank you",

        "Bịa ebe a": "Come here",

        "Ka chi fo": "Good night"

    }

    return demo.get(

        text,

        f"[Demo Translation] {text}"

    )