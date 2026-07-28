from fastapi import APIRouter

router = APIRouter()


@router.get("/dashboard")
def dashboard():

    return {
        "bleu": 75.66,
        "chrf": 80.90,
        "dataset_size": 2230,
        "training": 1780,
        "validation": 222,
        "test": 223,
        "igbo_vocab": 684,
        "english_vocab": 953,
        "model": "Transformer + Iterative Back Translation",
    }