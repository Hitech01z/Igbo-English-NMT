from app.ml.evaluate import compute_bleu
from app.services.dataset_service import dataset_statistics


def dashboard_statistics():

    stats = dataset_statistics()

    bleu = compute_bleu(
        "hello world",
        "hello world"
    )

    return {
        "total_sentences": stats["total_sentences"],
        "education": stats["education"],
        "translations": stats["total_sentences"],
        "bleu_score": round(bleu, 2),

        "health": stats["health"],
        "technology": stats["technology"],
        "agriculture": stats["agriculture"],
        "government": stats["government"],
        "culture": stats["culture"],

        "ai_ml": stats.get("ai_ml", 0),
        "nlp": stats.get("nlp", 0),
        "translation": stats.get("translation", 0),
    }