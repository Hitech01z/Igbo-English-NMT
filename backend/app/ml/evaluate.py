from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction
)


def compute_bleu(reference, prediction):

    smoothie = (
        SmoothingFunction().method1
    )

    return sentence_bleu(
        [reference.split()],
        prediction.split(),
        smoothing_function=smoothie
    )