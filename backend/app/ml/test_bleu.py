from app.ml.evaluate import compute_bleu


samples = [

    ("hello world", "hello world"),

    ("good morning", "good morning"),

    ("my name is john", "my name is john")

]


scores = []

for ref, pred in samples:

    scores.append(

        compute_bleu(ref, pred)

    )


print(

    sum(scores) / len(scores)

)