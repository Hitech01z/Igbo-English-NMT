import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "evaluation",
    "domain_analysis",
    "sentence_level_model_comparison.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "evaluation",
    "final_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("FINAL SENTENCE-LEVEL MODEL ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

baseline_col = "word_overlap_percent_Baseline"
round1_col = "word_overlap_percent_Round 1 Iterative Back-Translation"
round2_col = "word_overlap_percent_Round 2 Iterative Back-Translation"

baseline = 0
round1 = 0
round2 = 0
ties = 0

winners = []

for _, row in df.iterrows():

    scores = {
        "Baseline": float(row[baseline_col]),
        "Round 1": float(row[round1_col]),
        "Round 2": float(row[round2_col])
    }

    best = max(scores.values())

    best_models = [
        k
        for k, v in scores.items()
        if v == best
    ]

    if len(best_models) == 1:

        winner = best_models[0]

        if winner == "Baseline":
            baseline += 1

        elif winner == "Round 1":
            round1 += 1

        else:
            round2 += 1

    else:

        winner = "Tie"
        ties += 1

    winners.append(winner)

df["winner"] = winners

total = len(df)

print()
print(f"Total sentences : {total}")
print()

print(f"Baseline wins : {baseline}")
print(f"Round 1 wins  : {round1}")
print(f"Round 2 wins  : {round2}")
print(f"Ties          : {ties}")

print()

print("Percentages")
print(f"Baseline : {baseline/total*100:.2f}%")
print(f"Round 1  : {round1/total*100:.2f}%")
print(f"Round 2  : {round2/total*100:.2f}%")
print(f"Ties     : {ties/total*100:.2f}%")

output = os.path.join(
    OUTPUT_DIR,
    "sentence_level_winner_analysis.csv"
)

df.to_csv(output, index=False)

print()
print("Saved:")
print(output)

print("=" * 70)
print("COMPLETE")
print("=" * 70)