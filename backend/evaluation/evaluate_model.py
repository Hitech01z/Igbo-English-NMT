from pathlib import Path
import pandas as pd
from tqdm import tqdm
import sacrebleu

from inference.dual_translator import translate

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = (
    BASE_DIR
    / "processed"
    / "round2"
    / "splits"
    / "test.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "evaluation"
    / "results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "predictions.csv"

# ============================================================
# LOAD TEST DATA
# ============================================================

df = pd.read_csv(TEST_FILE)

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)
print(f"Loaded {len(df)} test sentence pairs")

predictions = []
references = []

results = []

# ============================================================
# TRANSLATE EACH TEST SENTENCE
# ============================================================

for _, row in tqdm(df.iterrows(), total=len(df)):

    igbo = str(row["igbo"]).strip()
    english = str(row["english"]).strip()

    prediction = translate(
        text=igbo,
        source_language="ig",
        target_language="en"
    )

    predictions.append(prediction)
    references.append([english])

    results.append(
        {
            "igbo": igbo,
            "reference": english,
            "prediction": prediction
        }
    )

    # ============================================================
# CALCULATE BLEU & chrF++
# ============================================================

bleu = sacrebleu.corpus_bleu(
    predictions,
    list(zip(*references))
)

chrf = sacrebleu.corpus_chrf(
    predictions,
    list(zip(*references))
)

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print()
print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(f"BLEU Score : {bleu.score:.2f}")
print(f"chrF++     : {chrf.score:.2f}")

print()
print(f"Predictions saved to:")
print(OUTPUT_FILE)

print("=" * 60)

# ============================================================
# SAVE METRICS
# ============================================================

METRICS_FILE = OUTPUT_DIR / "metrics.txt"

with open(METRICS_FILE, "w", encoding="utf-8") as f:

    f.write("IGBO-ENGLISH NMT EVALUATION\n")
    f.write("=" * 40 + "\n\n")

    f.write(f"Test Samples : {len(results)}\n")
    f.write(f"BLEU Score   : {bleu.score:.2f}\n")
    f.write(f"chrF++ Score : {chrf.score:.2f}\n")

print(f"\nMetrics saved to:\n{METRICS_FILE}")

# ============================================================
# SHOW SAMPLE TRANSLATIONS
# ============================================================

print("\nSample Translations")
print("=" * 60)

sample_size = min(10, len(results_df))

for i in range(sample_size):

    print(f"\nExample {i + 1}")

    print(f"Igbo      : {results_df.iloc[i]['igbo']}")

    print(f"Reference : {results_df.iloc[i]['reference']}")

    print(f"Prediction: {results_df.iloc[i]['prediction']}")

print("\nEvaluation Complete.")
print("=" * 60)