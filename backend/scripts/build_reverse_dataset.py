from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

SOURCE_DIR = BASE_DIR / "dataset" / "parallel"
OUTPUT_DIR = BASE_DIR / "dataset" / "parallel_reverse"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

total_rows = 0

print("=" * 60)
print("BUILDING ENGLISH → IGBO DATASET")
print("=" * 60)

for csv_file in sorted(SOURCE_DIR.glob("*.csv")):

    print(f"\nProcessing {csv_file.name}")

    df = pd.read_csv(csv_file)

    columns = [c.lower() for c in df.columns]

    igbo_col = None
    english_col = None

    for c in df.columns:

        if c.lower() in [
            "igbo",
            "source",
            "src"
        ]:
            igbo_col = c

        if c.lower() in [
            "english",
            "target",
            "trg"
        ]:
            english_col = c

    if igbo_col is None or english_col is None:

        print("Skipped (columns not recognised)")
        continue

    reverse_df = pd.DataFrame({
        "english": df[english_col],
        "igbo": df[igbo_col]
    })

    reverse_df.to_csv(
        OUTPUT_DIR / csv_file.name,
        index=False,
        encoding="utf-8"
    )

    print(f"Saved {len(reverse_df)} rows")

    total_rows += len(reverse_df)

print("\n" + "=" * 60)
print(f"Finished.")
print(f"Total sentence pairs: {total_rows}")
print("=" * 60)