from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PARALLEL_DIR = (
    BASE_DIR
    / "dataset"
    / "parallel"
)

OUTPUT_DIR = (
    BASE_DIR
    / "processed"
    / "baseline"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "all_parallel.csv"
)


# ============================================================
# DOMAINS
# ============================================================

DOMAINS = [
    "agriculture",
    "business",
    "culture",
    "daily_conversation",
    "education",
    "general",
    "government",
    "health",
    "religion",
    "technology",
]


# ============================================================
# START
# ============================================================

print("=" * 60)
print("PREPARING 10-DOMAIN PARALLEL DATASET")
print("=" * 60)


all_data = []


# ============================================================
# LOAD ALL DOMAINS
# ============================================================

for domain in DOMAINS:

    file_path = (
        PARALLEL_DIR
        / f"{domain}.csv"
    )

    if not file_path.exists():

        print(
            f"WARNING: Missing {file_path}"
        )

        continue

    df = pd.read_csv(
        file_path
    )

    print(
        f"{domain:20s}: {len(df)} pairs"
    )

    # Keep only required columns
    df = df[
        [
            "igbo",
            "english"
        ]
    ].copy()

    # Add domain explicitly
    df["domain"] = domain

    all_data.append(
        df
    )


# ============================================================
# COMBINE
# ============================================================

combined = pd.concat(
    all_data,
    ignore_index=True
)


# ============================================================
# CLEAN TEXT
# ============================================================

combined["igbo"] = (
    combined["igbo"]
    .fillna("")
    .astype(str)
    .str.strip()
)

combined["english"] = (
    combined["english"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Remove empty pairs

combined = combined[
    (combined["igbo"] != "")
    &
    (combined["english"] != "")
]


# Remove exact duplicate sentence pairs

combined = combined.drop_duplicates(
    subset=[
        "igbo",
        "english"
    ]
)


# Shuffle deterministically

combined = combined.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


# Add unique IDs

combined.insert(
    0,
    "id",
    [
        f"PAR{i:05d}"
        for i in range(
            1,
            len(combined) + 1
        )
    ]
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


combined.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# REPORT
# ============================================================

print()

print("=" * 60)
print("PARALLEL DATASET PREPARATION COMPLETE")
print("=" * 60)

print(
    f"Total unique pairs: {len(combined)}"
)

print()

print("Domain distribution:")

print(
    combined["domain"].value_counts()
)

print()

print(
    f"Saved to: {OUTPUT_FILE}"
)