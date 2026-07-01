from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_NAME = "facebook/m2m100_418M"

TRAIN_DATA = BASE_DIR / "data/final_dataset_clean.csv"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"

BATCH_SIZE = 4

EPOCHS = 3

MAX_LENGTH = 128

LEARNING_RATE = 5e-5