import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nmt.dataset import TranslationDataset, collate_fn
from nmt.tokenizer import SimpleTokenizer
from nmt.transformer import TransformerNMT


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "processed" / "round2"

TRAIN_FILE = DATA_DIR / "splits" / "train.csv"
VALID_FILE = DATA_DIR / "splits" / "valid.csv"
TEST_FILE  = DATA_DIR / "splits" / "test.csv"

TOKENIZER_DIR = DATA_DIR / "tokenizers"

CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "round2"
LOG_DIR = BASE_DIR / "logs" / "round2"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32

NUM_EPOCHS = 40

LEARNING_RATE = 5e-4

WEIGHT_DECAY = 1e-4

CLIP = 1.0

PATIENCE = 7


# Transformer

D_MODEL = 256

NHEAD = 8

NUM_ENCODER_LAYERS = 3

NUM_DECODER_LAYERS = 3

DIM_FEEDFORWARD = 512

DROPOUT = 0.1

PAD_IDX = 0


# ============================================================
# TOKENIZERS
# ============================================================

def load_tokenizers():

    src_tokenizer = SimpleTokenizer.load(
        TOKENIZER_DIR / "igbo_tokenizer.json"
    )

    trg_tokenizer = SimpleTokenizer.load(
        TOKENIZER_DIR / "english_tokenizer.json"
    )

    return src_tokenizer, trg_tokenizer


# ============================================================
# DATASETS
# ============================================================

def build_datasets(src_tokenizer, trg_tokenizer):

    train_dataset = TranslationDataset(

        TRAIN_FILE,

        src_tokenizer=src_tokenizer,

        trg_tokenizer=trg_tokenizer,

        src_column="igbo",

        trg_column="english"

    )

    valid_dataset = TranslationDataset(

        VALID_FILE,

        src_tokenizer=src_tokenizer,

        trg_tokenizer=trg_tokenizer,

        src_column="igbo",

        trg_column="english"

    )

    return train_dataset, valid_dataset


# ============================================================
# DATALOADERS
# ============================================================

def build_dataloaders(train_dataset, valid_dataset):

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        collate_fn=collate_fn

    )

    valid_loader = DataLoader(

        valid_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        collate_fn=collate_fn

    )

    return train_loader, valid_loader


# ============================================================
# MODEL
# ============================================================

def build_model(src_tokenizer, trg_tokenizer):

    model = TransformerNMT(

        src_vocab_size=len(src_tokenizer),

        trg_vocab_size=len(trg_tokenizer),

        d_model=D_MODEL,

        nhead=NHEAD,

        num_encoder_layers=NUM_ENCODER_LAYERS,

        num_decoder_layers=NUM_DECODER_LAYERS,

        dim_feedforward=DIM_FEEDFORWARD,

        dropout=DROPOUT

    )

    return model.to(DEVICE)

    # ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
):

    model.train()

    total_loss = 0

    for batch in dataloader:

        src = batch["src"].to(DEVICE)

        trg = batch["trg"].to(DEVICE)

        # ------------------------------------------
        # Teacher forcing
        # ------------------------------------------

        trg_input = trg[:, :-1]

        target = trg[:, 1:]

        optimizer.zero_grad()

        src_padding_mask = (src == PAD_IDX)

        trg_padding_mask = (trg_input == PAD_IDX)

        output = model(

            src,

            trg_input,

            src_padding_mask=src_padding_mask,

            trg_padding_mask=trg_padding_mask,

        )

        output = output.reshape(
            -1,
            output.shape[-1]
        )

        target = target.reshape(-1)

        loss = criterion(
            output,
            target
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            CLIP

        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    dataloader,
    criterion,
):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for batch in dataloader:

            src = batch["src"].to(DEVICE)

            trg = batch["trg"].to(DEVICE)

            trg_input = trg[:, :-1]

            target = trg[:, 1:]

            src_padding_mask = (src == PAD_IDX)

            trg_padding_mask = (trg_input == PAD_IDX)

            output = model(

                src,

                trg_input,

                src_padding_mask=src_padding_mask,

                trg_padding_mask=trg_padding_mask,

            )

            output = output.reshape(

                -1,

                output.shape[-1]

            )

            target = target.reshape(-1)

            loss = criterion(

                output,

                target

            )

            total_loss += loss.item()

    return total_loss / len(dataloader)

    # ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("IGBO → ENGLISH TRANSFORMER TRAINING")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print()

    # --------------------------------------------------------
    # TOKENIZERS
    # --------------------------------------------------------

    src_tokenizer, trg_tokenizer = load_tokenizers()

    print(f"Igbo Vocabulary    : {len(src_tokenizer)}")
    print(f"English Vocabulary : {len(trg_tokenizer)}")
    print()

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset, valid_dataset = build_datasets(
        src_tokenizer,
        trg_tokenizer
    )

    print(f"Training Samples   : {len(train_dataset)}")
    print(f"Validation Samples : {len(valid_dataset)}")
    print()

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader, valid_loader = build_dataloaders(
        train_dataset,
        valid_dataset
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = build_model(
        src_tokenizer,
        trg_tokenizer
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )

    # --------------------------------------------------------
    # LR SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=3

    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(

        ignore_index=PAD_IDX,

        label_smoothing=0.1

    )

    # --------------------------------------------------------
    # TRAINING VARIABLES
    # --------------------------------------------------------

    best_valid_loss = float("inf")

    history = []

    counter = 0

    # --------------------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------------------

    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):

        start = time.time()

        train_loss = train_epoch(

            model,

            train_loader,

            optimizer,

            criterion

        )

        valid_loss = evaluate(

            model,

            valid_loader,

            criterion

        )

        scheduler.step(valid_loss)

        elapsed = time.time() - start

        current_lr = optimizer.param_groups[0]["lr"]

        print(

            f"Epoch {epoch:02d}"

            f" | LR {current_lr:.6f}"

            f" | Train {train_loss:.4f}"

            f" | Valid {valid_loss:.4f}"

            f" | {elapsed:.2f}s"

        )

        history.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "valid_loss": valid_loss,

            "learning_rate": current_lr,

            "time": elapsed

        })

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss

            counter = 0

            torch.save(

                {

                    "epoch": epoch,

                    "model_state_dict": model.state_dict(),

                    "optimizer_state_dict": optimizer.state_dict(),

                    "valid_loss": valid_loss,

                    "train_loss": train_loss,

                    "src_vocab_size": len(src_tokenizer),

                    "trg_vocab_size": len(trg_tokenizer),

                    "d_model": D_MODEL,

                    "nhead": NHEAD,

                    "num_encoder_layers": NUM_ENCODER_LAYERS,

                    "num_decoder_layers": NUM_DECODER_LAYERS,

                    "dim_feedforward": DIM_FEEDFORWARD,

                    "dropout": DROPOUT,

                },

                CHECKPOINT_DIR / "round2_best.pt"

            )

            print("✓ Best model updated.")

        else:

            counter += 1

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if counter >= PATIENCE:

            print()

            print("Early stopping triggered.")

            break

    # --------------------------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------------------------

    history_df = pd.DataFrame(history)

    history_df.to_csv(

        LOG_DIR / "training_history.csv",

        index=False

    )

    print()

    print("=" * 60)

    print("TRAINING FINISHED")

    print("=" * 60)

    print()

    print(f"Best Validation Loss : {best_valid_loss:.4f}")

    print()

    print("Checkpoint:")

    print(CHECKPOINT_DIR / "round2_best.pt")

    print()

    print("History:")

    print(LOG_DIR / "training_history.csv")# ============================================================
# TRAINING LOOP
# ============================================================

    best_valid_loss = float("inf")
    patience = 7
    counter = 0
    history = []

    print("\nStarting training...\n")

    for epoch in range(1, NUM_EPOCHS + 1):

        start_time = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        valid_loss = evaluate(
            model,
            valid_loader,
            criterion,
        )

        scheduler.step(valid_loss)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train {train_loss:.4f} | "
            f"Valid {valid_loss:.4f} | "
            f"{elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "time": elapsed,
        })

        # -----------------------------
        # Save best model
        # -----------------------------

        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss
            counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "src_vocab_size": len(igbo_tokenizer),
                    "trg_vocab_size": len(english_tokenizer),
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "num_encoder_layers": NUM_ENCODER_LAYERS,
                    "num_decoder_layers": NUM_DECODER_LAYERS,
                    "dim_feedforward": DIM_FEEDFORWARD,
                    "dropout": DROPOUT,
                },
                CHECKPOINT_DIR / "round2_best.pt",
            )

            print("✓ Best model updated.")

        else:

            counter += 1

            print(
                f"No improvement ({counter}/{patience})"
            )

            if counter >= patience:

                print("\nEarly stopping triggered.")

                break

    # ============================================================
    # SAVE TRAINING HISTORY
    # ============================================================

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        LOG_DIR / "round2_training_history.csv",
        index=False,
    )

    # ============================================================
    # FINISHED
    # ============================================================

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"Best Validation Loss : {best_valid_loss:.4f}")

    print(
        "\nModel saved to:\n",
        CHECKPOINT_DIR / "round2_best.pt",
    )

    print(
        "\nTraining history:\n",
        LOG_DIR / "round2_training_history.csv",
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()