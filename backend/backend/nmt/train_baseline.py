import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(BASE_DIR)
)


from nmt.dataset import (
    TranslationDataset,
    collate_fn
)

from nmt.tokenizer import SimpleTokenizer

from nmt.transformer import TransformerNMT


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = (
    BASE_DIR
    / "processed"
    / "baseline"
    / "train.csv"
)

VALID_FILE = (
    BASE_DIR
    / "processed"
    / "baseline"
    / "valid.csv"
)

TOKENIZER_DIR = (
    BASE_DIR
    / "processed"
    / "baseline"
    / "tokenizers"
)

CHECKPOINT_DIR = (
    BASE_DIR
    / "checkpoints"
    / "baseline"
)

LOG_DIR = (
    BASE_DIR
    / "logs"
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32

EPOCHS = 40

LEARNING_RATE = 0.0003

D_MODEL = 256

NHEAD = 8

NUM_ENCODER_LAYERS = 3

NUM_DECODER_LAYERS = 3

DIM_FEEDFORWARD = 512

DROPOUT = 0.1


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MASK
# ============================================================

def create_padding_mask(
    sequence,
    pad_idx=0
):

    return sequence == pad_idx


# ============================================================
# TRAINING
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    total_loss = 0

    for batch in loader:

        src = batch["src"].to(device)

        trg = batch["trg"].to(device)

        optimizer.zero_grad()

        trg_input = trg[:, :-1]

        trg_output = trg[:, 1:]

        src_padding_mask = (
            create_padding_mask(src)
        )

        trg_padding_mask = (
            create_padding_mask(trg_input)
        )

        output = model(
            src,
            trg_input,
            src_padding_mask=src_padding_mask,
            trg_padding_mask=trg_padding_mask
        )

        output_dim = output.shape[-1]

        output = output.contiguous().view(
            -1,
            output_dim
        )

        trg_output = trg_output.contiguous().view(
            -1
        )

        loss = criterion(
            output,
            trg_output
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()


    return (
        total_loss
        /
        len(loader)
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for batch in loader:

            src = batch["src"].to(device)

            trg = batch["trg"].to(device)

            trg_input = trg[:, :-1]

            trg_output = trg[:, 1:]

            src_padding_mask = (
                create_padding_mask(src)
            )

            trg_padding_mask = (
                create_padding_mask(trg_input)
            )

            output = model(
                src,
                trg_input,
                src_padding_mask=src_padding_mask,
                trg_padding_mask=trg_padding_mask
            )

            output_dim = output.shape[-1]

            output = output.contiguous().view(
                -1,
                output_dim
            )

            trg_output = trg_output.contiguous().view(
                -1
            )

            loss = criterion(
                output,
                trg_output
            )

            total_loss += loss.item()


    return (
        total_loss
        /
        len(loader)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "IGBO-ENGLISH TRANSFORMER BASELINE TRAINING"
    )

    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )


    # --------------------------------------------------------
    # TOKENIZERS
    # --------------------------------------------------------

    igbo_tokenizer = (
        SimpleTokenizer.load(
            TOKENIZER_DIR
            / "igbo_tokenizer.json"
        )
    )


    english_tokenizer = (
        SimpleTokenizer.load(
            TOKENIZER_DIR
            / "english_tokenizer.json"
        )
    )


    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = (
        TranslationDataset(
            TRAIN_FILE,
            igbo_tokenizer,
            english_tokenizer
        )
    )


    valid_dataset = (
        TranslationDataset(
            VALID_FILE,
            igbo_tokenizer,
            english_tokenizer
        )
    )


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


    print(
        f"Training samples: {len(train_dataset)}"
    )


    print(
        f"Validation samples: {len(valid_dataset)}"
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = TransformerNMT(

        src_vocab_size=len(
            igbo_tokenizer
        ),

        trg_vocab_size=len(
            english_tokenizer
        ),

        d_model=D_MODEL,

        nhead=NHEAD,

        num_encoder_layers=(
            NUM_ENCODER_LAYERS
        ),

        num_decoder_layers=(
            NUM_DECODER_LAYERS
        ),

        dim_feedforward=(
            DIM_FEEDFORWARD
        ),

        dropout=DROPOUT

    )


    model = model.to(
        DEVICE
    )


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    best_valid_loss = float(
        "inf"
    )


    history = []


    for epoch in range(
        1,
        EPOCHS + 1
    ):

        start_time = time.time()


        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE
        )


        valid_loss = evaluate(
            model,
            valid_loader,
            criterion,
            DEVICE
        )


        elapsed = (
            time.time()
            -
            start_time
        )


        print(
            f"Epoch {epoch:02d} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Valid Loss: {valid_loss:.4f}"
        )


        history.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "valid_loss": valid_loss

        })


        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss


            checkpoint_path = (
                CHECKPOINT_DIR
                / "baseline_best.pt"
            )


            torch.save({

                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "valid_loss":
                    valid_loss,

                "src_vocab_size":
                    len(igbo_tokenizer),

                "trg_vocab_size":
                    len(english_tokenizer)

            }, checkpoint_path)


            print(
                "  ✓ Saved best model"
            )


    history_df = pd.DataFrame(
        history
    )


    history_df.to_csv(

        LOG_DIR
        / "baseline_training_history.csv",

        index=False

    )


    print()

    print("=" * 60)

    print(
        "BASELINE TRAINING COMPLETE"
    )

    print("=" * 60)

    print(
        f"Best validation loss: "
        f"{best_valid_loss:.4f}"
    )


    print()

    print(
        "Saved checkpoint:"
    )

    print(
        CHECKPOINT_DIR
        / "baseline_best.pt"
    )


if __name__ == "__main__":

    main()