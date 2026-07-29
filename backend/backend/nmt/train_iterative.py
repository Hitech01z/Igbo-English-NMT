from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nmt.dataset import (
    TranslationDataset,
    collate_fn
)

from nmt.dataset import (
    TranslationDataset,
    collate_fn
)
from nmt.transformer import TransformerNMT
from nmt.tokenizer import SimpleTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

TRAIN_FILE = Path(
    "processed/round1/splits/train.csv"
)

VALID_FILE = Path(
    "processed/round1/splits/valid.csv"
)

IGBO_TOKENIZER_FILE = Path(
    "processed/round1/tokenizers/igbo_tokenizer.json"
)

ENGLISH_TOKENIZER_FILE = Path(
    "processed/round1/tokenizers/english_tokenizer.json"
)

CHECKPOINT_DIR = Path(
    "checkpoints/round1"
)

BEST_MODEL_FILE = (
    CHECKPOINT_DIR
    /
    "round1_best.pt"
)


BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.0003


# ============================================================
# START
# ============================================================

print("=" * 60)
print("IGBO-ENGLISH TRANSFORMER TRAINING - ROUND 1")
print("=" * 60)

print(
    f"Device: {DEVICE}"
)


# ============================================================
# LOAD TOKENIZERS
# ============================================================

igbo_tokenizer = SimpleTokenizer.load(
    IGBO_TOKENIZER_FILE
)

english_tokenizer = SimpleTokenizer.load(
    ENGLISH_TOKENIZER_FILE
)


# ============================================================
# LOAD DATASETS
# ============================================================

train_dataset = TranslationDataset(
    TRAIN_FILE,
    igbo_tokenizer,
    english_tokenizer
)

valid_dataset = TranslationDataset(
    VALID_FILE,
    igbo_tokenizer,
    english_tokenizer
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
    f"Training samples: "
    f"{len(train_dataset)}"
)

print(
    f"Validation samples: "
    f"{len(valid_dataset)}"
)


# ============================================================
# MODEL
# ============================================================

model = TransformerNMT(

    src_vocab_size=len(
        igbo_tokenizer
    ),

    trg_vocab_size=len(
        english_tokenizer
    )

).to(DEVICE)


# ============================================================
# OPTIMIZER AND LOSS
# ============================================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)


criterion = nn.CrossEntropyLoss(

    ignore_index=
    english_tokenizer.token_to_id[
        "<pad>"
    ]

)


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

CHECKPOINT_DIR.mkdir(

    parents=True,

    exist_ok=True

)


best_valid_loss = float(
    "inf"
)


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(

    1,

    NUM_EPOCHS + 1

):

    start_time = time.time()


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    total_train_loss = 0


    for batch in train_loader:

        src = batch["src"].to(
            DEVICE
        )

        trg = batch["trg"].to(
            DEVICE
        )


        optimizer.zero_grad()


        output = model(

            src,

            trg[:, :-1]

        )


        output_dim = output.size(
            -1
        )


        output = output.reshape(

            -1,

            output_dim

        )


        target = trg[:, 1:].reshape(

            -1

        )


        loss = criterion(

            output,

            target

        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            1.0

        )


        optimizer.step()


        total_train_loss += (

            loss.item()

        )


    train_loss = (

        total_train_loss
        /
        len(train_loader)

    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    total_valid_loss = 0


    with torch.no_grad():

        for batch in valid_loader:

            src = batch["src"].to(
                DEVICE
            )

            trg = batch["trg"].to(
                DEVICE
            )


            output = model(

                src,

                trg[:, :-1]

            )


            output_dim = output.size(
                -1
            )


            output = output.reshape(

                -1,

                output_dim

            )


            target = trg[:, 1:].reshape(

                -1

            )


            loss = criterion(

                output,

                target

            )


            total_valid_loss += (

                loss.item()

            )


    valid_loss = (

        total_valid_loss
        /
        len(valid_loader)

    )


    elapsed = time.time() - start_time


    print(

        f"Epoch {epoch:02d} | "

        f"Time: {elapsed:.2f}s | "

        f"Train Loss: {train_loss:.4f} | "

        f"Valid Loss: {valid_loss:.4f}"

    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss


        torch.save(

            model.state_dict(),

            BEST_MODEL_FILE

        )


        print(
            "  ✓ Saved best model"
        )


# ============================================================
# COMPLETE
# ============================================================

print()

print("=" * 60)

print(
    "ROUND 1 TRAINING COMPLETE"
)

print("=" * 60)

print(

    f"Best validation loss: "
    f"{best_valid_loss:.4f}"

)

print(

    f"Saved to: "
    f"{BEST_MODEL_FILE}"

)