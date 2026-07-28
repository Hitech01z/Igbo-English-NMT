import pandas as pd
import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TranslationDataset(Dataset):
    """
    Generic Translation Dataset

    Supports both:

        Igbo -> English

    and

        English -> Igbo

    simply by changing the source and target columns.
    """

    def __init__(
        self,
        csv_file,
        src_tokenizer,
        trg_tokenizer,
        src_column="igbo",
        trg_column="english"
    ):

        self.data = pd.read_csv(csv_file)

        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

        self.src_column = src_column
        self.trg_column = trg_column

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        src_sentence = str(
            row[self.src_column]
        )

        trg_sentence = str(
            row[self.trg_column]
        )

        src_ids = self.src_tokenizer.encode(
            src_sentence
        )

        trg_ids = self.trg_tokenizer.encode(
            trg_sentence
        )

        return {

            "src": torch.tensor(
                src_ids,
                dtype=torch.long
            ),

            "trg": torch.tensor(
                trg_ids,
                dtype=torch.long
            )

        }


def collate_fn(batch):

    src_batch = [
        item["src"]
        for item in batch
    ]

    trg_batch = [
        item["trg"]
        for item in batch
    ]

    src_batch = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=0
    )

    trg_batch = pad_sequence(
        trg_batch,
        batch_first=True,
        padding_value=0
    )

    return {

        "src": src_batch,

        "trg": trg_batch

    }