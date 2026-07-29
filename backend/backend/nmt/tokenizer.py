import json
import re
from collections import Counter
from pathlib import Path


class SimpleTokenizer:
    """
    Lightweight tokenizer for the Igbo-English NMT system.
    """

    SPECIAL_TOKENS = [
        "<pad>",
        "<unk>",
        "<sos>",
        "<eos>",
    ]

    def __init__(self, min_freq=1):

        self.min_freq = min_freq

        self.token_to_id = {}
        self.id_to_token = {}

        self._initialize_special_tokens()

    def _initialize_special_tokens(self):

        for idx, token in enumerate(self.SPECIAL_TOKENS):

            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    # =====================================================
    # TOKENIZATION
    # =====================================================

    def tokenize(self, text):

        text = str(text).lower().strip()

        # Separate punctuation
        text = re.sub(
            r"([.,!?;:()\"'])",
            r" \1 ",
            text,
        )

        # Separate hyphenated words
        text = re.sub(
            r"([-])",
            r" \1 ",
            text,
        )

        # Remove duplicate spaces
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.split()

    # =====================================================
    # BUILD VOCABULARY
    # =====================================================

    def build_vocab(self, sentences):

        counter = Counter()

        for sentence in sentences:

            counter.update(
                self.tokenize(sentence)
            )

        next_id = len(self.token_to_id)

        for token, frequency in counter.most_common():

            if frequency < self.min_freq:
                continue

            if token in self.token_to_id:
                continue

            self.token_to_id[token] = next_id
            self.id_to_token[next_id] = token

            next_id += 1

    # =====================================================
    # ENCODE
    # =====================================================

    def encode(self, text, add_special_tokens=True):

        ids = []

        if add_special_tokens:
            ids.append(self.sos_token_id)

        for token in self.tokenize(text):

            ids.append(
                self.token_to_id.get(
                    token,
                    self.unk_token_id,
                )
            )

        if add_special_tokens:
            ids.append(self.eos_token_id)

        return ids

    # =====================================================
    # DECODE
    # =====================================================

    def decode(
        self,
        ids,
        remove_special_tokens=True,
    ):

        tokens = []

        for idx in ids:

            token = self.id_to_token.get(
                int(idx),
                "<unk>",
            )

            if (
                remove_special_tokens
                and token in self.SPECIAL_TOKENS
            ):
                continue

            tokens.append(token)

        text = " ".join(tokens)

        text = re.sub(
            r"\s+([.,!?;:])",
            r"\1",
            text,
        )

        text = re.sub(
            r"\(\s+",
            "(",
            text,
        )

        text = re.sub(
            r"\s+\)",
            ")",
            text,
        )

        return text

    # =====================================================
    # SAVE
    # =====================================================

    def save(self, path):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {

            "min_freq": self.min_freq,

            "token_to_id": self.token_to_id,

            "id_to_token": {

                str(k): v

                for k, v in self.id_to_token.items()

            }

        }

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

    # =====================================================
    # LOAD
    # =====================================================

    @classmethod
    def load(cls, path):

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        tokenizer = cls(
            min_freq=data["min_freq"]
        )

        tokenizer.token_to_id = data["token_to_id"]

        tokenizer.id_to_token = {

            int(k): v

            for k, v in data["id_to_token"].items()

        }

        return tokenizer

    # =====================================================
    # SPECIAL TOKEN IDS
    # =====================================================

    @property
    def pad_token_id(self):
        return self.token_to_id["<pad>"]

    @property
    def unk_token_id(self):
        return self.token_to_id["<unk>"]

    @property
    def sos_token_id(self):
        return self.token_to_id["<sos>"]

    @property
    def eos_token_id(self):
        return self.token_to_id["<eos>"]

    # =====================================================
    # LENGTH
    # =====================================================

    def __len__(self):
        return len(self.token_to_id)