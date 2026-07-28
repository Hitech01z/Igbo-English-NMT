import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Adds positional information to token embeddings so the Transformer
    can understand the order of words in a sentence.
    """

    def __init__(
        self,
        d_model,
        max_len=500
    ):

        super().__init__()

        position = torch.arange(
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float
            )
            *
            (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            max_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        # Shape:
        # [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(
        self,
        x
    ):

        sequence_length = x.size(1)

        return (
            x
            +
            self.pe[
                :,
                :sequence_length,
                :
            ]
        )


class TransformerNMT(nn.Module):
    """
    Transformer Encoder-Decoder Neural Machine Translation model.

    Architecture:

        Igbo sentence
              |
              v
        Source Embedding
              |
              v
        Positional Encoding
              |
              v
        Transformer Encoder
              |
              v
          Encoder Memory
              |
              v
        Transformer Decoder
              ^
              |
        English Target Embedding
              |
              v
        Positional Encoding
              |
              v
        Linear Output Layer
              |
              v
        English Vocabulary Probabilities
    """

    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_length=100
    ):

        super().__init__()

        self.d_model = d_model

        # ========================================================
        # SOURCE EMBEDDING
        # ========================================================

        self.src_embedding = nn.Embedding(
            num_embeddings=src_vocab_size,
            embedding_dim=d_model,
            padding_idx=0
        )

        # ========================================================
        # TARGET EMBEDDING
        # ========================================================

        self.trg_embedding = nn.Embedding(
            num_embeddings=trg_vocab_size,
            embedding_dim=d_model,
            padding_idx=0
        )

        # ========================================================
        # POSITIONAL ENCODINGS
        # ========================================================

        # Separate positional encoding modules for source
        # and target sequences.

        self.src_positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_length
        )

        self.trg_positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_length
        )

        # ========================================================
        # TRANSFORMER
        # ========================================================

        self.transformer = nn.Transformer(
            d_model=d_model,

            nhead=nhead,

            num_encoder_layers=num_encoder_layers,

            num_decoder_layers=num_decoder_layers,

            dim_feedforward=dim_feedforward,

            dropout=dropout,

            batch_first=True,

            norm_first=False
        )

        # ========================================================
        # OUTPUT PROJECTION
        # ========================================================

        self.output_layer = nn.Linear(
            d_model,
            trg_vocab_size
        )

    # ============================================================
    # CAUSAL MASK
    # ============================================================

    def generate_square_subsequent_mask(
        self,
        size,
        device
    ):

        """
        Prevents the decoder from seeing future target tokens.

        Example:

            Target: I am a student

            When predicting:

            I
            I am
            I am a
            I am a student

        The decoder must only see tokens to the left.
        """

        mask = torch.triu(
            torch.ones(
                size,
                size,
                device=device
            ),
            diagonal=1
        )

        mask = mask.masked_fill(
            mask == 1,
            float("-inf")
        )

        return mask

    # ============================================================
    # FORWARD PASS
    # ============================================================

    def forward(
        self,
        src,
        trg,
        src_padding_mask=None,
        trg_padding_mask=None
    ):

        # ========================================================
        # SOURCE EMBEDDING
        # ========================================================

        src_embedded = self.src_embedding(
            src
        )

        # Scale embeddings according to the original
        # Transformer formulation.

        src_embedded = (
            src_embedded
            *
            math.sqrt(
                self.d_model
            )
        )

        # Add positional information.

        src_embedded = (
            self.src_positional_encoding(
                src_embedded
            )
        )

        # ========================================================
        # TARGET EMBEDDING
        # ========================================================

        trg_embedded = self.trg_embedding(
            trg
        )

        trg_embedded = (
            trg_embedded
            *
            math.sqrt(
                self.d_model
            )
        )

        trg_embedded = (
            self.trg_positional_encoding(
                trg_embedded
            )
        )

        # ========================================================
        # DECODER CAUSAL MASK
        # ========================================================

        trg_mask = (
            self.generate_square_subsequent_mask(
                trg.size(1),
                trg.device
            )
        )

        # ========================================================
        # TRANSFORMER ENCODER-DECODER
        # ========================================================

        output = self.transformer(

            # Encoder input
            src=src_embedded,

            # Decoder input
            tgt=trg_embedded,

            # Prevent decoder from seeing future tokens
            tgt_mask=trg_mask,

            # Ignore padding in source
            src_key_padding_mask=src_padding_mask,

            # Ignore padding in target
            tgt_key_padding_mask=trg_padding_mask,

            # Ignore padding when decoder attends to encoder
            memory_key_padding_mask=src_padding_mask
        )

        # ========================================================
        # VOCABULARY PREDICTION
        # ========================================================

        output = self.output_layer(
            output
        )

        return output