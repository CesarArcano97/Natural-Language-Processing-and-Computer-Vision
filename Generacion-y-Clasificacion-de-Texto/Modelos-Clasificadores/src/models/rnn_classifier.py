# -*- coding: utf-8 -*-
# src/models/rnn_classifier.py
from typing import Literal, Optional
import torch
import torch.nn as nn

PoolType = Literal["max", "mean", "maxmean", "last"]
CellType = Literal["rnn", "lstm", "gru"]

class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 1,
        bidirectional: bool = True,
        rnn_type: CellType = "lstm",
        pad_idx: int = 0,
        emb_dropout: float = 0.2,
        rnn_dropout: float = 0.2,
        proj_dropout: float = 0.5,
        pool: PoolType = "max",
    ):
        super().__init__()
        self.pool: PoolType = pool
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Recurrente: solo RNN clásico lleva "nonlinearity"
        rnn_kwargs = dict(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=rnn_dropout if num_layers > 1 else 0.0,
        )
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(**rnn_kwargs, nonlinearity="tanh")
        else:
            raise ValueError(f"rnn_type no soportado: {rnn_type}")

        feat_dim = hidden_size * (2 if bidirectional else 1)
        if self.pool == "maxmean":
            feat_dim *= 2  # concat(max, mean)

        self.proj = nn.Sequential(
            nn.Dropout(proj_dropout),
            nn.Linear(feat_dim, num_classes),
        )

    @staticmethod
    def _masked_mean(reps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # reps: [B,T,H], mask: [B,T] bool
        lens = mask.sum(1).clamp(min=1).unsqueeze(1)  # [B,1]
        return (reps * mask.unsqueeze(-1)).sum(1) / lens

    @staticmethod
    def _masked_max(reps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # reps: [B,T,H], mask: [B,T] bool
        # Rellenar con -1e9 (en lugar de -inf) para evitar NaNs en casos límite
        fill = torch.full_like(reps, -1e9)
        reps_masked = torch.where(mask.unsqueeze(-1), reps, fill)
        vals = reps_masked.max(dim=1).values  # [B,H]
        # Si alguna fila quedó totalmente enmascarada (raro), devolver ceros
        empty_rows = (mask.sum(1) == 0)
        if empty_rows.any():
            vals[empty_rows] = 0.0
        return vals

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids: [B,T], attention_mask: [B,T] (0/1)
        x = self.embedding(input_ids)          # [B,T,E]
        x = self.emb_dropout(x)

        out, _ = self.rnn(x)                   # out: [B,T,H*dir]
        mask = attention_mask.bool()           # asegurar bool

        if self.pool == "max":
            feats = self._masked_max(out, mask)
        elif self.pool == "mean":
            feats = self._masked_mean(out, mask)
        elif self.pool == "maxmean":
            feats = torch.cat([self._masked_max(out, mask),
                               self._masked_mean(out, mask)], dim=1)
        elif self.pool == "last":
            # Úsalo solo si garantizas que el último token no es pad
            feats = out[:, -1, :]
        else:
            raise ValueError(f"pool no soportado: {self.pool}")

        logits = self.proj(feats)              # [B,C]
        return logits


