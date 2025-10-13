# -*- coding: utf-8 -*-
"""
textcnn.py
TextCNN (Kim, 2014) con múltiples anchos de kernel + global max-pooling.
"""

from typing import Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialDropout1D(nn.Dropout2d):
    """
    Dropout sobre el eje de canales (como en NLP).
    Entrada esperada: (B, L, d). Convierte a (B, d, L) para aplicar 2D-dropout por canal.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).unsqueeze(3)   # (B, d, L, 1)
        x = super().forward(x)
        return x.squeeze(3).permute(0, 2, 1)  # (B, L, d)

class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        pad_idx: int,
        kernel_sizes: Sequence[int] = (2, 3, 4, 5),
        num_filters: int = 128,
        emb_dropout: float = 0.2,
        proj_dropout: float = 0.5,
        use_batchnorm: bool = False,
        embedding_weights: Optional[torch.Tensor] = None,   # preentrenadas (opcional)
        freeze_embeddings: bool = False
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(embedding_weights)
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.spatial_drop = SpatialDropout1D(p=emb_dropout) if emb_dropout > 0 else nn.Identity()

        # Conv1d espera (B, C_in, L). Usaremos C_in = embed_dim
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in kernel_sizes]) if use_batchnorm else None

        self.proj_dropout = nn.Dropout(p=proj_dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

        # Inicialización recomendada para estabilidad
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            nn.init.constant_(conv.bias, 0.0)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        input_ids: (B, L), attention_mask se ignora salvo para futuras extensiones.
        """
        x = self.embedding(input_ids)          # (B, L, d)
        x = self.spatial_drop(x)
        x = x.transpose(1, 2)                  # (B, d, L) para Conv1d

        feats = []
        for i, conv in enumerate(self.convs):
            h = conv(x)                        # (B, num_filters, L-k+1)
            h = F.relu(h)
            if self.bns is not None:
                h = self.bns[i](h)
            h = F.max_pool1d(h, kernel_size=h.shape[-1]).squeeze(-1)  # (B, num_filters)
            feats.append(h)

        z = torch.cat(feats, dim=1)            # (B, num_filters * n_kernels)
        z = self.proj_dropout(z)
        logits = self.classifier(z)            # (B, C)
        return logits
