# -*- coding: utf-8 -*-
"""
dataset.py
Carga los .pt creados por prepare_meia.py y arma DataLoaders.
Cada .pt es una lista de dicts con: input_ids, attention_mask, label (tensores).
"""

from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader

class TensorListDataset(Dataset):
    def __init__(self, items: List[Dict[str, torch.Tensor]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]

def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Los tensores ya están paddeados a la misma longitud.
    input_ids      = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels         = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def build_loaders(train_pt: str, val_pt: str, batch_size: int, num_workers: int = 2):
    train_items = torch.load(train_pt)
    val_items   = torch.load(val_pt)

    dset_train = TensorListDataset(train_items)
    dset_val   = TensorListDataset(val_items)

    train_loader = DataLoader(
        dset_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=_collate
    )
    val_loader = DataLoader(
        dset_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=_collate
    )
    return train_loader, val_loader
