#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lstm.py
Entrenamiento de un modelo LSTM para generación de canciones.
Similar a train_rnn.py, pero usando LSTM.
"""

import argparse
import json
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

# ============================
# Modelo LSTM
# ============================
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden


# ============================
# Utilidades
# ============================
def load_data(level):
    base = os.path.join("data", "processed", level)
    train_x, train_y = torch.load(os.path.join(base, "train.pt"))
    val_x, val_y = torch.load(os.path.join(base, "val.pt"))

    with open(os.path.join(base, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)["itos"]

    with open(os.path.join(base, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)
    return train_ds, val_ds, vocab, meta


def perplexity(loss):
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


# ============================
# Entrenamiento
# ============================
def train(args):
    print("🚀 Iniciando entrenamiento LSTM...")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Carga de datos
    train_ds, val_ds, vocab, meta = load_data(args.level)
    print(f"📊 Dataset cargado: {len(train_ds)} train | {len(val_ds)} val | vocab={len(vocab)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Usando dispositivo: {device}")

    # Modelo
    model = LSTMLanguageModel(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    os.makedirs(args.out_dir, exist_ok=True)
    log_records = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        # Entrenamiento
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # estabilidad
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_ppl = perplexity(avg_train_loss)

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                xb, yb = xb.to(device), yb.to(device)
                logits, _ = model(xb)
                loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_ppl = perplexity(avg_val_loss)

        print(f"[Epoch {epoch:02d}] Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")

        # Guardar mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab,
                "meta": meta,
                "args": vars(args)
            }, save_path)
            print(f"✅ Guardado mejor modelo en {save_path}")

        # Log
        log_records.append({
            "epoch": epoch,
            "train_ppl": train_ppl,
            "val_ppl": val_ppl
        })

    # Guardar log CSV
    pd.DataFrame(log_records).to_csv(os.path.join(args.out_dir, "training_log.csv"), index=False)
    print(f"📑 Log de entrenamiento guardado en {args.out_dir}/training_log.csv")


# ============================
# Main
# ============================
if __name__ == "__main__":
    print("⚙️ Parsing argumentos...")
    p = argparse.ArgumentParser(description="Entrenar LSTM LM para canciones")
    p.add_argument("--level", type=str, choices=["char", "word"], required=True)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_dir", type=str, default="models/lstm")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train(args)

