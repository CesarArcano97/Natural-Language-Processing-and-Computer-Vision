#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_rnn.py
Entrenamiento de un modelo RNN para generación de canciones.
Carga datos preparados por prepare_corpus.py y guarda el modelo entrenado.
Incluye:
- Visualización de entrenamiento (ppl vs epoch)
- Visualización de embeddings y estados ocultos (con manejo robusto de t-SNE)
- Muestras generadas durante el entrenamiento
- Log en CSV para análisis posterior
"""

import argparse
import json
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import random
import warnings
from tqdm import tqdm   # 🔹 barra de progreso

# ============================
# Utilidades
# ============================
def safe_tsne_perplexity(n_samples: int, default: int = 30) -> int:
    if n_samples < 3:
        return None
    p = min(default, n_samples - 1)
    p = max(2, p)
    return p


# ============================
# Modelo
# ============================
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        logits = self.fc(out)
        return logits, hidden


# ============================
# Funciones auxiliares
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
# Visualizaciones
# ============================
def plot_history(train_history, val_history, out_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label="Train PPL")
    plt.plot(val_history, label="Validation PPL")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Training & Validation Perplexity")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(out_dir, "training_history.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Gráfica del historial guardada en {save_path}")


def plot_embeddings(model, vocab, epoch, out_dir):
    V = len(vocab)
    perp = safe_tsne_perplexity(V, default=30)
    if perp is None:
        warnings.warn(f"[plot_embeddings] Vocab demasiado pequeño (V={V}). Se omite t-SNE.")
        return
    try:
        embeddings = model.embedding.weight.data.detach().cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
        embeddings_2d = tsne.fit_transform(embeddings)
        plt.figure(figsize=(12, 12))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
        num_labels = min(200, V)
        for i, word in enumerate(vocab[:num_labels]):
            plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
        plt.title(f"t-SNE of Embeddings (Epoch {epoch}) | perp={perp}")
        save_path = os.path.join(out_dir, f"embeddings_epoch_{epoch}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"🎨 Visualización de embeddings guardada en {save_path}")
    except Exception as e:
        warnings.warn(f"[plot_embeddings] Falló t-SNE (perp={perp}, V={V}). Se omite. Detalle: {e}")


def plot_hidden_states(model, vocab, device, epoch, out_dir):
    model.eval()
    stoi = {s: i for i, s in enumerate(vocab)}

    sample_sentences = [
        "the sun will rise and we will try again",
        "there is hope for tomorrow",
        "shadows will scream that i'm alone",
        "my hometown's in the dark",
        "i am on the run and go",
        "i'm wanted and on the run",
        "we paint the town",
        "welcome to the new way of living"
    ]

    n_samples = len(sample_sentences)
    perp = safe_tsne_perplexity(n_samples, default=5)
    if perp is None:
        warnings.warn(f"[plot_hidden_states] Muy pocas frases (n={n_samples}). Se omite t-SNE.")
        return

    final_hidden_states = []
    try:
        with torch.no_grad():
            for sentence in sample_sentences:
                tokens = sentence.split()
                ids = [stoi.get(t, stoi.get("<UNK>", 1)) for t in tokens]
                input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
                _, hidden = model(input_tensor)
                final_hidden = hidden[-1, 0, :].detach().cpu().numpy()
                final_hidden_states.append(final_hidden)

        tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
        states_2d = tsne.fit_transform(final_hidden_states)

        plt.figure(figsize=(14, 10))
        plt.scatter(states_2d[:, 0], states_2d[:, 1])
        for i, sentence in enumerate(sample_sentences):
            plt.annotate(sentence, (states_2d[i, 0], states_2d[i, 1]))
        plt.title(f"t-SNE of Sentence Hidden States (Epoch {epoch}) | perp={perp}")
        save_path = os.path.join(out_dir, f"hidden_states_epoch_{epoch}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"🧠 Visualización de estados ocultos guardada en {save_path}")
    except Exception as e:
        warnings.warn(f"[plot_hidden_states] Falló t-SNE (perp={perp}, n={n_samples}). Se omite. Detalle: {e}")


# ============================
# Generación
# ============================
def generate_sample(model, vocab, device, start_token="<BOS>",
                    max_len=50, temperature=1.0, top_k=0, top_p=1.0):
    model.eval()
    stoi = {s: i for i, s in enumerate(vocab)}
    itos = vocab
    start_id = stoi.get(start_token, stoi.get("<UNK>", 1))
    input_ids = torch.tensor([[start_id]], dtype=torch.long, device=device)
    generated_ids = [start_id]

    hidden = None
    with torch.no_grad():
        for _ in range(max_len - 1):
            logits, hidden = model(input_ids, hidden)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            probs = torch.softmax(logits, dim=-1)

            if top_k and top_k > 0:
                top_k_vals, top_k_idx = torch.topk(probs, min(top_k, probs.size(-1)))
                probs_filtered = torch.zeros_like(probs).scatter_(1, top_k_idx, top_k_vals)
                probs = probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)

            if top_p and top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs > top_p
                sorted_probs[mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)

            next_token_id = torch.multinomial(probs, num_samples=1).item()
            generated_ids.append(next_token_id)
            input_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

            if itos[next_token_id] == "<EOS>":
                break

    return " ".join([itos[idx] for idx in generated_ids])


# ============================
# Entrenamiento
# ============================
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_ds, val_ds, vocab, meta = load_data(args.level)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RNNLanguageModel(
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
    train_ppl_history, val_ppl_history = [], []
    log_records = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        # 🔹 barra de progreso en entrenamiento
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))
        train_ppl = perplexity(avg_train_loss)
        train_ppl_history.append(train_ppl)

        # 🔹 barra de progreso en validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                logits, _ = model(xb)
                loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_ppl = perplexity(avg_val_loss)
        val_ppl_history.append(val_ppl)

        print(f"[Epoch {epoch:02d}] Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")

        sample = generate_sample(
            model, vocab, device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print(f"📝 Muestra: {sample}\n" + "-"*50)

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

            plot_embeddings(model, vocab, epoch, args.out_dir)
            plot_hidden_states(model, vocab, device, epoch, args.out_dir)

        log_records.append({
            "epoch": epoch,
            "train_ppl": train_ppl,
            "val_ppl": val_ppl
        })

    plot_history(train_ppl_history, val_ppl_history, args.out_dir)
    pd.DataFrame(log_records).to_csv(os.path.join(args.out_dir, "training_log.csv"), index=False)
    print(f"📑 Log de entrenamiento guardado en {args.out_dir}/training_log.csv")


# ============================
# Main
# ============================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Entrenar RNN LM para canciones")
    p.add_argument("--level", type=str, choices=["char", "word"], required=True)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_dir", type=str, default="models/rnn")
    p.add_argument("--seed", type=int, default=314353311)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    args = p.parse_args()

    train(args)


