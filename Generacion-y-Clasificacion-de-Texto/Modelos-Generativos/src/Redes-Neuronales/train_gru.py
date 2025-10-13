#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gru.py
Entrenamiento de un modelo de lenguaje (GRU por defecto) para generación de canciones.
- Arquitecturas soportadas: --arch {gru, lstm, rnn}
- Niveles: --level {char, word}
- Valida con PPL, genera muestras por época, (opcional) t-SNE,
  (opcional) visualizaciones char-level (t-SNE etiquetado + heatmap de bigramas),
  y grafica PPL vs época (incremental y final).
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

# Plots
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# t-SNE
try:
    from sklearn.manifold import TSNE
    _HAS_TSNE = True
except Exception:
    _HAS_TSNE = False


# ============================
# Modelos
# ============================
class RNNAutoRegLM(nn.Module):
    """RNN vanilla."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        logits = self.fc(out)
        return logits, hidden


class LSTMLanguageModel(nn.Module):
    """LSTM (referencia)."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden


class GRULanguageModel(nn.Module):
    """GRU LM (por defecto)."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        logits = self.fc(out)
        return logits, hidden


def model_factory(arch, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
    arch = arch.lower()
    if arch == "gru":
        return GRULanguageModel(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    elif arch == "lstm":
        return LSTMLanguageModel(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    elif arch == "rnn":
        return RNNAutoRegLM(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")


# ============================
# Utilidades de datos y métrica
# ============================
def load_data(level):
    base = os.path.join("data", "processed", level)
    train_x, train_y = torch.load(os.path.join(base, "train.pt"))
    val_x, val_y = torch.load(os.path.join(base, "val.pt"))

    with open(os.path.join(base, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)["itos"]  # lista: índice -> token

    with open(os.path.join(base, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)  # info del preprocessing (seq_len, pad_idx, unk_idx, etc.)

    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)
    return train_ds, val_ds, vocab, meta


def perplexity(loss):
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


# ============================
# Generación de texto
# ============================
def build_stoi(vocab):
    return {tok: i for i, tok in enumerate(vocab)}

def decode_tokens(indices, vocab, level):
    if level == "char":
        return "".join(vocab[i] for i in indices if i < len(vocab))
    else:  # word
        return " ".join(vocab[i] for i in indices if i < len(vocab))

@torch.no_grad()
def generate_sample(model, vocab, meta, device, level="char",
                    max_new_tokens=200, temperature=1.0, seed_text=None, pad_idx=0):
    stoi = build_stoi(vocab)
    model.eval()

    # Semilla
    if seed_text is None or len(seed_text) == 0:
        # Elegimos un token inicial no-PAD
        start_idx = random.randint(1, len(vocab) - 1)
        context = [start_idx]
    else:
        # Convertimos seed_text a índices
        if level == "char":
            context = [stoi.get(ch, 1) for ch in seed_text]  # 1: OOV si aplica
        else:
            toks = seed_text.strip().split()
            context = [stoi.get(w, 1) for w in toks]  # 1: OOV

    x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    hidden = None
    for _ in range(max_new_tokens):
        logits, hidden = model(x[:, -1:].contiguous(), hidden)  # último token
        logits = logits[:, -1, :] / max(1e-6, temperature)
        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        if next_idx == pad_idx:
            # re-muestrear si cae en PAD para evitar secuencias degeneradas
            mask = torch.arange(probs.size(-1), device=device) == pad_idx
            probs = probs.masked_fill(mask, 0.0)
            next_idx = torch.multinomial(probs, num_samples=1).item()
        x = torch.cat([x, torch.tensor([[next_idx]], device=device, dtype=torch.long)], dim=1)

    return decode_tokens(x[0].tolist(), vocab, level)


# ============================
# Visualización: t-SNE + PPL vs epoch + char-viz
# ============================
def save_tsne_of_embeddings(model, vocab, out_dir, epoch, max_points=300):
    if not _HAS_TSNE:
        print("scikit-learn/TSNE no disponible: se omite t-SNE.")
        return
    try:
        emb = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                emb = module.weight.detach().cpu().numpy()
                break
        if emb is None:
            print("No se encontró capa Embedding para t-SNE.")
            return

        V = emb.shape[0]
        idx = np.arange(V)
        # evitamos incluir PAD=0 para que no sesgue
        idx = idx[idx != 0]
        if len(idx) > max_points:
            idx = np.random.choice(idx, size=max_points, replace=False)
        X = emb[idx]

        tsne = TSNE(n_components=2, learning_rate="auto", init="random",
                    perplexity=min(30, len(idx)-1), random_state=42)
        Z = tsne.fit_transform(X)

        plt.figure(figsize=(7, 6))
        plt.scatter(Z[:, 0], Z[:, 1], s=10, alpha=0.7)
        for i, vidx in enumerate(idx[:80]):  # hasta 80 anotaciones
            tok = vocab[vidx]
            if len(tok) <= 15:
                plt.text(Z[i, 0], Z[i, 1], tok, fontsize=6, alpha=0.8)
        plt.title(f"t-SNE de Embeddings (epoch {epoch})")
        plt.tight_layout()
        path = os.path.join(out_dir, f"tsne_epoch_{epoch:02d}.png")
        plt.savefig(path, dpi=160)
        plt.close()
        print(f"t-SNE guardado en {path}")
    except Exception as e:
        print(f"Falló t-SNE en epoch {epoch}: {e}")

def save_ppl_plot(log_records, out_dir, arch):
    df = pd.DataFrame(log_records)
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_ppl"], label="Train PPL")
    plt.plot(df["epoch"], df["val_ppl"], label="Val PPL")
    plt.xlabel("Época")
    plt.ylabel("Perplejidad")
    plt.title(f"PPL vs Época ({arch.upper()})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "ppl_vs_epoch.png")
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Gráfica PPL vs Época guardada en {path}")

# --- incremental para no perder figura si el job muere ---
def save_ppl_plot_latest(log_records, out_dir, arch):
    df = pd.DataFrame(log_records)
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_ppl"], label="Train PPL")
    plt.plot(df["epoch"], df["val_ppl"], label="Val PPL")
    plt.xlabel("Época")
    plt.ylabel("Perplejidad")
    plt.title(f"PPL vs Época ({arch.upper()})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "ppl_vs_epoch_latest.png")
    plt.savefig(path, dpi=160)
    plt.close()

# --- NUEVO: visualizaciones char-level ---
def plot_tsne_chars(model, vocab, out_path, pad_idx=0, unk_idx=1, perplexity=10):
    if not _HAS_TSNE:
        print("scikit-learn/TSNE no disponible: se omite t-SNE char-level.")
        return
    import numpy as np
    import torch.nn as nn

    # 1) extrae la matriz de embedding
    emb = None
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            emb = m.weight.detach().cpu().numpy()
            break
    if emb is None:
        print("No Embedding layer found para char t-SNE."); return

    V = emb.shape[0]
    idx = np.arange(V)
    idx = idx[idx != pad_idx]  # quitar PAD
    X = emb[idx]

    perplexity = max(5, min(perplexity, len(idx) - 1))
    Z = TSNE(n_components=2, learning_rate="auto", init="random",
             perplexity=perplexity, random_state=42).fit_transform(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(Z[:, 0], Z[:, 1], s=10, alpha=0.7)
    for i, vidx in enumerate(idx):
        tok = vocab[vidx]
        lab = tok
        if vidx == unk_idx: lab = "<UNK>"
        if vidx == pad_idx: lab = "<PAD>"
        if tok == " ":      lab = "<SP>"
        if tok == "\n":     lab = "<NL>"
        if len(lab) > 15:   lab = lab[:12] + "…"
        plt.text(Z[i, 0], Z[i, 1], lab, fontsize=7, alpha=0.9)
    plt.title("t-SNE de embeddings (char-level)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

@torch.no_grad()
def plot_bigram_heatmap(model, vocab, out_path, temperature=1.0, pad_idx=0, top=None):
    import numpy as np
    device = next(model.parameters()).device
    V = len(vocab)

    x = torch.arange(V, device=device).view(V, 1)  # (V, 1)
    logits, _ = model(x)                           # (V, 1, V)
    logits = logits[:, -1, :] / max(1e-6, temperature)
    probs  = torch.softmax(logits, dim=-1).cpu().numpy()  # (V, V)

    keep = np.arange(V)
    if top is not None and top > 0 and top < V:
        keep = keep[:top]
        probs = probs[keep][:, keep]

    labels = []
    for i in keep:
        t = vocab[i]
        if i == pad_idx: t = "<PAD>"
        elif t == " ":   t = "<SP>"
        elif t == "\n":  t = "<NL>"
        labels.append(t)

    plt.figure(figsize=(8, 7))
    plt.imshow(probs, aspect="auto")
    plt.colorbar(label="P(next | current)")
    plt.xticks(range(len(keep)), labels, rotation=90)
    plt.yticks(range(len(keep)), labels)
    plt.title("Matriz de prob. de bigramas inducida por el LM (char-level)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ============================
# Entrenamiento
# ============================
def train(args):
    # Semillas
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Datos
    train_ds, val_ds, vocab, meta = load_data(args.level)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pad/unk desde meta.json (fallbacks)
    pad_idx = int(meta.get("pad_idx", 0))
    unk_idx = int(meta.get("unk_idx", 1))
    unk_idx = unk_idx if 0 <= unk_idx < len(vocab) else 1

    # Modelo
    model = model_factory(
        arch=args.arch,
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # Optimizador y criterio
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Salidas
    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float("inf")
    log_records = []

    # Entrenamiento por épocas
    for epoch in range(1, args.epochs + 1):
        # ------- Train -------
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_ppl = perplexity(avg_train_loss)

        # ------- Val -------
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

        # ------- t-SNE genérico (si se pidió y NO estamos usando char-viz) -------
        if args.tsne_each_epoch and not (args.viz_chars and args.level == "char"):
            save_tsne_of_embeddings(model, vocab, args.out_dir, epoch, max_points=args.tsne_tokens)

        # ------- Muestra de generación -------
        try:
            sample = generate_sample(
                model, vocab, meta, device,
                level=args.level,
                max_new_tokens=args.sample_tokens,
                temperature=args.temperature,
                seed_text=args.seed_text,
                pad_idx=pad_idx
            )
            sample_path = os.path.join(args.out_dir, f"sample_epoch_{epoch:02d}.txt")
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write(sample)
            print(f"Muestra generada (epoch {epoch}) guardada en {sample_path}")
        except Exception as e:
            print(f"Falló la generación en epoch {epoch}: {e}")

        # ------- Guardar mejor modelo -------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab,
                "meta": meta,
                "args": {**vars(args), "arch": args.arch.lower()}
            }, save_path)
            print(f"Guardado mejor modelo en {save_path}")

        # ------- Log -------
        log_records.append({
            "epoch": epoch,
            "train_ppl": train_ppl,
            "val_ppl": val_ppl
        })

        # --- guardar figura incremental por época ---
        try:
            save_ppl_plot_latest(log_records, args.out_dir, args.arch)
        except Exception:
            pass

        # --- NUEVO: visualizaciones char-level (si aplica) ---
        if args.viz_chars and args.level == "char" and (args.viz_chars_each_epoch or epoch == args.epochs):
            try:
                out_tsne = os.path.join(args.out_dir, f"viz_tsne_epoch_{epoch:02d}.png")
                out_bi   = os.path.join(args.out_dir, f"viz_bigram_epoch_{epoch:02d}.png")
                plot_tsne_chars(model, vocab, out_tsne,
                                pad_idx=pad_idx, unk_idx=unk_idx,
                                perplexity=args.viz_tsne_perplexity)
                plot_bigram_heatmap(model, vocab, out_bi,
                                    temperature=1.0, pad_idx=pad_idx,
                                    top=(args.viz_bigram_top if args.viz_bigram_top > 0 else None))
                print(f"Char-viz guardadas en {out_tsne} y {out_bi}")
            except Exception as e:
                print(f"Char-viz fallaron en epoch {epoch}: {e}")

    # Guardar log CSV y gráfica PPL final
    log_df = pd.DataFrame(log_records)
    csv_path = os.path.join(args.out_dir, "training_log.csv")
    log_df.to_csv(csv_path, index=False)
    print(f"Log de entrenamiento guardado en {csv_path}")

    save_ppl_plot(log_records, args.out_dir, args.arch)


# ============================
# Main
# ============================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Entrenar LM (GRU/LSTM/RNN) para canciones")
    p.add_argument("--arch", type=str, choices=["gru", "lstm", "rnn"], default="gru",
                   help="Arquitectura del modelo")
    p.add_argument("--level", type=str, choices=["char", "word"], required=True)

    # Hiperparámetros
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)

    # Salidas
    p.add_argument("--out_dir", type=str, default="models/gru")

    # Reproducibilidad
    p.add_argument("--seed", type=int, default=42)

    # Generación
    p.add_argument("--sample_tokens", type=int, default=200,
                   help="N° de tokens a generar tras cada época")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed_text", type=str, default="",
                   help="Semilla de generación (texto). Vacío = aleatorio")

    # t-SNE genérico
    p.add_argument("--tsne_each_epoch", action="store_true",
                   help="Si se pasa, guardará un t-SNE de embeddings por época (genérico)")
    p.add_argument("--tsne_tokens", type=int, default=300,
                   help="Máx. vocab tokens a muestrear para t-SNE genérico")

    # --- NUEVO: visualizaciones char-level ---
    p.add_argument("--viz_chars", action="store_true",
                   help="Activa visualizaciones específicas para char-level (t-SNE etiquetado + heatmap bigramas)")
    p.add_argument("--viz_chars_each_epoch", action="store_true",
                   help="Si se pasa, guarda char-viz en cada época; si no, solo en la última")
    p.add_argument("--viz_bigram_top", type=int, default=0,
                   help="Si >0, recorta heatmap a los primeros N caracteres (por índice)")
    p.add_argument("--viz_tsne_perplexity", type=int, default=10,
                   help="Perplexity para t-SNE en char-level (5–15 recomendado)")

    args = p.parse_args()
    train(args)
