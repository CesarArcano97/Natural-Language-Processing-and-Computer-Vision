#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate.py
Genera texto/canciones a partir de un modelo entrenado (RNN, LSTM o GRU).
Uso:
python src/generate.py --model models/lstm_baseline/best_model.pt --prompt "In Dema theres no choice, but in Trench I'm not afraid"
"""

import argparse
import torch
import torch.nn as nn
import json
import numpy as np

# ============================
# Modelos
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


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.3):
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


class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        logits = self.fc(out)
        return logits, hidden


# ============================
# Generación
# ============================
def generate(model, vocab, device, prompt, max_len=100,
             temperature=1.0, top_k=0, top_p=1.0):
    stoi = {s: i for i, s in enumerate(vocab)}
    itos = vocab

    # Tokenizar prompt
    tokens = prompt.split()
    input_ids = [stoi.get(t, stoi["<UNK>"]) for t in tokens]
    input_tensor = torch.tensor([input_ids], device=device)

    generated_ids = input_ids[:]  # arrancamos con el prompt
    hidden = None

    with torch.no_grad():
        for _ in range(max_len):
            logits, hidden = model(input_tensor, hidden)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            probs = torch.softmax(logits, dim=-1)

            # --- Top-k ---
            if top_k > 0:
                top_k_vals, top_k_idx = torch.topk(probs, top_k)
                probs_filtered = torch.zeros_like(probs).scatter_(1, top_k_idx, top_k_vals)
                probs = probs_filtered / probs_filtered.sum()

            # --- Top-p (nucleus) ---
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumulative_probs > top_p
                sorted_probs[cutoff] = 0
                sorted_probs /= sorted_probs.sum()
                probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)

            next_token_id = torch.multinomial(probs, num_samples=1).item()
            generated_ids.append(next_token_id)
            input_tensor = torch.tensor([[next_token_id]], device=device)

            if itos[next_token_id] == "<EOS>":
                break

    return " ".join([itos[idx] for idx in generated_ids])


# ============================
# Main
# ============================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generar texto/canciones desde un modelo entrenado")
    p.add_argument("--model", type=str, required=True, help="Ruta al modelo entrenado (.pt)")
    p.add_argument("--prompt", type=str, default="<BOS>", help="Texto inicial para la generación")
    p.add_argument("--max_len", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    vocab = checkpoint["vocab"]
    model_args = checkpoint["args"]

    arch = model_args.get("arch", "rnn")  # fallback = rnn
    if arch == "rnn":
        model = RNNLanguageModel(
            vocab_size=len(vocab),
            embed_dim=model_args["embed_dim"],
            hidden_dim=model_args["hidden_dim"],
            num_layers=model_args["num_layers"],
            dropout=model_args["dropout"]
        )
    elif arch == "lstm":
        model = LSTMLanguageModel(
            vocab_size=len(vocab),
            embed_dim=model_args["embed_dim"],
            hidden_dim=model_args["hidden_dim"],
            num_layers=model_args["num_layers"],
            dropout=model_args["dropout"]
        )
    elif arch == "gru":
        model = GRULanguageModel(
            vocab_size=len(vocab),
            embed_dim=model_args["embed_dim"],
            hidden_dim=model_args["hidden_dim"],
            num_layers=model_args["num_layers"],
            dropout=model_args["dropout"]
        )
    else:
        raise ValueError(f"Arquitectura desconocida en checkpoint: {arch}")

    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    # Generar texto
    output = generate(
        model, vocab, device,
        prompt=args.prompt,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    print("\n=== 🎶 Canción generada ===")
    print(output)
    print("===========================\n")


