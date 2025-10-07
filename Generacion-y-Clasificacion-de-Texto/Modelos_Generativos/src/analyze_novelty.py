#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_novelty.py
Analiza memorization vs. novelty en modelos (RNN, LSTM, GRU) a nivel palabra.
Calcula porcentaje de n-gramas (bi y tri) de las generaciones que ya existen en el corpus.
"""

import os
import argparse
from collections import Counter
from nltk.util import ngrams
import matplotlib.pyplot as plt
import pandas as pd

# ============================
# Funciones
# ============================
def extract_ngrams(texts, n=2):
    all_ngrams = []
    for t in texts:
        tokens = t.strip().split()
        all_ngrams.extend(list(ngrams(tokens, n)))
    return set(all_ngrams)

def novelty_score(sample, train_ngrams, n=2):
    tokens = sample.strip().split()
    sample_ngrams = list(ngrams(tokens, n))
    if len(sample_ngrams) == 0:
        return 0.0, 0.0
    seen = sum(1 for ng in sample_ngrams if ng in train_ngrams)
    memorization = seen / len(sample_ngrams)
    novelty = 1 - memorization
    return memorization, novelty

# ============================
# Main
# ============================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analizar memorization vs novelty en generaciones")
    p.add_argument("--corpus", type=str, default="data/raw/canciones.txt",
                   help="Ruta al corpus de entrenamiento")
    p.add_argument("--outputs", nargs="+", required=True,
                   help="Archivos .txt con generaciones (ej: sample_gru.txt sample_rnn.txt)")
    p.add_argument("--out_dir", type=str, default="results/novelty",
                   help="Directorio donde guardar CSV y gráficas")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Leer corpus
    with open(args.corpus, encoding="utf-8") as f:
        raw_text = f.read()
    # Ajustar delimitador si usaste otro
    songs = raw_text.split("<|startsong|>")

    # Construir n-gram sets
    bigrams_train = extract_ngrams(songs, n=2)
    trigrams_train = extract_ngrams(songs, n=3)

    # Procesar muestras
    records = []
    for path in args.outputs:
        with open(path, encoding="utf-8") as f:
            sample = f.read()
        name = os.path.splitext(os.path.basename(path))[0]

        mem2, nov2 = novelty_score(sample, bigrams_train, n=2)
        mem3, nov3 = novelty_score(sample, trigrams_train, n=3)

        records.append({
            "model": name,
            "bigram_mem": mem2, "bigram_nov": nov2,
            "trigram_mem": mem3, "trigram_nov": nov3
        })
        print(f"[{name}] Bi-Mem={mem2:.2%}, Bi-Nov={nov2:.2%} | Tri-Mem={mem3:.2%}, Tri-Nov={nov3:.2%}")

    # Guardar CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join(args.out_dir, "novelty_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"Resultados guardados en {csv_path}")

    # Graficar
    plt.figure(figsize=(8, 5))
    x = range(len(records))
    plt.bar([i - 0.2 for i in x], [r["bigram_nov"] for r in records],
            width=0.4, label="Novelty (Bigrams)")
    plt.bar([i + 0.2 for i in x], [r["trigram_nov"] for r in records],
            width=0.4, label="Novelty (Trigrams)")
    plt.xticks(x, [r["model"] for r in records])
    plt.ylabel("Proporción de novedad")
    plt.title("Memorization vs. Novelty en generaciones")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "novelty_plot.png"), dpi=160)
    plt.close()
    print(f"Gráfica guardada en {args.out_dir}/novelty_plot.png")
