#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problema07_Tarea01_NLP.py

Entrenamiento de Word2Vec sobre un corpus y generación de analogías semánticas.

Uso:
  python Problema07_Tarea01_NLP.py --csv "/home/cesar/Descargas/MeIA.csv" \
      --text-col Review --remove-stop --save-dir out_w2v --plots
"""

import os
import argparse
import pandas as pd
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -----------------------------
# Funciones auxiliares
# -----------------------------
def load_and_clean(csv_path, text_col, remove_stop):
    df = pd.read_csv(csv_path, encoding="latin-1")
    df = df.dropna(subset=[text_col])
    texts = df[text_col].astype(str).tolist()

    # Tokenización básica con gensim (rápida)
    tokenized = [simple_preprocess(t, deacc=True) for t in texts]

    if remove_stop:
        tokenized = [[tok for tok in doc if tok not in STOP_WORDS] for doc in tokenized]

    return tokenized

def train_word2vec(sentences, vector_size=100, window=5, min_count=5, sg=1, workers=4):
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=workers
    )
    return model

def plot_embeddings(model, save_dir, top_n=100):
    words = list(model.wv.key_to_index.keys())[:top_n]
    X = model.wv[words]

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, (coords[i, 0], coords[i, 1]))
    plt.title("Proyección PCA de embeddings Word2Vec")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "word2vec_pca.png"))
    plt.close()

def run_analogies(model, save_dir):
    results = []
    examples = [
    (["comida", "deliciosa"], ["horrible"]),      
    (["precio", "barato"], ["caro"]),        
    (["hotel", "caro"], ["barato"]),     
    (["servicio", "excelente"], ["malo"]), 
    # analogías geográficas / contextuales
    (["tulum", "playa"], ["ciudad"]),        # Tulum es a playa como X es a ciudad
    (["queretaro", "moderno"], ["antiguo"]), # Monterrey moderno vs lugar antiguo ≈ ?
    (["cancun", "caro"], ["barato"]),        # Cancún caro ≈ lugar barato
]
    for pos, neg in examples:
        try:
            res = model.wv.most_similar(positive=pos, negative=neg, topn=5)
            results.append((pos, neg, res))
        except KeyError as e:
            results.append((pos, neg, f"Error: {e}"))

    # Guardar resultados
    with open(os.path.join(save_dir, "analogies.txt"), "w", encoding="utf-8") as f:
        for pos, neg, res in results:
            f.write(f"Analogy: {pos} - {neg}\n")
            f.write(f"Result: {res}\n\n")
    print(f"[OK] Analogías guardadas en {os.path.join(save_dir, 'analogies.txt')}")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Ruta al dataset CSV")
    parser.add_argument("--text-col", type=str, required=True, help="Columna de texto")
    parser.add_argument("--remove-stop", action="store_true", help="Eliminar stopwords")
    parser.add_argument("--save-dir", type=str, required=True, help="Directorio de salida")
    parser.add_argument("--plots", action="store_true", help="Generar gráficos PCA")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("[INFO] Cargando y limpiando dataset...")
    sentences = load_and_clean(args.csv, args.text_col, args.remove_stop)

    print("[INFO] Entrenando modelo Word2Vec...")
    model = train_word2vec(sentences)

    print("[INFO] Guardando modelo...")
    model.save(os.path.join(args.save_dir, "word2vec.model"))
    model.wv.save(os.path.join(args.save_dir, "word2vec.kv"))

    print("[INFO] Ejecutando analogías...")
    run_analogies(model, args.save_dir)

    if args.plots:
        print("[INFO] Generando gráfico PCA de embeddings...")
        plot_embeddings(model, args.save_dir)

    print("[OK] Pipeline completado.")

if __name__ == "__main__":
    main()
