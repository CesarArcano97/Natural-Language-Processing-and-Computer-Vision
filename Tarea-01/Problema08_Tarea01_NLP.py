#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problema08_Tarea01_NLP.py

Embeddings de documento (promedio de Word2Vec) + K-means y visualización (PCA/t-SNE).

Uso:

- Para Polarity:
  python Problema08_Tarea01_NLP.py --csv "/home/cesar/Descargas/MeIA.csv" \
      --text-col Review --class-col Polarity --remove-stop --save-dir "/home/cesar/Documentos/Tareas-CIMAT/Tercer_Semestre/NLP+CV/Tarea-01/Problema08_polaridad" \
      --k 5 --plots --plot-method both

- Para Type:
  python Problema08_Tarea01_NLP.py --csv "/home/cesar/Descargas/MeIA.csv" \
      --text-col Review --class-col Type --remove-stop --save-dir "/home/cesar/Documentos/Tareas-CIMAT/Tercer_Semestre/NLP+CV/Tarea-01/Problema08" \
      --k 5 --plots --plot-method both    
"""

import os
import argparse
import pandas as pd
import numpy as np

from spacy.lang.es.stop_words import STOP_WORDS
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random

# -----------------------------
# Auxiliares
# -----------------------------
def load_and_clean(csv_path, text_col, remove_stop):
    df = pd.read_csv(csv_path, encoding="latin-1")
    df = df.dropna(subset=[text_col])
    texts = df[text_col].astype(str).tolist()
    tokenized = [simple_preprocess(t, deacc=True) for t in texts]
    if remove_stop:
        tokenized = [[tok for tok in doc if tok not in STOP_WORDS] for doc in tokenized]
    return df, tokenized

def train_word2vec(sentences, vector_size=100, window=5, min_count=5, sg=1, workers=4, seed=42):
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=workers,
        seed=seed
    )
    return model

def document_embeddings(tokenized, model):
    vecs = []
    for doc in tokenized:
        W = [model.wv[w] for w in doc if w in model.wv]
        if W:
            vecs.append(np.mean(W, axis=0))
        else:
            vecs.append(np.zeros(model.vector_size))
    return np.vstack(vecs)

def cluster_and_report(docs, embeddings, k, save_dir, class_labels=None, seed=42):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(embeddings)
    labels = km.labels_
    centroids = km.cluster_centers_

    results_path = os.path.join(save_dir, "clusters_report.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        for c in range(k):
            f.write(f"\n=== CLUSTER {c} ===\n")
            idxs = np.where(labels == c)[0]
            dists = euclidean_distances(embeddings[idxs], [centroids[c]]).ravel()
            nearest = idxs[np.argsort(dists)[:5]]
            for i in nearest:
                text = str(docs[i]).replace("\n", " ")
                f.write(f"- {text[:220]}{'...' if len(text)>220 else ''}\n")

            if class_labels is not None:
                counts = pd.Series(class_labels[idxs]).value_counts()
                f.write("\nDistribución de etiquetas originales en este clúster:\n")
                f.write(str(counts))
                f.write("\n")
    print(f"[OK] Reporte de clusters guardado en {results_path}")
    return labels

def plot_docs_2d(X2d, cluster_labels, save_path, class_labels=None, title="Proyección 2D"):
    plt.figure(figsize=(9, 7))
    k = len(np.unique(cluster_labels))
    rng = np.random.RandomState(42)
    colors = [tuple(rng.rand(3)) for _ in range(k)]

    if class_labels is None:
        for c in range(k):
            idx = (cluster_labels == c)
            plt.scatter(X2d[idx, 0], X2d[idx, 1], alpha=0.7, label=f"Cluster {c}", s=18, c=[colors[c]])
        plt.legend()
    else:
        unique_classes = pd.unique(class_labels)
        markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>']
        for c in range(k):
            idx_c = (cluster_labels == c)
            for j, cls in enumerate(unique_classes):
                idx = idx_c & (class_labels == cls)
                if np.any(idx):
                    plt.scatter(X2d[idx, 0], X2d[idx, 1], alpha=0.7, s=18,
                                c=[colors[c]], marker=markers[j % len(markers)])
        plt.legend(loc="best")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"[OK] Figura guardada en {save_path}")

def make_plots(embeddings, labels, save_dir, class_labels=None, method="both", seed=42):
    # PCA
    if method in ("pca", "both"):
        pca = PCA(n_components=2, random_state=seed)
        Xp = pca.fit_transform(embeddings)
        plot_docs_2d(Xp, labels,
                     os.path.join(save_dir, "docs_pca.png"),
                     class_labels=class_labels,
                     title="Embeddings de documento - PCA (2D)")
    # t-SNE
    if method in ("tsne", "both"):
        try:
            tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto",
                        init="pca", random_state=seed, n_iter=1000)
            Xt = tsne.fit_transform(embeddings)
            plot_docs_2d(Xt, labels,
                         os.path.join(save_dir, "docs_tsne.png"),
                         class_labels=class_labels,
                         title="Embeddings de documento - t-SNE (2D)")
        except Exception as e:
            print(f"[WARN] t-SNE falló: {e}. Generando solo PCA.")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Ruta al dataset CSV")
    parser.add_argument("--text-col", type=str, required=True, help="Columna de texto")
    parser.add_argument("--class-col", type=str, default=None, help="Columna de etiquetas originales (opcional)")
    parser.add_argument("--remove-stop", action="store_true", help="Eliminar stopwords")
    parser.add_argument("--save-dir", type=str, required=True, help="Directorio de salida")
    parser.add_argument("--k", type=int, default=5, help="Número de clusters (default=5)")
    parser.add_argument("--plots", action="store_true", help="Generar gráficos 2D")
    parser.add_argument("--plot-method", type=str, default="pca",
                        choices=["pca", "tsne", "both"], help="Método de proyección 2D")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("[INFO] Cargando dataset...")
    df, tokenized = load_and_clean(args.csv, args.text_col, args.remove_stop)

    print("[INFO] Entrenando Word2Vec...")
    model = train_word2vec(tokenized, seed=args.seed)

    print("[INFO] Calculando embeddings de documentos...")
    doc_embeds = document_embeddings(tokenized, model)

    print(f"[INFO] K-means con k={args.k}...")
    class_labels = df[args.class_col].values if args.class_col else None
    labels = cluster_and_report(df[args.text_col].values, doc_embeds, args.k, args.save_dir, class_labels, seed=args.seed)

    if args.plots:
        print(f"[INFO] Generando figuras ({args.plot_method})...")
        make_plots(doc_embeds, labels, args.save_dir, class_labels=class_labels, method=args.plot_method, seed=args.seed)

    print("[OK] Pipeline completado.")

if __name__ == "__main__":
    main()


