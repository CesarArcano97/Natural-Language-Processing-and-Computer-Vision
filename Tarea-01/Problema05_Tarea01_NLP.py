#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problema05_Tarea01_NLP.py

Construcción de representaciones BoW (TF y TF-IDF) y selección de características
usando tres métricas: chi-cuadrada, mutual information, information gain.

Uso:
  python3 Problema05_Tarea01_NLP.py --csv "/home/cesar/Descargas/MeIA.csv" \
      --text-col Review --class-col Type --top-n 20 \
      --remove-stop --save-dir out_type --plots
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy


# =====================
# Preprocesamiento
# =====================
def load_data(csv_path, text_col, class_col):
    df = pd.read_csv(csv_path)
    df = df[[text_col, class_col]].dropna()
    return df


def get_vectorizers(remove_stop):
    """Devuelve vectorizadores TF y TF-IDF."""
    if remove_stop:
        try:
            import nltk
            from nltk.corpus import stopwords
            nltk.download("stopwords", quiet=True)
            stop_es = stopwords.words("spanish")
        except:
            import spacy
            nlp = spacy.blank("es")
            stop_es = list(nlp.Defaults.stop_words)
        kwargs = dict(stop_words=stop_es)
    else:
        kwargs = {}

    vec_tf = CountVectorizer(**kwargs)
    vec_tfidf = TfidfVectorizer(**kwargs)
    return vec_tf, vec_tfidf


# =====================
# Medidas estadísticas
# =====================
def compute_chi2(X, y, feature_names):
    chi2_vals, _ = chi2(X, y)
    return pd.DataFrame({"term": feature_names, "score": chi2_vals}).sort_values(
        "score", ascending=False
    )


def compute_mi(X, y, feature_names):
    mi_vals = mutual_info_classif(X, y, discrete_features=True)
    return pd.DataFrame({"term": feature_names, "score": mi_vals}).sort_values(
        "score", ascending=False
    )


def compute_ig(X, y, feature_names):
    """Information Gain como reducción de entropía."""
    n_docs = X.shape[0]
    H_C = entropy(np.bincount(y) / n_docs, base=2)

    ig_vals = []
    for j in range(X.shape[1]):
        col = X[:, j].toarray().ravel()
        mask = col > 0
        H_C_t = entropy(np.bincount(y[mask]) / max(mask.sum(), 1), base=2) if mask.sum() > 0 else 0
        H_C_not = entropy(np.bincount(y[~mask]) / max((~mask).sum(), 1), base=2) if (~mask).sum() > 0 else 0
        H_cond = (mask.mean() * H_C_t) + ((~mask).mean() * H_C_not)
        ig_vals.append(H_C - H_cond)

    return pd.DataFrame({"term": feature_names, "score": ig_vals}).sort_values(
        "score", ascending=False
    )


# =====================
# Visualización
# =====================
def plot_top(df, metric, top_n, save_dir, class_name):
    top_terms = df.head(top_n)
    plt.figure(figsize=(8, 6))
    sns.barplot(x="score", y="term", data=top_terms, color="skyblue")
    plt.title(f"Top {top_n} términos ({metric}) - Clase: {class_name}")
    plt.tight_layout()
    fname = f"top_{metric}_{class_name}.png".replace(" ", "_")
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()


# =====================
# Main Pipeline
# =====================
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Cargar datos
    df = load_data(args.csv, args.text_col, args.class_col)

    le = LabelEncoder()
    y_full = le.fit_transform(df[args.class_col])
    class_names = le.classes_

    # 2) Vectorización
    vec_tf, vec_tfidf = get_vectorizers(args.remove_stop)
    X_tf = vec_tf.fit_transform(df[args.text_col])
    X_tfidf = vec_tfidf.fit_transform(df[args.text_col])
    feature_names_tf = vec_tf.get_feature_names_out()
    feature_names_tfidf = vec_tfidf.get_feature_names_out()

    # 3) One-vs-rest por clase
    for class_idx, class_name in enumerate(class_names):
        print(f"🔎 Procesando clase: {class_name}")
        y_bin = (y_full == class_idx).astype(int)

        for name, X, feats in [
            ("TF", X_tf, feature_names_tf),
            ("TFIDF", X_tfidf, feature_names_tfidf),
        ]:
            chi2_df = compute_chi2(X, y_bin, feats)
            mi_df = compute_mi(X, y_bin, feats)
            ig_df = compute_ig(X, y_bin, feats)

            # Guardar CSVs
            chi2_df.to_csv(os.path.join(args.save_dir, f"{name}_chi2_{class_name}.csv"), index=False)
            mi_df.to_csv(os.path.join(args.save_dir, f"{name}_mi_{class_name}.csv"), index=False)
            ig_df.to_csv(os.path.join(args.save_dir, f"{name}_ig_{class_name}.csv"), index=False)

            # Graficar
            if args.plots:
                plot_top(chi2_df, f"{name}_chi2", args.top_n, args.save_dir, class_name)
                plot_top(mi_df, f"{name}_mi", args.top_n, args.save_dir, class_name)
                plot_top(ig_df, f"{name}_ig", args.top_n, args.save_dir, class_name)

    print("✅ Proceso completado. Resultados guardados en:", args.save_dir)
    print("📂 Archivos generados:", os.listdir(args.save_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--text-col", type=str, required=True)
    parser.add_argument("--class-col", type=str, required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--remove-stop", action="store_true")
    parser.add_argument("--save-dir", type=str, default="out")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    main(args)

