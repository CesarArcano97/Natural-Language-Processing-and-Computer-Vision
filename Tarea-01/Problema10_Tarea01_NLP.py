#!/usr/bin/env python3
# ============================================
# Problema 10 - Tarea NLP
# LSA con 50 tópicos + métricas + visualización
# ============================================

'''
Correr como:

python Problema10_Tarea01_NLP.py     --csv "/home/cesar/Descargas/MeIA.csv" 
 --text-col Review     --class-col Type     
 --save-dir "/home/cesar/Documentos/Tareas-CIMAT/Tercer_Semestre/NLP+CV/Tarea-01/Problema10"     
 --metric mi
'''

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import chi2, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
spanish_stopwords = stopwords.words("spanish")

from wordcloud import WordCloud


# ------------------------------
# Funciones auxiliares
# ------------------------------
def top_terms_for_topic(svd, terms, topic_idx, n_terms=10):
    """Devuelve las palabras más importantes de un tópico"""
    comp = svd.components_[topic_idx]
    top_idx = np.argsort(comp)[::-1][:n_terms]
    return list(terms[top_idx])


# ------------------------------
# Main
# ------------------------------
def main(args):
    # 1. Cargar datos
    df = pd.read_csv(args.csv)
    texts = df[args.text_col].astype(str).tolist()
    y = df[args.class_col]

    # 2. Matriz TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=spanish_stopwords)
    X_tfidf = vectorizer.fit_transform(texts)
    print("Shape TF-IDF:", X_tfidf.shape)

    # 3. LSA (SVD truncado)
    n_topics = 50
    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    doc_topic_matrix = svd.fit_transform(X_tfidf)
    terms = np.array(vectorizer.get_feature_names_out())
    print("Shape doc-topic:", doc_topic_matrix.shape)

    # 4. Palabras más relevantes por tópico (ejemplo primeros 5)
    print("\n=== Palabras más relevantes por tópico (ejemplo) ===")
    for i in range(5):
        print(f"Tópico {i}: {top_terms_for_topic(svd, terms, i, 10)}")

    # 5. Ranking con métrica elegida
    if args.metric == "chi2":
        # Shift para asegurar no-negatividad
        X_for_metric = doc_topic_matrix - doc_topic_matrix.min()
        scores, p_values = chi2(X_for_metric, y)
        ranking = np.argsort(scores)[::-1]
        metric_name = "Chi2"
        score_values = scores
    elif args.metric == "mi":
        scores = mutual_info_classif(doc_topic_matrix, y, random_state=42)
        ranking = np.argsort(scores)[::-1]
        metric_name = "Mutual Information"
        score_values = scores
        p_values = None
    else:
        raise ValueError("Métrica no válida. Usa --metric chi2 o mi")

    os.makedirs(args.save_dir, exist_ok=True)

    # 6. Guardar CSV con TODOS los tópicos
    table_rows = []
    for idx in range(n_topics):
        row = {
            "Topic": idx,
            metric_name: score_values[idx],
            "Top Terms": ", ".join(top_terms_for_topic(svd, terms, idx, 10))
        }
        if p_values is not None:
            row["p-value"] = p_values[idx]
        table_rows.append(row)

    df_out = pd.DataFrame(table_rows)
    df_out.to_csv(os.path.join(args.save_dir, f"all_topics_{args.metric}.csv"), index=False)

    # 7. Top-10 tópicos
    top_k = 10
    top_topics = ranking[:top_k]

    print(f"\n=== Top {top_k} tópicos más informativos ({metric_name}) ===")
    for idx in top_topics:
        if p_values is not None:
            print(f"Tópico {idx} -> {metric_name}={score_values[idx]:.2f}, p={p_values[idx]:.3e}")
        else:
            print(f"Tópico {idx} -> {metric_name}={score_values[idx]:.4f}")

    # 8. Gráfico de barras (Top-10)
    plt.figure(figsize=(8, 4))
    plt.bar(range(top_k), score_values[top_topics])
    plt.xticks(range(top_k), [f"T{idx}" for idx in top_topics])
    plt.title(f"Top {top_k} Tópicos más informativos ({metric_name})")
    plt.ylabel(metric_name)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, f"top{top_k}_topics_{args.metric}.png"))
    plt.close()

    # 9. Heatmap (clases × top-10)
    topic_class_matrix = pd.DataFrame(doc_topic_matrix[:, top_topics],
                                      columns=[f"T{t}" for t in top_topics])
    topic_class_matrix["Class"] = y.values
    mean_by_class = topic_class_matrix.groupby("Class").mean()

    plt.figure(figsize=(10, 6))
    sns.heatmap(mean_by_class, cmap="viridis", annot=True, fmt=".2f")
    plt.title(f"Activación promedio de tópicos por clase (Top {top_k})")
    plt.xlabel("Tópico")
    plt.ylabel("Clase")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, f"heatmap_top{top_k}_{args.metric}.png"))
    plt.close()

    # 10. WordClouds (Top-10)
    for idx in top_topics:
        words = top_terms_for_topic(svd, terms, idx, 30)
        text = " ".join(words)
        wc = WordCloud(width=500, height=300, background_color="white").generate(text)

        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Tópico {idx} (Top Terms)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, f"wordcloud_topic{idx}.png"))
        plt.close()

    print(f"\nResultados guardados en: {args.save_dir}")


# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Problema 10 - LSA con 50 tópicos")
    parser.add_argument("--csv", type=str, required=True, help="Ruta al CSV de entrada")
    parser.add_argument("--text-col", type=str, required=True, help="Nombre de la columna de texto")
    parser.add_argument("--class-col", type=str, required=True, help="Nombre de la columna de clase")
    parser.add_argument("--save-dir", type=str, required=True, help="Directorio para guardar resultados")
    parser.add_argument("--metric", type=str, default="mi", choices=["chi2", "mi"],
                        help="Métrica para evaluar tópicos: chi2 o mi (default: mi)")

    args = parser.parse_args()
    main(args)
