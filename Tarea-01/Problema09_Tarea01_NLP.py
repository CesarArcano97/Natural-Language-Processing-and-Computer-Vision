#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problema09_Tarea01_NLP.py

Clasificación de texto (SVM o Regresión Logística) con 4 configuraciones acumulativas de preprocesamiento:
(a) Sin preprocesamiento
(b) Minúsculas
(c) Minúsculas + lematización
(d) Minúsculas + lematización + filtrado de palabras con frecuencia mínima de 10

Guarda métricas, matrices de confusión y un resumen comparativo.

Uso:
  python Problema09_Tarea01_NLP.py --csv "/home/cesar/Descargas/MeIA.csv" \
      --text-col Review --class-col Type \
      --save-dir "/home/cesar/Documentos/Tareas-CIMAT/Tercer_Semestre/NLP+CV/Tarea-01/Problema09_log" \
      --model svm
"""

import os
import argparse
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

import spacy
nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

# ==============================
# Preprocesamiento
# ==============================
def preprocess_text(texts, experiment):
    """
    Aplica preprocesamiento acumulativo según experimento:
    a = nada
    b = minúsculas
    c = minúsculas + lematización
    d = minúsculas + lematización (filtrado min_df posterior)
    """
    proc = []
    for t in texts:
        if experiment == "a":
            proc.append(t)

        elif experiment == "b":
            proc.append(t.lower())

        elif experiment in ("c", "d"):
            doc = nlp(t.lower())
            tokens = [tok.lemma_ for tok in doc if not tok.is_punct and not tok.is_space]
            proc.append(" ".join(tokens))

    return proc

# ==============================
# Matriz de confusión (plot)
# ==============================
def plot_confusion(cm, labels, save_path, title="Matriz de confusión"):
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Guardada figura en {save_path}")

# ==============================
# Pipeline de entrenamiento
# ==============================
def run_experiment(df, text_col, class_col, experiment, model_type, save_dir, seed=42):
    print(f"\n[INFO] Ejecutando experimento {experiment.upper()}...")

    X_raw = df[text_col].astype(str).tolist()
    y = df[class_col].astype(str).values

    # Preprocesamiento
    X_proc = preprocess_text(X_raw, experiment)

    # Vectorización
    min_df_val = 1 if experiment != "d" else 2
    vectorizer = CountVectorizer(min_df=min_df_val)
    X = vectorizer.fit_transform(X_proc)

    # Split 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    # Clasificador
    if model_type == "svm":
        clf = LinearSVC(random_state=seed)
    else:
        clf = LogisticRegression(max_iter=200, random_state=seed)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))

    # Reporte individual
    out_path = os.path.join(save_dir, f"exp_{experiment}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Resultados experimento {experiment.upper()}\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1-macro: {f1:.4f}\n\n")
        f.write("Matriz de confusión:\n")
        f.write(str(cm))
        f.write("\n\nReporte de clasificación:\n")
        f.write(classification_report(y_test, y_pred))
    print(f"[OK] Resultados guardados en {out_path}")

    # Figura de matriz de confusión
    fig_path = os.path.join(save_dir, f"exp_{experiment}_cm.png")
    plot_confusion(cm, np.unique(y), fig_path,
                   title=f"Matriz de confusión - Exp {experiment.upper()}")

    return {"exp": experiment, "accuracy": acc, "f1_macro": f1}

# ==============================
# Main
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Ruta al dataset CSV")
    parser.add_argument("--text-col", type=str, required=True, help="Columna de texto")
    parser.add_argument("--class-col", type=str, required=True, help="Columna de etiquetas")
    parser.add_argument("--save-dir", type=str, required=True, help="Directorio de salida")
    parser.add_argument("--model", type=str, default="svm", choices=["svm","logreg"],
                        help="Clasificador a usar")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding="latin-1")
    os.makedirs(args.save_dir, exist_ok=True)

    resultados = []
    for exp in ["a","b","c","d"]:
        res = run_experiment(df, args.text_col, args.class_col, exp, args.model, args.save_dir, args.seed)
        resultados.append(res)

    # Guardar comparativa en CSV
    df_results = pd.DataFrame(resultados)
    out_csv = os.path.join(args.save_dir, "resumen_experimentos.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\n[OK] Resumen comparativo guardado en {out_csv}")
    print(df_results)

if __name__ == "__main__":
    main()
