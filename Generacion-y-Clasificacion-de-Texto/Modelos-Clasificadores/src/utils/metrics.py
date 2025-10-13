# -*- coding: utf-8 -*-
"""
metrics.py
Métricas y plots para clasificación multiclase.
"""

from typing import Dict, List, Tuple
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless (cluster)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc
)

def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }

def classification_report_dict(y_true, y_pred, label_map: Dict[str, int]) -> Dict:
    inv = {v:k for k,v in label_map.items()}
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0,
        target_names=[inv[i] for i in range(len(inv))]
    )
    return report

def plot_confusion(y_true, y_pred, class_names: List[str], out_path: str, normalize: bool = True):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        cm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        ylabel="True", xlabel="Predicted", title="Confusion Matrix"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]),
                    ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def roc_ovr(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, float]]:
    """
    ROC One-vs-Rest para multiclase.
    y_true: (N,) etiquetas en [0..C-1]
    y_proba: (N, C) probabilidades (softmax)
    Devuelve diccionarios fpr, tpr, auc por clase.
    """
    classes = np.unique(y_true)
    fpr, tpr, roc_auc = {}, {}, {}
    for c in classes:
        # binariza: clase c vs resto
        y_bin = (y_true == c).astype(int)
        fpr[c], tpr[c], _ = roc_curve(y_bin, y_proba[:, c])
        roc_auc[c] = auc(fpr[c], tpr[c])
    return fpr, tpr, roc_auc

def plot_roc_ovr(fpr, tpr, roc_auc, class_names: List[str], out_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    for c, name in enumerate(class_names):
        if c in fpr:
            ax.plot(fpr[c], tpr[c], label=f"{name} (AUC={roc_auc[c]:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC OvR (val/test)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
