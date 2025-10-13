# -*- coding: utf-8 -*-
"""
evaluate.py
Evalúa un modelo TextCNN (baseline) guardado como best_model.pt sobre el split de test.
Genera: accuracy/F1, matriz de confusión (png), ROC OvR (png), reporte por clase (json)
y un CSV con y_true, y_pred y probas para análisis de errores.

Ejemplo:
python src/eval/evaluate.py \
  --data_dir data/processed/classif/five \
  --ckpt models/cnn/exp_textcnn_five/best_model.pt \
  --out_dir models/cnn/exp_textcnn_five
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import TensorListDataset, _collate  # reutilizamos el collate del loader
from src.models.textcnn import TextCNN
from src.utils.metrics import (
    compute_basic_metrics, classification_report_dict,
    plot_confusion, roc_ovr, plot_roc_ovr
)

def run_eval(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            logits = model(input_ids=input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item() * input_ids.size(0)

            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            y_probs.append(probs)
            y_pred.append(preds)
            y_true.append(labels.detach().cpu().numpy())

    N = len(loader.dataset)
    avg_loss = total_loss / max(N, 1)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_probs = np.concatenate(y_probs, axis=0)
    metrics = compute_basic_metrics(y_true, y_pred)
    return avg_loss, metrics, (y_true, y_pred, y_probs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="carpeta con test.pt y meta.json")
    ap.add_argument("--ckpt", required=True, help="ruta al best_model.pt")
    ap.add_argument("--out_dir", required=True, help="carpeta donde guardar figuras y json")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- meta / label_map ----
    with open(os.path.join(args.data_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    pad_idx    = meta["pad_idx"]
    vocab_size = meta["vocab_size"]
    label_map  = meta["label_map"]
    class_names = [k for k,_ in sorted(label_map.items(), key=lambda x: x[1])]
    num_classes = len(class_names)

    # ---- datos de test ----
    test_items = torch.load(os.path.join(args.data_dir, "test.pt"))
    test_loader = DataLoader(
        TensorListDataset(test_items),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=_collate
    )

    # ---- carga del checkpoint y construcción del modelo ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    margs = ckpt["args"]  # parámetros usados en entrenamiento

    kernel_sizes = [int(x) for x in margs["kernel_sizes"].split(",")]
    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=margs["embed_dim"],
        num_classes=num_classes,
        pad_idx=pad_idx,
        kernel_sizes=kernel_sizes,
        num_filters=margs["num_filters"],
        emb_dropout=margs["emb_dropout"],
        proj_dropout=margs["proj_dropout"],
        use_batchnorm=margs["use_batchnorm"]
    )
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # misma pérdida (sin pesos basta para evaluar, pero podemos recargar si quieres)
    class_weights = torch.tensor(meta["class_weights"], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ---- evaluación ----
    test_loss, test_metrics, (y_true, y_pred, y_proba) = run_eval(model, test_loader, device, criterion)

    # ---- guardar resultados ----
    print(f"[Test] loss={test_loss:.4f} acc={test_metrics['accuracy']:.4f} f1M={test_metrics['f1_macro']:.4f}")
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "loss": test_loss,
            **test_metrics,
            "num_examples": len(test_items)
        }, f, ensure_ascii=False, indent=2)

    # reporte por clase
    with open(os.path.join(args.out_dir, "cls_report_test.json"), "w", encoding="utf-8") as f:
        json.dump(classification_report_dict(y_true, y_pred, label_map), f, ensure_ascii=False, indent=2)

    # matriz de confusión
    plot_confusion(y_true, y_pred, class_names, os.path.join(args.out_dir, "confusion_test.png"))

    # ROC OvR (solo para clases presentes en y_true)
    fpr, tpr, roc_auc = roc_ovr(y_true, y_proba)
    plot_roc_ovr(fpr, tpr, roc_auc, class_names, os.path.join(args.out_dir, "roc_test.png"))

    # CSV con predicciones y proba máxima (útil para análisis de errores)
    import pandas as pd
    top_p = y_proba.max(axis=1)
    df_out = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "p_max": top_p
    })
    df_out.to_csv(os.path.join(args.out_dir, "predictions_test.csv"), index=False)
    print("[OK] Resultados de test guardados.")

if __name__ == "__main__":
    main()
