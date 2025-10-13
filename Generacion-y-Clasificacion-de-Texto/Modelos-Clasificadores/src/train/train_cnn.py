# -*- coding: utf-8 -*-
"""
train_cnn.py
Entrena TextCNN sobre los .pt preparados.
Incluye: optimizer (AdamW), pérdida (CrossEntropy con class_weights), early stopping en F1-macro,
plots de curvas, matriz de confusión y ROC OvR en validación.

Ejemplo:
python src/train/train_cnn.py \
  --data_dir data/processed/classif/five \
  --out_dir models/cnn/exp_textcnn_five
"""

import os, json, argparse, math
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import build_loaders
from src.models.textcnn import TextCNN
from src.utils.metrics import (
    compute_basic_metrics, classification_report_dict,
    plot_confusion, roc_ovr, plot_roc_ovr
)
from src.utils.seed import set_seed

def plot_train_curves(history: Dict[str, List[float]], out_dir: str):
    # Loss
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=160)
    plt.close(fig)
    # F1-macro
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(history["train_f1_macro"], label="train")
    ax.plot(history["val_f1_macro"], label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1-macro"); ax.set_title("F1-macro")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "f1_curves.png"), dpi=160)
    plt.close(fig)

def run_epoch(model, loader, device, criterion, optimizer=None, max_grad_norm=1.0):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    y_true, y_pred, y_probs = [], [], []

    for batch in tqdm(loader, disable=False):
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        logits = model(input_ids=input_ids)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

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
    ap.add_argument("--data_dir", required=True, help="carpeta con train.pt/val.pt/meta.json")
    ap.add_argument("--out_dir", required=True)
    # Modelo
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--kernel_sizes", type=str, default="2,3,4,5")
    ap.add_argument("--num_filters", type=int, default=128)
    ap.add_argument("--emb_dropout", type=float, default=0.2)
    ap.add_argument("--proj_dropout", type=float, default=0.5)
    ap.add_argument("--use_batchnorm", action="store_true")
    # Optimizador
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--onecycle", action="store_true", help="usar OneCycleLR")
    # Entrenamiento
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Cargar meta
    with open(os.path.join(args.data_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    pad_idx      = meta["pad_idx"]
    vocab_size   = meta["vocab_size"]
    label_map    = meta["label_map"]
    class_weights = torch.tensor(meta["class_weights"], dtype=torch.float32)

    class_names = [k for k,_ in sorted(label_map.items(), key=lambda x: x[1])]
    num_classes = len(class_names)

    # DataLoaders
    train_loader, val_loader = build_loaders(
        train_pt=os.path.join(args.data_dir, "train.pt"),
        val_pt=os.path.join(args.data_dir, "val.pt"),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- (Bloque 5) Modelo ----
    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_classes=num_classes,
        pad_idx=pad_idx,
        kernel_sizes=[int(x) for x in args.kernel_sizes.split(",")],
        num_filters=args.num_filters,
        emb_dropout=args.emb_dropout,
        proj_dropout=args.proj_dropout,
        use_batchnorm=args.use_batchnorm
    ).to(device)

    # ---- (Bloque 3) Función de pérdida ----
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ---- (Bloque 2) Optimizador (+ scheduler opcional) ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.onecycle:
        steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)
        scheduler = OneCycleLR(
            optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch
        )
    else:
        scheduler = None

    # ---- (Bloque 6) Entrenamiento + Early Stopping en F1-macro ----
    history = {"train_loss": [], "val_loss": [], "train_f1_macro": [], "val_f1_macro": []}
    best_f1, best_state, patience = -1.0, None, args.patience

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_metrics, _ = run_epoch(model, train_loader, device, criterion, optimizer)
        if scheduler is not None:
            scheduler.step()
        val_loss, val_metrics, (y_true_val, y_pred_val, y_proba_val) = run_epoch(model, val_loader, device, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1_macro"].append(train_metrics["f1_macro"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        print(f"  train: loss={train_loss:.4f} acc={train_metrics['accuracy']:.4f} f1M={train_metrics['f1_macro']:.4f}")
        print(f"  valid: loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f} f1M={val_metrics['f1_macro']:.4f}")

        # guardar mejor
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            patience = args.patience
            best_state = {
                "model_state": model.state_dict(),
                "args": vars(args),
                "meta": meta,
                "val_metrics": val_metrics
            }
            torch.save(best_state, os.path.join(args.out_dir, "best_model.pt"))
            # plots de validación en el mejor punto
            plot_confusion(y_true_val, y_pred_val, class_names, os.path.join(args.out_dir, "confusion_val.png"))
            fpr, tpr, roc_auc = roc_ovr(y_true_val, y_proba_val)
            plot_roc_ovr(fpr, tpr, roc_auc, class_names, os.path.join(args.out_dir, "roc_val.png"))
            # reporte por clase
            with open(os.path.join(args.out_dir, "cls_report_val.json"), "w", encoding="utf-8") as f:
                json.dump(classification_report_dict(y_true_val, y_pred_val, label_map), f, ensure_ascii=False, indent=2)
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping por paciencia agotada.")
                break

    # Curvas
    plot_train_curves(history, args.out_dir)

    print(f"\n[OK] Mejor F1-macro val = {best_f1:.4f}")
    print(f"Modelo guardado en: {os.path.join(args.out_dir, 'best_model.pt')}")

if __name__ == "__main__":
    main()
