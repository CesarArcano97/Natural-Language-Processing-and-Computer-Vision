# -*- coding: utf-8 -*-
# src/train/train_rnnclf_01.py
import os, json, argparse
from typing import Dict, List, Tuple
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
from src.models.rnn_classifier import RNNClassifier
from src.utils.metrics import (
    compute_basic_metrics, classification_report_dict,
    plot_confusion, roc_ovr, plot_roc_ovr
)
from src.utils.seed import set_seed

# --------------------- utilidades de trazas ---------------------
def plot_train_curves(history: Dict[str, List[float]], out_dir: str):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(history["train_f1_macro"], label="train")
    ax.plot(history["val_f1_macro"], label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1-macro"); ax.set_title("F1-macro")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "f1_curves.png"), dpi=160); plt.close(fig)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_param_groups(model: nn.Module, weight_decay: float):
    """WD para todo MENOS embeddings, biases y capas de norma."""
    wd, no_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_low = n.lower()
        if ("embedding" in n_low) or n_low.endswith(".bias") or ("norm" in n_low) or (".bn" in n_low):
            no_wd.append(p)
        else:
            wd.append(p)
    return [
        {"params": wd, "weight_decay": weight_decay},
        {"params": no_wd, "weight_decay": 0.0},
    ]

# --------------------- un epoch ---------------------
def run_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer = None,
    scheduler: OneCycleLR = None,
    max_grad_norm: float = 1.0,
    word_dropout: float = 0.0,
    pad_idx: int = None,
    unk_idx: int = None,
) -> Tuple[float, Dict[str, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    y_true, y_pred, y_probs = [], [], []

    for batch in tqdm(loader, disable=False):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)

        # ---- word dropout (solo train) ----
        if is_train and word_dropout > 0.0 and pad_idx is not None and unk_idx is not None:
            drop_mask = (torch.rand_like(input_ids, dtype=torch.float32) < word_dropout) & (input_ids != pad_idx)
            input_ids = input_ids.masked_fill(drop_mask, unk_idx)

        logits = model(input_ids=input_ids, attention_mask=attn_mask)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            # ---- OneCycle por batch ----
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        y_probs.append(probs); y_pred.append(preds); y_true.append(labels.detach().cpu().numpy())

    N = len(loader.dataset)
    avg_loss = total_loss / max(N, 1)
    y_true = np.concatenate(y_true, axis=0) if len(y_true) > 0 else np.array([])
    y_pred = np.concatenate(y_pred, axis=0) if len(y_pred) > 0 else np.array([])
    y_probs = np.concatenate(y_probs, axis=0) if len(y_probs) > 0 else np.array([])
    metrics = compute_basic_metrics(y_true, y_pred) if y_true.size > 0 else {"accuracy":0.0,"f1_macro":0.0}
    return avg_loss, metrics, (y_true, y_pred, y_probs)

# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser(prog="train_rnnclf_01")

    # Rutas
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    # Modelo
    ap.add_argument("--rnn_type", type=str, default="rnn", choices=["rnn","lstm","gru"])
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--pool", type=str, default="max", choices=["max","mean","maxmean","last"])
    ap.add_argument("--emb_dropout", type=float, default=0.2)
    ap.add_argument("--rnn_dropout", type=float, default=0.2)
    ap.add_argument("--proj_dropout", type=float, default=0.5)

    # Optimizador / regularización
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4,
                    help="L2 para TODO menos embeddings/bias/norm")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--word_dropout", type=float, default=0.0,
                    help="Prob. de reemplazar tokens por <unk> durante TRAIN")
    ap.add_argument("--onecycle", action="store_true")
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # Entrenamiento
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)

    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- meta ----
    with open(os.path.join(args.data_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    pad_idx       = meta["pad_idx"]
    unk_idx       = meta.get("unk_idx", 1)
    vocab_size    = meta["vocab_size"]
    label_map     = meta["label_map"]
    class_weights = torch.tensor(meta["class_weights"], dtype=torch.float32)
    # sanity
    class_names = [k for k,_ in sorted(label_map.items(), key=lambda x: x[1])]
    num_classes = len(class_names)
    if class_weights.numel() != num_classes:
        class_weights = class_weights[:num_classes]

    # ---- datos ----
    train_loader, val_loader = build_loaders(
        train_pt=os.path.join(args.data_dir, "train.pt"),
        val_pt=os.path.join(args.data_dir, "val.pt"),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- modelo ----
    model = RNNClassifier(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_classes=num_classes,
        pad_idx=pad_idx,
        rnn_type=args.rnn_type,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        emb_dropout=args.emb_dropout,
        rnn_dropout=args.rnn_dropout,
        proj_dropout=args.proj_dropout,
        pool=args.pool
    ).to(device)

    # ---- pérdida con label smoothing ----
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=args.label_smoothing
    )

    # ---- optimizador: sin WD en embeddings/bias/norm ----
    optimizer = optim.AdamW(build_param_groups(model, args.weight_decay), lr=args.lr)

    # ---- OneCycle por batch (opcional) ----
    scheduler = None
    if args.onecycle:
        steps_per_epoch = max(1, len(train_loader))
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )

    # ---- log config ----
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        cfg = dict(args=vars(args), meta={k: meta[k] for k in ["vocab_size","pad_idx","unk_idx","label_map"] if k in meta},
                   num_params=count_parameters(model))
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # ---- entrenamiento ----
    history = {"train_loss": [], "val_loss": [], "train_f1_macro": [], "val_f1_macro": []}
    best_f1, patience = -1.0, args.patience

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_metrics, _ = run_epoch(
            model, train_loader, device, criterion, optimizer,
            scheduler=scheduler,
            max_grad_norm=args.max_grad_norm,
            word_dropout=args.word_dropout, pad_idx=pad_idx, unk_idx=unk_idx
        )

        val_loss, val_metrics, (y_true_v, y_pred_v, y_proba_v) = run_epoch(
            model, val_loader, device, criterion, optimizer=None,
            scheduler=None,
            max_grad_norm=None,
            word_dropout=0.0, pad_idx=pad_idx, unk_idx=unk_idx
        )

        history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
        history["train_f1_macro"].append(tr_metrics["f1_macro"]); history["val_f1_macro"].append(val_metrics["f1_macro"])
        print(f"  train: loss={tr_loss:.4f} acc={tr_metrics['accuracy']:.4f} f1M={tr_metrics['f1_macro']:.4f}")
        print(f"  valid: loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f} f1M={val_metrics['f1_macro']:.4f}")

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]; patience = args.patience
            ckpt = {
                "model_state": model.state_dict(),
                "args": vars(args),
                "meta": meta,
                "val_metrics": val_metrics,
                "arch": args.rnn_type,   # <-- corregido
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best_model.pt"))
            # Plots y reportes de validación
            plot_confusion(y_true_v, y_pred_v, class_names, os.path.join(args.out_dir, "confusion_val.png"))
            fpr, tpr, roc_auc = roc_ovr(y_true_v, y_proba_v)
            plot_roc_ovr(fpr, tpr, roc_auc, class_names, os.path.join(args.out_dir, "roc_val.png"))
            with open(os.path.join(args.out_dir, "cls_report_val.json"), "w", encoding="utf-8") as f:
                json.dump(classification_report_dict(y_true_v, y_pred_v, label_map), f, ensure_ascii=False, indent=2)
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping por paciencia agotada.")
                break

    plot_train_curves(history, args.out_dir)
    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Mejor F1-macro val = {best_f1:.4f}")
    print(f"Modelo guardado en: {os.path.join(args.out_dir, 'best_model.pt')}")

if __name__ == "__main__":
    main()


