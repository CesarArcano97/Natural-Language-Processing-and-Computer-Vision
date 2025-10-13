# -*- coding: utf-8 -*-
"""
Prepara el corpus MeIA para clasificación:
- Lee data/raw/MeIA.csv con columnas: Review, Polarity (1..5 flotante o int)
- Estratifica en train/val/test
- Tokeniza (regex español), construye vocab desde train
- Convierte textos a ids con trunc/pad
- Genera class_weights, label_map y meta.json
- Guarda .pt (listas de dicts) y vocab.json

Ejemplo:
python src/data/prepare_meia.py \
  --input data/raw/MeIA.csv \
  --out_dir data/processed/classif/five \
  --scheme five --max_len 256 --min_freq 2 --val_size 0.1 --test_size 0.1
"""
import os, json, argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from collections import Counter

from src.utils.text import tokenize_es
from src.utils.vocab import build_vocab, save_vocab, PAD, UNK
from src.utils.seed import set_seed

def make_label_map(scheme: str):
    # Devuelve función mapper y label_map (dict str->id)
    if scheme == "five":
        # 1..5 -> 0..4
        classes = ["1","2","3","4","5"]
        def mapper(x):
            v = int(round(float(x)))
            return v-1
    elif scheme == "ternary":
        classes = ["neg","neu","pos"]
        def mapper(x):
            v = float(x)
            if v <= 2: return 0
            if v >= 4: return 2
            return 1
    elif scheme == "binary":
        classes = ["neg","pos"]
        def mapper(x):
            v = float(x)
            return 0 if v <= 3 else 1
    else:
        raise ValueError("scheme must be one of {five, ternary, binary}")
    label_map = {c:i for i,c in enumerate(classes)}
    return mapper, label_map

def encode(tokens, stoi, max_len, pad_idx, unk_idx):
    ids = [stoi.get(t, unk_idx) for t in tokens][:max_len]
    attn = [1]*len(ids)
    if len(ids) < max_len:
        pad_n = max_len - len(ids)
        ids += [pad_idx]*pad_n
        attn += [0]*pad_n
    return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scheme", choices=["five","ternary","binary"], default="five")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    assert "Review" in df.columns and "Polarity" in df.columns, "Faltan columnas"

    # Limpieza mínima: quitamos NaN y espacios
    df = df.dropna(subset=["Review","Polarity"]).copy()
    df["Review"] = df["Review"].astype(str).str.strip()

    mapper, label_map = make_label_map(args.scheme)
    df["label"] = df["Polarity"].apply(mapper)
    y = df["label"].values

    # train/val/test estratificado
    df_train, df_tmp = train_test_split(
        df, test_size=args.val_size + args.test_size,
        stratify=y, random_state=args.seed
    )
    rel_val = args.val_size / (args.val_size + args.test_size)
    df_val, df_test = train_test_split(
        df_tmp, test_size=1-rel_val,
        stratify=df_tmp["label"].values, random_state=args.seed
    )

    # Tokenización y vocab (solo con train)
    train_tokens = [tokenize_es(t) for t in df_train["Review"].tolist()]
    stoi, itos = build_vocab(train_tokens, min_freq=args.min_freq, specials=(PAD, UNK))
    pad_idx, unk_idx = stoi[PAD], stoi[UNK]

    def convert_split(df_split):
        items = []
        for text, lab in zip(df_split["Review"].tolist(), df_split["label"].tolist()):
            toks = tokenize_es(text)
            ids, attn = encode(toks, stoi, args.max_len, pad_idx, unk_idx)
            items.append({"input_ids": ids, "attention_mask": attn, "label": int(lab)})
        return items

    train_items = convert_split(df_train)
    val_items   = convert_split(df_val)
    test_items  = convert_split(df_test)

    # Pesos de clase (para CrossEntropy)
    counts = Counter([it["label"] for it in train_items])
    num_classes = len(label_map)
    N = len(train_items)
    class_weights = [N / (num_classes * counts[c]) for c in range(num_classes)]

    # Guardar tensores (listas de dicts)
    torch.save(train_items, os.path.join(args.out_dir, "train.pt"))
    torch.save(val_items,   os.path.join(args.out_dir, "val.pt"))
    torch.save(test_items,  os.path.join(args.out_dir, "test.pt"))
    save_vocab(stoi, os.path.join(args.out_dir, "vocab.json"))

    meta = {
        "pad_idx": pad_idx,
        "unk_idx": unk_idx,
        "max_len": args.max_len,
        "vocab_size": len(stoi),
        "label_map": label_map,      # e.g., {"1":0,...} o {"neg":0,"neu":1,"pos":2}
        "class_weights": class_weights,
        "scheme": args.scheme,
        "min_freq": args.min_freq,
        "splits": {
            "train": len(train_items),
            "val": len(val_items),
            "test": len(test_items)
        }
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Procesado en {args.out_dir}")
    print(f"Vocab size: {len(stoi)} | train/val/test = "
          f"{len(train_items)}/{len(val_items)}/{len(test_items)}")

if __name__ == "__main__":
    main()
