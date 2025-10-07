#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class_terms.py
Extracción de 4-gramas de POS por clase con frecuencia, log-odds (Monroe et al., 2008),
divergencia de Jensen–Shannon (JSD) entre clases y ejemplos de patrones explicativos.

Uso típico:
  python3 class_terms.py --csv "/home/cesar/Descargas/MeIA.csv" \
      --text-col Review --class-col Type --top-n 20 \
      --remove-stop --save-dir out_type --plots
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter
from itertools import islice

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from scipy.spatial.distance import jensenshannon


# ============================
# Utilidades generales
# ============================

def slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-\.áéíóúñü]", "", s)
    return s

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_json(d: dict, path: Path):
    path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================
# Extracción POS 4-gramas
# ============================

def sliding_ngrams(seq, n):
    for i in range(len(seq) - n + 1):
        yield tuple(seq[i:i+n])

def pos_sequence(doc, keep_punct=False, remove_stop=False):
    tags = []
    for tok in doc:
        if not keep_punct and tok.is_punct:
            continue
        if remove_stop and tok.is_stop:
            continue
        if tok.is_space:
            continue
        tags.append(tok.pos_)
    return tags

def extract_pos4_counts(nlp, texts, keep_punct=False, remove_stop=False, n_proc=1, batch_size=128, max_docs=None):
    cnt = Counter()
    if max_docs is not None:
        texts = list(islice(texts, max_docs))
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_proc):
        tags = pos_sequence(doc, keep_punct=keep_punct, remove_stop=remove_stop)
        if len(tags) < 4:
            continue
        for g in sliding_ngrams(tags, 4):
            cnt[g] += 1
    return cnt


# ============================
# Log-odds con prior Dirichlet
# ============================

def dirichlet_log_odds(counts_a: Counter, counts_b: Counter, alpha: float = 0.1) -> pd.DataFrame:
    vocab = set(counts_a) | set(counts_b)
    if not vocab:
        return pd.DataFrame(columns=["pos4", "count_a", "count_b", "delta", "z"])

    idx = {g: i for i, g in enumerate(vocab)}
    A = np.zeros(len(vocab), dtype=np.float64)
    B = np.zeros(len(vocab), dtype=np.float64)
    for g, c in counts_a.items():
        A[idx[g]] = c
    for g, c in counts_b.items():
        B[idx[g]] = c

    alpha_vec = np.full_like(A, float(alpha))
    alpha0 = alpha_vec.sum()
    NA = A.sum()
    NB = B.sum()
    if NA == 0 and NB == 0:
        return pd.DataFrame(columns=["pos4", "count_a", "count_b", "delta", "z"])

    numA = A + alpha_vec
    denA = (NA + alpha0) - numA
    numB = B + alpha_vec
    denB = (NB + alpha0) - numB
    denA[denA <= 0] = 1e-9
    denB[denB <= 0] = 1e-9

    delta = np.log(numA) - np.log(denA) - (np.log(numB) - np.log(denB))
    var = 1.0/numA + 1.0/denA + 1.0/numB + 1.0/denB
    z = delta / np.sqrt(var)

    return pd.DataFrame({
        "pos4": list(vocab),
        "count_a": A.astype(int),
        "count_b": B.astype(int),
        "delta": delta,
        "z": z
    }).sort_values("z", ascending=False, kind="mergesort")


# ============================
# Probabilidades y JSD
# ============================

def counter_to_prob(counter: Counter) -> dict:
    total = sum(counter.values())
    if total == 0:
        return {k: 0.0 for k in counter}
    return {k: v / total for k, v in counter.items()}

def jensen_shannon_divergence(p: dict, q: dict) -> float:
    vocab = set(p) | set(q)
    p_vec = np.array([p.get(g, 0.0) for g in vocab])
    q_vec = np.array([q.get(g, 0.0) for g in vocab])
    return jensenshannon(p_vec, q_vec, base=2.0) ** 2

def jsd_per_token_contrib(p: dict, q: dict, base=2.0):
    vocab = set(p) | set(q)
    contrib = {}
    for g in vocab:
        pg, qg = p.get(g, 0.0), q.get(g, 0.0)
        mg = 0.5 * (pg + qg)
        c = 0.0
        if pg > 0 and mg > 0:
            c += 0.5 * pg * (np.log(pg/mg) / np.log(base))
        if qg > 0 and mg > 0:
            c += 0.5 * qg * (np.log(qg/mg) / np.log(base))
        contrib[g] = c
    return contrib


# ============================
# Ejemplos textuales
# ============================

def contains_pos4(doc, target4, keep_punct=False, remove_stop=False):
    tags = pos_sequence(doc, keep_punct=keep_punct, remove_stop=remove_stop)
    for i in range(len(tags)-3):
        if tuple(tags[i:i+4]) == target4:
            return True
    return False

def sample_texts_with_pos4(nlp, texts, target4, k=3, keep_punct=False, remove_stop=False):
    ex = []
    for doc in nlp.pipe(texts, batch_size=64):
        if contains_pos4(doc, target4, keep_punct=keep_punct, remove_stop=remove_stop):
            ex.append(doc.text)
            if len(ex) >= k:
                break
    return ex


# ============================
# Plotting
# ============================

def _label_4gram(g):
    return "–".join(g)

def plot_bar_top(items, title, out_path: Path, xlabel="conteos / z-score"):
    if not items:
        return
    labels, vals = zip(*items)
    plt.figure(figsize=(10, max(3, 0.4*len(items))))
    y = np.arange(len(items))
    plt.barh(y, vals)
    plt.yticks(y, labels)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ============================
# Runner principal
# ============================

def main():
    ap = argparse.ArgumentParser(description="4-gramas POS por clase (frecuencia, log-odds, JSD, ejemplos).")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--class-col", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--remove-stop", action="store_true")
    ap.add_argument("--keep-punct", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--model", default="es_core_news_md")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n-proc", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)
    save_json(vars(args), save_dir / "run_meta.json")

    print("Cargando spaCy...")
    nlp = spacy.load(args.model, disable=["ner"])
    df = pd.read_csv(csv_path)
    df = df.copy()
    df[args.text_col] = df[args.text_col].astype(str)
    df[args.class_col] = df[args.class_col].astype(str).fillna("NA")

    # Conteos por clase
    class_values = sorted(df[args.class_col].unique())
    class_counts = {}
    total_counts = Counter()
    for val in class_values:
        texts = df[df[args.class_col] == val][args.text_col].tolist()
        cnt = extract_pos4_counts(nlp, texts,
                                  keep_punct=args.keep_punct,
                                  remove_stop=args.remove_stop,
                                  n_proc=args.n_proc,
                                  batch_size=args.batch_size,
                                  max_docs=args.max_docs)
        class_counts[val] = cnt
        total_counts.update(cnt)

    # ============================
    # Frecuencia (MLE)
    # ============================
    print("\n== TOP por frecuencia (por clase) ==")
    for val in class_values:
        cnt = class_counts[val]
        top = cnt.most_common(args.top_n)
        print(f"\n-- {args.class_col}={val} --")
        for g, c in top:
            print(f"{g}: {c} (p={c/sum(cnt.values()):.4f})")
        df_top = pd.DataFrame([{"pos4": g, "count": c, "prob": c/sum(cnt.values())} for g, c in top])
        df_top.to_csv(save_dir / f"{slugify(args.class_col)}__{slugify(val)}__top_freq_pos4.csv",
                      index=False, encoding="utf-8")
        if args.plots and top:
            items = [(_label_4gram(g), c) for g, c in top]
            out_png = save_dir / f"{slugify(args.class_col)}__{slugify(val)}__top_freq_pos4.png"
            plot_bar_top(items, title=f"Top {args.top_n} POS 4-gramas por frecuencia\n{args.class_col}={val}",
                         out_path=out_png, xlabel="conteo")

    # ============================
    # Log-odds
    # ============================
    print("\n== TOP por log-odds (discriminatividad, clase vs. resto) ==")
    for val in class_values:
        a = class_counts[val]
        b = total_counts.copy()
        for k, v in a.items():
            b[k] -= v
            if b[k] < 0:
                b[k] = 0

        df_log = dirichlet_log_odds(a, b, alpha=args.alpha)
        df_top = df_log.head(args.top_n).copy()
        print(f"\n-- {args.class_col}={val} --")
        for _, row in df_top.iterrows():
            print(f"{tuple(row['pos4'])}: z={row['z']:.2f}, count_a={int(row['count_a'])}, count_b={int(row['count_b'])}")
        out_csv = save_dir / f"{slugify(args.class_col)}__{slugify(val)}__top_logodds_pos4.csv"
        df_top.to_csv(out_csv, index=False, encoding="utf-8")
        if args.plots and not df_top.empty:
            items = [(_label_4gram(tuple(p)), float(z)) for p, z in zip(df_top["pos4"], df_top["z"])]
            out_png = save_dir / f"{slugify(args.class_col)}__{slugify(val)}__top_logodds_pos4.png"
            plot_bar_top(items, title=f"Top {args.top_n} POS 4-gramas por log-odds (z)\n{args.class_col}={val}",
                         out_path=out_png, xlabel="z-score (log-odds)")

    # ============================
    # JSD + contribuyentes + ejemplos
    # ============================
    print("\n== Divergencia Jensen–Shannon entre clases ==")
    probs_by_class = {val: counter_to_prob(cnt) for val, cnt in class_counts.items()}
    rows, ex_rows = [], []
    for i in range(len(class_values)):
        for j in range(i+1, len(class_values)):
            c1, c2 = class_values[i], class_values[j]
            p, q = probs_by_class[c1], probs_by_class[c2]
            jsd = jensen_shannon_divergence(p, q)
            print(f"  JSD({c1}, {c2}) = {jsd:.4f}")
            rows.append({"class1": c1, "class2": c2, "jsd": jsd})
            # top contribuyentes
            contrib = jsd_per_token_contrib(p, q)
            topK = sorted(contrib.items(), key=lambda x: x[1], reverse=True)[:10]
            texts_c1 = df[df[args.class_col]==c1][args.text_col].tolist()
            texts_c2 = df[df[args.class_col]==c2][args.text_col].tolist()
            for g, cval in topK:
                ex1 = sample_texts_with_pos4(nlp, texts_c1, g, k=2,
                                             keep_punct=args.keep_punct, remove_stop=args.remove_stop)
                ex2 = sample_texts_with_pos4(nlp, texts_c2, g, k=2,
                                             keep_punct=args.keep_punct, remove_stop=args.remove_stop)
                ex_rows.append({
                    "class1": c1, "class2": c2,
                    "pos4": "–".join(g),
                    "contrib": cval,
                    "p_c1": p.get(g,0.0), "p_c2": q.get(g,0.0),
                    "examples_c1": " ||| ".join(ex1),
                    "examples_c2": " ||| ".join(ex2)
                })
    pd.DataFrame(rows).to_csv(save_dir / "pairwise_jsd_classes.csv", index=False, encoding="utf-8")
    pd.DataFrame(ex_rows).to_csv(save_dir / "pairwise_jsd_top_examples.csv", index=False, encoding="utf-8")
    print(f"\nListo. Resultados en: {save_dir.resolve()}")


if __name__ == "__main__":
    main()




