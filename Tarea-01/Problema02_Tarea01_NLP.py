#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cálculo de frecuencias f(w), ranking por frecuencia y gráfica Zipf:
puntos (log r, log f) + ajuste lineal en escala log–log, con reporte de C, s y R^2.

Uso:
  python3 Problema02_Tarea01_NLP.py --csv "/home/cesar/Descargas/MeIA.csv" --text-col Review \
      --save-plot zipf_meia.png --save-csv zipf_table.csv --save-json zipf_fit.json \
      --top-k 50000 --base 10

Requiere: pandas, spacy, matplotlib, numpy
"""

import re
import sys
import math
import json
import argparse
import unicodedata
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from spacy.tokens import Token
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex, compile_infix_regex
from spacy.symbols import ORTH
from spacy.language import Language


# ===============================
# Carga de datos
# ===============================

def load_texts(csv_path: Path, text_col: str, encoding: str = "utf-8") -> pd.Series:
    df = pd.read_csv(csv_path, encoding=encoding)
    if text_col not in df.columns:
        raise ValueError(f"No encuentro la columna de texto '{text_col}' en {csv_path}")
    texts = df[text_col].dropna().astype(str)
    meta = {
        "total_rows": len(df),
        "non_null": len(texts),
    }
    return texts, meta


def try_get_url_regex():
    """URLs compatibles con varias versiones de spaCy."""
    try:
        from spacy.util import URL_MATCH as _URL_MATCH  # spaCy >= 3.1
        return _URL_MATCH
    except Exception:
        return re.compile(r"(?:https?://\S+|www\.\S+)", re.IGNORECASE)


# ===============================
# Tokenizador social + normalización
# ===============================

def make_social_tokenizer(nlp, url_re):
    prefixes = list(nlp.Defaults.prefixes)
    suffixes = list(nlp.Defaults.suffixes)
    infixes  = list(nlp.Defaults.infixes)

    # No separar '#' y '@' del resto
    prefixes = [p for p in prefixes if p not in (r"#", r"@")]

    # Menos agresivo con ciertos infijos (si existe esa regla)
    try:
        infixes.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")
    except ValueError:
        pass

    hashtag_re  = re.compile(r"#[A-Za-z0-9_]+")
    mention_re  = re.compile(r"@[A-Za-z0-9_]+")

    def token_match(text):
        # Mantener hashtags, menciones y URLs como un token
        return hashtag_re.match(text) or mention_re.match(text) or url_re.match(text)

    tokenizer = Tokenizer(
        nlp.vocab,
        rules=nlp.Defaults.tokenizer_exceptions,
        prefix_search=compile_prefix_regex(prefixes).search,
        suffix_search=compile_suffix_regex(suffixes).search,
        infix_finditer=compile_infix_regex(infixes).finditer,
        token_match=token_match,
        url_match=url_re.match
    )

    # Emoticonos comunes como un solo token
    emoticons = [":)", ":-)", ":D", ":-D", ":(", ":-(", ";)", ";-)", ":P", ":-P", "xD", "XD", "<3", ":'(", ":'-("]
    for emo in emoticons:
        tokenizer.add_special_case(emo, [{ORTH: emo}])
    return tokenizer


# Normalización de elongaciones (soooo -> soo)
ELONG_RE = re.compile(r"(.)\1{2,}", flags=re.IGNORECASE)
def collapse_elongations(text: str) -> str:
    return ELONG_RE.sub(r"\1\1", text)


def ensure_token_extension():
    if not Token.has_extension("norm_elong"):
        Token.set_extension("norm_elong", default=None)


@Language.component("elong_norm")
def elong_norm_component(doc):
    for token in doc:
        token._.norm_elong = collapse_elongations(token.text)
    return doc


def build_nlp():
    """Devuelve pipeline en blanco 'es' con tokenizador social y componente de elongación."""
    ensure_token_extension()
    url_re = try_get_url_regex()
    nlp = spacy.blank("es")
    nlp.tokenizer = make_social_tokenizer(nlp, url_re)
    nlp.add_pipe("elong_norm", first=True)
    return nlp


# ===============================
# Filtros y forma canónica
# ===============================

NUM_RE = re.compile(r"[0-9]+([.,][0-9]+)?$")

def is_valid_token(t) -> bool:
    if t.is_space or t.is_punct or t.is_quote or t.is_bracket:
        return False
    if t.like_url:
        return False
    # Excluir números puros (10, 3.14, 1,000)
    if t.like_num or NUM_RE.fullmatch(t.text):
        return False
    return True


def canonical_form(t) -> str:
    """Lower + colapso de elongaciones (si existe)"""
    base = t._.norm_elong if t._.norm_elong else t.text
    return base.lower()


# ===============================
# Frecuencias y Zipf
# ===============================

def compute_frequencies(texts: pd.Series, batch_size: int = 1000, n_process: int = 1) -> Counter:
    """
    Devuelve Counter de frecuencias f(w) reusando tokenizador social
    y reglas canónicas (lower + colapso de elongaciones).
    """
    nlp = build_nlp()
    counter = Counter()
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        for t in doc:
            if not is_valid_token(t):
                continue
            counter.update([canonical_form(t)])
    return counter


def rank_and_table(counter: Counter, top_k: int = None, base: float = 10.0) -> pd.DataFrame:
    """
    Ordena por frecuencia desc., asigna rango r = 1..K, y devuelve DataFrame con:
    palabra, freq, rank, log_rank, log_freq
    """
    items = counter.most_common(top_k) if top_k else counter.most_common()
    if len(items) == 0:
        return pd.DataFrame(columns=["word", "freq", "rank", "log_rank", "log_freq"])

    words = [w for w, _ in items]
    freqs = np.array([f for _, f in items], dtype=float)
    ranks = np.arange(1, len(freqs) + 1, dtype=float)

    # logs con base configurable
    if base == 10:
        log = np.log10
        base_label = "10"
    elif base == math.e:
        log = np.log
        base_label = "e"
    else:
        log = lambda x: np.log(x) / np.log(base)
        base_label = str(base)

    log_r = log(ranks)
    log_f = log(freqs)

    df = pd.DataFrame({
        "word": words,
        "freq": freqs.astype(int),
        "rank": ranks.astype(int),
        "log_rank": log_r,
        "log_freq": log_f
    })
    df.attrs["log_base"] = base
    df.attrs["log_base_label"] = base_label
    return df


def fit_zipf_line(df_zipf: pd.DataFrame):
    """
    Ajuste lineal y = a + b x sobre (x=log_rank, y=log_freq).
    Retorna dict con a, b, R2.
    """
    x = df_zipf["log_rank"].values
    y = df_zipf["log_freq"].values
    A = np.vstack([np.ones_like(x), x]).T
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coeffs
    y_hat = a + b * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"a": float(a), "b": float(b), "r2": float(r2)}


def plot_zipf(df_zipf: pd.DataFrame,
              title: str = "Zipf: log-rango vs log-frecuencia",
              save_path: Path | None = None,
              show_fit: bool = True):
    """Genera la gráfica Zipf a partir del DataFrame con columnas log_rank y log_freq."""
    if df_zipf.empty:
        print("[warn] DataFrame vacío: no hay puntos para graficar.")
        return

    x = df_zipf["log_rank"].values
    y = df_zipf["log_freq"].values

    plt.figure()
    plt.scatter(x, y, s=8, alpha=0.5)
    plt.xlabel("log(rango)")
    plt.ylabel("log(frecuencia)")
    plt.title(title)

    if show_fit and len(x) >= 2:
        fit = fit_zipf_line(df_zipf)
        a, b, r2 = fit["a"], fit["b"], fit["r2"]
        y_hat = a + b * x
        plt.plot(x, y_hat, linewidth=2, label=f"ajuste: y = {a:.3f} + {b:.3f} x, R²={r2:.3f}")
        plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"[guardado] Zipf plot → {save_path}")
    else:
        plt.show()


# ===============================
# CLI
# ===============================

def build_arg_parser():
    p = argparse.ArgumentParser(description="Análisis Zipf: frecuencias, ranking, ajuste y gráfica (log–log).")
    p.add_argument("--csv", type=Path, required=True, help="Ruta al CSV de entrada")
    p.add_argument("--text-col", type=str, required=True, help="Nombre de la columna de texto")
    p.add_argument("--encoding", type=str, default="utf-8", help="Encoding del CSV (default: utf-8)")
    p.add_argument("--batch-size", type=int, default=1000, help="Tamaño de batch para nlp.pipe")
    p.add_argument("--n-process", type=int, default=1, help="Procesos para nlp.pipe (blank 'es' suele ir 1)")
    p.add_argument("--top-k", type=int, default=None, help="Usar solo las top-K palabras por frecuencia (opcional)")
    p.add_argument("--base", type=float, default=10.0, help="Base del log (10 por defecto; usa 2 o e si prefieres)")
    p.add_argument("--save-plot", type=Path, default=None, help="Ruta para guardar la imagen (PNG/SVG)")
    p.add_argument("--save-csv", type=Path, default=None, help="Ruta para guardar la tabla (CSV)")
    p.add_argument("--save-json", type=Path, default=None, help="Ruta para guardar parámetros del ajuste (JSON)")
    p.add_argument("--title", type=str, default=None, help="Título de la figura")
    p.add_argument("--no-fit", action="store_true", help="No dibujar ajuste lineal en log–log")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    # 1) Cargar textos
    texts, meta = load_texts(args.csv, args.text_col, args.encoding)
    print(f"[info] Filas totales: {meta['total_rows']}, con texto no nulo: {meta['non_null']}")

    # 2) Frecuencias absolutas f(w)
    print("[info] Tokenizando y contando frecuencias...")
    counter = compute_frequencies(texts, batch_size=args.batch_size, n_process=args.n_process)
    print(f"[info] Vocabulario distinto: {len(counter)}")

    # 3) Ranking + tabla log–log
    df_zipf = rank_and_table(counter, top_k=args.top_k, base=args.base)
    if args.save_csv:
        df_zipf.to_csv(args.save_csv, index=False)
        print(f"[guardado] Tabla Zipf (word, freq, rank, log_rank, log_freq) → {args.save_csv}")

    # 4) Gráfica Zipf
    title = args.title if args.title else f"Zipf: log-rango vs log-frecuencia ({args.text_col})"
    plot_zipf(df_zipf, title=title, save_path=args.save_plot, show_fit=not args.no_fit)

    # 5) Ajuste lineal + parámetros Zipf (C y s)
    if not df_zipf.empty:
        fit = fit_zipf_line(df_zipf)
        a, b, r2 = fit["a"], fit["b"], fit["r2"]  # y = a + b x  =>  a = log C, b = -s

        # Recuperar C según la base elegida
        base = args.base
        if base == 10:
            C = 10 ** a
            base_label = "10"
        elif base == math.e:
            C = math.exp(a)
            base_label = "e"
        else:
            C = base ** a
            base_label = str(base)

        s = -b  # exponente de Zipf
        top_freq = int(df_zipf["freq"].iloc[0]) if "freq" in df_zipf.columns else None
        rel_diff = (C - top_freq) / max(1, top_freq) if top_freq is not None else float("nan")

        print("\n===== Modelo Zipf (a partir de la recta en log–log) =====")
        print(f"log_base({base_label})(C) = a = {a:.6f}")
        print(f"pendiente b = {b:.6f}  ⇒  s = -b = {s:.6f}")
        print(f"C = {C:.6f}  (≈ frecuencia de la palabra más común)")
        if top_freq is not None:
            print(f"f(1) observado = {top_freq}  |  diferencia relativa = {rel_diff:.3%}")
        print(f"R² = {r2:.4f}")
        print("\nForma del modelo:  f(r) ≈ C / r^s")

        # 6) Guardar JSON con parámetros del ajuste (opcional)
        if args.save_json:
            payload = {
                "log_base": base,
                "a": a,
                "b": b,
                "C": C,
                "s": s,
                "R2": r2,
                "f1_observado": top_freq,
                "diferencia_relativa_C_vs_f1": rel_diff
            }
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[guardado] Parámetros del ajuste → {args.save_json}")


if __name__ == "__main__":
    sys.exit(main())

