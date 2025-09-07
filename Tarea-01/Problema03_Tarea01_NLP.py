#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class_terms.py

Limpieza/normalización + conteo por clase y ranking de términos:
- Top por frecuencia (tras remover stopwords si se indica)
- Top por "discriminatividad" usando log-odds con prior informativo (Monroe et al., 2008) [ACTIVADO POR DEFECTO]
- Gráficas: barras (freq), barras (log-odds), heatmap (log-odds), dispersión (freq vs log-odds)

Uso típico:
  python3 class_terms.py --csv "/home/cesar/Descargas/MeIA.csv" \
      --text-col Review --class-col Polarity --top-n 20 \
      --remove-stop --save-dir out_polarity --plots

Requiere: pandas, numpy, spacy, matplotlib
"""

import re
import sys
import math
import argparse
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from spacy.tokens import Token
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex, compile_infix_regex
from spacy.symbols import ORTH
from spacy.language import Language

# -----------------------------
# CARGA
# -----------------------------

def load_df(csv_path: Path, text_col: str, class_col: str, encoding: str = "utf-8") -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding=encoding)
    for col in [text_col, class_col]:
        if col not in df.columns:
            raise ValueError(f"No encuentro la columna '{col}' en {csv_path}")
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df.dropna(subset=[text_col, class_col])
    return df

def try_get_url_regex():
    try:
        from spacy.util import URL_MATCH as _URL_MATCH  # spaCy >= 3.1
        return _URL_MATCH
    except Exception:
        return re.compile(r"(?:https?://\S+|www\.\S+)", re.IGNORECASE)

# -----------------------------
# TOKENIZACIÓN SOCIAL + NORMALIZACIÓN
# -----------------------------

def make_social_tokenizer(nlp, url_re):
    prefixes = list(nlp.Defaults.prefixes)
    suffixes = list(nlp.Defaults.suffixes)
    infixes  = list(nlp.Defaults.infixes)
    # # y @ pegados a la palabra
    prefixes = [p for p in prefixes if p not in (r"#", r"@")]
    try:
        infixes.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")
    except ValueError:
        pass

    hashtag_re  = re.compile(r"#[A-Za-z0-9_]+")
    mention_re  = re.compile(r"@[A-Za-z0-9_]+")

    def token_match(text):
        return hashtag_re.match(text) or mention_re.match(text) or url_re.match(text)

    tok = Tokenizer(
        nlp.vocab,
        rules=nlp.Defaults.tokenizer_exceptions,
        prefix_search=compile_prefix_regex(prefixes).search,
        suffix_search=compile_suffix_regex(suffixes).search,
        infix_finditer=compile_infix_regex(infixes).finditer,
        token_match=token_match,
        url_match=url_re.match
    )
    emoticons = [":)", ":-)", ":D", ":-D", ":(", ":-(", ";)", ";-)", ":P", ":-P", "xD", "XD", "<3", ":'(", ":'-("]
    for emo in emoticons:
        tok.add_special_case(emo, [{ORTH: emo}])
    return tok

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
    ensure_token_extension()
    url_re = try_get_url_regex()
    nlp = spacy.blank("es")
    nlp.tokenizer = make_social_tokenizer(nlp, url_re)
    nlp.add_pipe("elong_norm", first=True)
    return nlp

NUM_RE = re.compile(r"[0-9]+([.,][0-9]+)?$")
def is_valid_token(t) -> bool:
    if t.is_space or t.is_punct or t.is_quote or t.is_bracket:
        return False
    if t.like_url:
        return False
    if t.like_num or NUM_RE.fullmatch(t.text):
        return False
    return True

def canonical_form(t) -> str:
    base = t._.norm_elong if t._.norm_elong else t.text
    return base.lower()

# -----------------------------
# STOPWORDS (opcional)
# -----------------------------

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def build_stopwords(nlp, normalize_accents=True, extra_add=None, extra_del=None) -> set:
    if extra_add is None: extra_add = set()
    if extra_del is None: extra_del = set()
    base = set(w.lower() for w in nlp.Defaults.stop_words)
    if normalize_accents:
        base = {strip_accents(w) for w in base}
        extra_add = {strip_accents(w.lower()) for w in extra_add}
        extra_del = {strip_accents(w.lower()) for w in extra_del}
    else:
        extra_add = {w.lower() for w in extra_add}
        extra_del = {w.lower() for w in extra_del}
    base |= extra_add
    base -= extra_del
    return base

# -----------------------------
# TOKENS POR DOCUMENTO
# -----------------------------

def is_stop_word(w: str, stop_set: set, norm_accents: bool) -> bool:
    key = strip_accents(w) if norm_accents else w
    return key in stop_set

# -----------------------------
# CONTADORES POR CLASE
# -----------------------------

def build_class_counters(df: pd.DataFrame, text_col: str, class_col: str,
                         remove_stop=True, extra_stop_add=None, extra_stop_del=None,
                         norm_accents=True, batch_size=1000, n_process=1):
    nlp = build_nlp()
    stop_set = build_stopwords(nlp, normalize_accents=norm_accents,
                               extra_add=set(extra_stop_add or []),
                               extra_del=set(extra_stop_del or [])) if remove_stop else None

    counters = defaultdict(Counter)
    n_tokens_by_class = Counter()
    n_docs_by_class = Counter()

    for (cls, text), doc in zip(df[[class_col, text_col]].itertuples(index=False, name=None),
                                nlp.pipe(df[text_col].tolist(), batch_size=batch_size, n_process=n_process)):
        toks = []
        for t in doc:
            if not is_valid_token(t):
                continue
            w = canonical_form(t)
            if remove_stop and stop_set is not None and is_stop_word(w, stop_set, norm_accents):
                continue
            toks.append(w)
        if toks:
            counters[cls].update(toks)
            n_tokens_by_class[cls] += len(toks)
            n_docs_by_class[cls] += 1

    return counters, n_tokens_by_class, n_docs_by_class

# -----------------------------
# TOP POR FRECUENCIA
# -----------------------------

def top_freq_by_class(counters: dict, top_n=20) -> dict:
    out = {}
    for cls, ctr in counters.items():
        out[cls] = ctr.most_common(top_n)
    return out

# -----------------------------
# LOG-ODDS CON PRIOR INFORMATIVO
# (Monroe, Colaresi & Quinn, 2008)
# -----------------------------

def logodds_informative_priors(counters: dict, alpha0: float = 1.0, min_count: int = 2):
    classes = list(counters.keys())
    n_k = {k: sum(counters[k].values()) for k in classes}
    pool = Counter()
    for k in classes:
        pool.update(counters[k])
    n_pool = sum(pool.values())
    p_w = {w: c / n_pool for w, c in pool.items()}

    rows = []
    for k in classes:
        n_rest = sum(n_k.values()) - n_k[k]
        ctr_k = counters[k]
        for w, c_pool in pool.items():
            if (ctr_k[w] + (c_pool - ctr_k[w])) < min_count:
                continue
            c_k = ctr_k[w]
            c_rest = c_pool - ctr_k[w]
            alpha_w = alpha0 * p_w[w]

            num_k = c_k + alpha_w
            den_k = (n_k[k] - c_k) + (alpha0 - alpha_w)
            num_r = c_rest + alpha_w
            den_r = (n_rest - c_rest) + (alpha0 - alpha_w)
            if min(num_k, den_k, num_r, den_r) <= 0:
                continue

            delta = math.log(num_k/den_k) - math.log(num_r/den_r)
            var   = 1.0/(num_k) + 1.0/(num_r)
            z     = delta / math.sqrt(var) if var > 0 else 0.0
            rows.append({
                "class": k,
                "word": w,
                "count": int(c_k),
                "p_hat": c_k / n_k[k] if n_k[k] > 0 else 0.0,
                "log_odds": float(delta),
                "z": float(z)
            })
    df = pd.DataFrame(rows).sort_values(["class", "z"], ascending=[True, False]).reset_index(drop=True)
    return df

# -----------------------------
# GRÁFICAS
# -----------------------------

def _safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))

def plot_top_freq_bars(topfreq: dict, outdir: Path, top_n: int):
    outdir.mkdir(parents=True, exist_ok=True)
    for cls, items in topfreq.items():
        words = [w for w, _ in items][:top_n]
        counts = [c for _, c in items][:top_n]
        plt.figure()
        plt.barh(range(len(words))[::-1], counts[::-1])
        plt.yticks(range(len(words))[::-1], words[::-1])
        plt.xlabel("frecuencia")
        plt.title(f"Top frecuencia — {cls}")
        plt.tight_layout()
        path = outdir / f"plot_top_freq_{_safe_filename(cls)}_top{top_n}.png"
        plt.savefig(path, dpi=160)
        plt.close()

def plot_top_logodds_bars(df_log: pd.DataFrame, outdir: Path, top_n: int):
    outdir.mkdir(parents=True, exist_ok=True)
    for cls, dfc in df_log.groupby("class"):
        head = dfc.head(top_n)
        words = head["word"].tolist()[::-1]
        zs = head["z"].tolist()[::-1]
        plt.figure()
        plt.barh(range(len(words)), zs)
        plt.yticks(range(len(words)), words)
        plt.xlabel("z (log-odds informativo)")
        plt.title(f"Top discriminativos — {cls}")
        plt.tight_layout()
        path = outdir / f"plot_top_logodds_{_safe_filename(cls)}_top{top_n}.png"
        plt.savefig(path, dpi=160)
        plt.close()

def plot_logodds_heatmap(df_log: pd.DataFrame, outdir: Path, per_class_top: int = 15):
    """
    Heatmap de z-scores de log-odds.
    Toma los per_class_top términos más distintivos de cada clase y arma
    una matriz [clase x término] con z (faltantes = 0).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    selected = []
    for cls, dfc in df_log.groupby("class"):
        selected += dfc.head(per_class_top)["word"].tolist()
    selected = sorted(set(selected))
    mat = df_log[df_log["word"].isin(selected)].pivot_table(index="class", columns="word", values="z", fill_value=0.0)
    plt.figure(figsize=(min(18, 0.5*mat.shape[1] + 3), 0.5*mat.shape[0] + 3))
    plt.imshow(mat.values, aspect="auto")
    plt.colorbar(label="z (log-odds)")
    plt.yticks(range(mat.shape[0]), mat.index.tolist())
    plt.xticks(range(mat.shape[1]), mat.columns.tolist(), rotation=90)
    plt.title(f"Heatmap de términos distintivos (top {per_class_top} por clase)")
    plt.tight_layout()
    path = outdir / f"plot_heatmap_logodds_top{per_class_top}.png"
    plt.savefig(path, dpi=160)
    plt.close()

def plot_freq_vs_logodds(df_log: pd.DataFrame, counters: dict, outdir: Path, per_class_top: int = 200):
    """
    Dispersión: frecuencia (x, en log10) vs z (y) por clase.
    Toma los top per_class_top términos por z para visualizar.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    for cls, dfc in df_log.groupby("class"):
        head = dfc.head(per_class_top).copy()
        # obtener frecuencia cruda del término en la clase
        head["freq"] = head["word"].map(lambda w: counters[cls][w])
        head["log10_freq"] = np.log10(head["freq"].clip(lower=1))
        plt.figure()
        plt.scatter(head["log10_freq"].values, head["z"].values, s=16, alpha=0.7)
        for _, r in head.nlargest(5, "z").iterrows():
            plt.annotate(r["word"], (r["log10_freq"], r["z"]), xytext=(3,3), textcoords="offset points", fontsize=8)
        plt.xlabel("log10(frecuencia en la clase)")
        plt.ylabel("z (log-odds)")
        plt.title(f"Frecuencia vs. discriminatividad — {cls}")
        plt.tight_layout()
        path = outdir / f"plot_scatter_freq_vs_logodds_{_safe_filename(cls)}.png"
        plt.savefig(path, dpi=160)
        plt.close()

# -----------------------------
# I/O RESULTADOS (CSV + previews)
# -----------------------------

def save_top_freq(topfreq: dict, outdir: Path, top_n: int):
    outdir.mkdir(parents=True, exist_ok=True)
    for cls, items in topfreq.items():
        df = pd.DataFrame(items, columns=["word", "count"])
        df.to_csv(outdir / f"top_freq_{_safe_filename(cls)}_top{top_n}.csv", index=False)

def save_logodds(df_log: pd.DataFrame, outdir: Path, top_n: int):
    outdir.mkdir(parents=True, exist_ok=True)
    for cls, dfc in df_log.groupby("class"):
        dfc.head(top_n).to_csv(outdir / f"top_logodds_{_safe_filename(cls)}_top{top_n}.csv", index=False)
    df_log.to_csv(outdir / "logodds_all.csv", index=False)

def pretty_print_top(topfreq: dict, title="Top por frecuencia"):
    print(f"\n===== {title} =====")
    for cls, items in topfreq.items():
        preview = ", ".join([f"{w}({c})" for w, c in items[:10]])
        print(f"[{cls}] {preview}")

def pretty_print_logodds(df_log: pd.DataFrame, top_n: int):
    print("\n===== Top discriminativos (log-odds con prior) =====")
    for cls, dfc in df_log.groupby("class"):
        head = dfc.head(top_n)
        preview = ", ".join([f"{r.word}(z={r.z:.2f})" for r in head.itertuples(index=False)])
        print(f"[{cls}] {preview}")

# -----------------------------
# CLI
# -----------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(description="Términos por clase: frecuencia y discriminatividad (log-odds).")
    
    # --- Se corrigió esta sección ---
    p.add_argument("--csv", type=Path, required=True, help="Ruta al CSV")
    p.add_argument("--text-col", type=str, required=True, help="Columna de texto (p. ej., Review)")
    
    p.add_argument("--class-col", type=str, required=True,
                   choices=["Polarity", "Region", "Type"],
                   help="Columna de clase a analizar")
    p.add_argument("--encoding", type=str, default="utf-8")
    p.add_argument("--top-n", type=int, default=20, help="Top N a reportar por clase")
    p.add_argument("--remove-stop", action="store_true", help="Quitar stopwords (recomendado)")
    p.add_argument("--no-accent-norm", action="store_true", help="No normalizar tildes para stopwords")
    p.add_argument("--add-stop", type=str, nargs="*", default=[], help="Stopwords extra a añadir")
    p.add_argument("--del-stop", type=str, nargs="*", default=[], help="Stopwords a quitar")
    # Log-odds SIEMPRE se calcula; puedes ajustar estos parámetros:
    p.add_argument("--alpha0", type=float, default=1.0, help="Fuerza del prior informativo (log-odds)")
    p.add_argument("--min-count", type=int, default=2, help="Mínimo de ocurrencias totales para evaluar log-odds")
    # Salida
    p.add_argument("--save-dir", type=Path, default=None, help="Directorio donde guardar CSVs y figuras")
    p.add_argument("--plots", action="store_true", help="Generar figuras (barras, heatmap, dispersión)")
    return p

# -----------------------------
# MAIN
# -----------------------------

def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    df = load_df(args.csv, args.text_col, args.class_col, args.encoding)
    print(f"[info] {len(df)} filas; clases únicas en {args.class_col}: {df[args.class_col].nunique()}")

    counters, n_tok, n_docs = build_class_counters(
        df, args.text_col, args.class_col,
        remove_stop=args.remove_stop,
        extra_stop_add=args.add_stop,
        extra_stop_del=args.del_stop,
        norm_accents=not args.no_accent_norm
    )

    # Top por frecuencia
    topfreq = top_freq_by_class(counters, top_n=args.top_n)
    pretty_print_top(topfreq, title=f"Top por frecuencia (stopwords {'removidas' if args.remove_stop else 'incluidas'})")

    # Log-odds (discriminatividad) — SIEMPRE activado
    df_log = logodds_informative_priors(counters, alpha0=args.alpha0, min_count=args.min_count)
    pretty_print_logodds(df_log, top_n=args.top_n)

    # Guardado CSVs
    if args.save_dir:
        save_top_freq(topfreq, args.save_dir, args.top_n)
        save_logodds(df_log, args.save_dir, args.top_n)

    # Figuras
    if args.plots:
        outdir = args.save_dir or Path("./figs_out")
        outdir.mkdir(parents=True, exist_ok=True)
        plot_top_freq_bars(topfreq, outdir, args.top_n)
        plot_top_logodds_bars(df_log, outdir, args.top_n)
        plot_logodds_heatmap(df_log, outdir, per_class_top=min(15, args.top_n))
        plot_freq_vs_logodds(df_log, counters, outdir, per_class_top=max(50, args.top_n))

    # Nota analítica
    print("\n[nota] Frecuencia ≠ discriminatividad. Consulta las barras de log-odds y el heatmap para ver términos realmente distintivos por clase.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
