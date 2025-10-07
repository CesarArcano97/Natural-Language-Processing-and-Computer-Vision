#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
nlp_metrics.py
Conteo de documentos, tokens y vocabulario, hapax y % stopwords
con spaCy + tokenizador social para reseñas.

Uso:
  python nlp_metrics.py --csv /ruta/MeIA.csv --text-col Review \
      --save-hapax hapax.csv --save-stop stopwords_usadas.csv

Requiere: pandas, spacy
"""

import re
import sys
import argparse
import unicodedata
from pathlib import Path
from collections import Counter

import pandas as pd
import spacy
from spacy.tokens import Token
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex, compile_infix_regex
from spacy.symbols import ORTH
from spacy.language import Language


# ===============================
# Utilidades generales
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
# Stopwords (spaCy es) + acentos
# ===============================

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def build_stopwords(nlp, normalize_accents: bool = True,
                    extra_add=None, extra_del=None) -> set:
    """
    Crea el conjunto de stopwords a usar.
    - normalize_accents: si True, compara sin tildes.
    - extra_add/extra_del: iterables de palabras para agregar/quitar.
    """
    if extra_add is None: extra_add = set()
    if extra_del is None: extra_del = set()

    base = set(w.lower() for w in nlp.Defaults.stop_words)
    if normalize_accents:
        base = set(strip_accents(w) for w in base)
        extra_add = {strip_accents(w.lower()) for w in extra_add}
        extra_del = {strip_accents(w.lower()) for w in extra_del}
    else:
        extra_add = {w.lower() for w in extra_add}
        extra_del = {w.lower() for w in extra_del}

    base |= extra_add
    base -= extra_del
    return base


def is_stopword_form(canon: str, stop_set: set, normalize_accents: bool = True) -> bool:
    key = strip_accents(canon) if normalize_accents else canon
    return key in stop_set


# ===============================
# Cómputo de métricas
# ===============================

def compute_metrics(texts: pd.Series,
                    batch_size: int = 1000,
                    n_process: int = 1,
                    normalize_accents: bool = True,
                    extra_stop_add=None,
                    extra_stop_del=None):
    """
    Procesa textos y devuelve un dict con:
      - basic: filas totales válidas, tokens (N), vocab (|V|)
      - hapax: lista y proporciones
      - stop: tokens_stop y %stop
      - counter: Counter de tokens
      - vocab_set: set del vocabulario
    """
    nlp = build_nlp()
    stop_set = build_stopwords(nlp, normalize_accents, extra_stop_add, extra_stop_del)

    total_tokens = 0
    valid_docs = 0
    vocab = set()
    counter = Counter()
    stop_tokens = 0

    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        toks = []
        for t in doc:
            if not is_valid_token(t):
                continue
            canon = canonical_form(t)
            toks.append(canon)
            if is_stopword_form(canon, stop_set, normalize_accents):
                stop_tokens += 1

        if toks:
            valid_docs += 1
            total_tokens += len(toks)
            vocab.update(toks)
            counter.update(toks)

    hapax = [w for w, c in counter.items() if c == 1]
    V = len(vocab)
    N = total_tokens
    pct_stop = (stop_tokens / N * 100.0) if N > 0 else 0.0
    prop_hapax_tipo = (len(hapax) / V) if V > 0 else 0.0
    prop_hapax_token = (len(hapax) / N) if N > 0 else 0.0

    return {
        "basic": {"valid_docs": valid_docs, "N_tokens": N, "Vocab_size": V},
        "hapax": {
            "num_hapax": len(hapax),
            "hapax_list": hapax,
            "prop_tipo": prop_hapax_tipo,
            "prop_token": prop_hapax_token
        },
        "stop": {"stop_tokens": stop_tokens, "pct_stop": pct_stop},
        "counter": counter,
        "vocab_set": vocab,
        "stop_set": stop_set
    }


def print_report(meta: dict, metrics: dict, text_col: str):
    print("===== Métricas básicas =====")
    print(f"Filas totales (CSV): {meta.get('total_rows', 'NA')}")
    print(f"Filas con {text_col} no nula: {meta.get('non_null', 'NA')}")
    print(f"Documentos válidos (≥1 token): {metrics['basic']['valid_docs']}")
    print(f"Número total de tokens (N): {metrics['basic']['N_tokens']}")
    print(f"Tamaño del vocabulario (|V|): {metrics['basic']['Vocab_size']}")

    print("\n===== Hapax legomena =====")
    print(f"Número de hapax (#hapax): {metrics['hapax']['num_hapax']}")
    print(f"Proporción tipo (#hapax/|V|): {metrics['hapax']['prop_tipo']:.4f}")
    print(f"Proporción token (#hapax/N): {metrics['hapax']['prop_token']:.4f}")

    print("\n===== Stopwords =====")
    print(f"Tokens stopwords: {metrics['stop']['stop_tokens']}")
    print(f"% stopwords (tokens_stop / N): {metrics['stop']['pct_stop']:.2f}%")


def save_optional_outputs(metrics: dict, save_hapax: Path = None, save_stop: Path = None):
    if save_hapax:
        pd.Series(sorted(metrics["hapax"]["hapax_list"])).to_csv(save_hapax, index=False)
        print(f"[guardado] hapax → {save_hapax}")
    if save_stop:
        pd.Series(sorted(metrics["stop_set"])).to_csv(save_stop, index=False)
        print(f"[guardado] stopwords usadas → {save_stop}")


# ===============================
# CLI
# ===============================

def build_arg_parser():
    p = argparse.ArgumentParser(description="Métricas NLP: tokens, vocab, hapax y % stopwords")
    p.add_argument("--csv", type=Path, required=True, help="Ruta al CSV de entrada")
    p.add_argument("--text-col", type=str, required=True, help="Nombre de la columna de texto")
    p.add_argument("--encoding", type=str, default="utf-8", help="Encoding del CSV (default: utf-8)")
    p.add_argument("--batch-size", type=int, default=1000, help="Tamaño de batch para nlp.pipe")
    p.add_argument("--n-process", type=int, default=1, help="Procesos para nlp.pipe (blank 'es' suele ir 1)")
    p.add_argument("--no-accent-norm", action="store_true",
                   help="Desactiva normalización de tildes para stopwords (por defecto está activada)")
    p.add_argument("--add-stop", type=str, nargs="*", default=[],
                   help="Palabras extra para añadir a stopwords (separadas por espacio)")
    p.add_argument("--del-stop", type=str, nargs="*", default=[],
                   help="Palabras para quitar de stopwords (separadas por espacio)")
    p.add_argument("--save-hapax", type=Path, default=None, help="Ruta para guardar lista de hapax (CSV)")
    p.add_argument("--save-stop", type=Path, default=None, help="Ruta para guardar stopwords usadas (CSV)")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    texts, meta = load_texts(args.csv, args.text_col, args.encoding)

    metrics = compute_metrics(
        texts=texts,
        batch_size=args.batch_size,
        n_process=args.n_process,
        normalize_accents=not args.no_accent_norm,
        extra_stop_add=set(args.add_stop or []),
        extra_stop_del=set(args.del_stop or []),
    )

    print_report(meta, metrics, args.text_col)
    save_optional_outputs(metrics, args.save_hapax, args.save_stop)


if __name__ == "__main__":
    sys.exit(main())
