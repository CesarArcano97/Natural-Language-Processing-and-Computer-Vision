#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_corpus.py
Crea datasets de lenguaje (X, y) para entrenamiento a nivel carácter o palabra.
Guarda: vocab.json, meta.json, train.pt, val.pt bajo data/processed/{char|word}/
"""

import argparse
import json
import math
import os
import re
import unicodedata
from collections import Counter
from typing import List, Dict, Tuple

import torch
from torch.utils.data import TensorDataset

# ----------------------------
# Constantes
# ----------------------------
START_SONG = "<|startsong|>"
END_SONG = "<|endsong|>"

SPECIAL_TOKENS = {
    "PAD": "<PAD>",
    "UNK": "<UNK>",
    "BOS": "<BOS>",
    "EOS": "<EOS>",
    "START_SONG": START_SONG,
    "END_SONG": END_SONG,
}

PAD_ID = 0  # asumiremos <PAD>=0 al construir vocab

# ----------------------------
# Limpieza / normalización
# ----------------------------
def normalize_and_clean_text(text: str) -> str:
    """
    Normaliza, limpia y prepara el texto del corpus.
    - Separa con espacios los delimitadores <|startsong|>, <|endsong|>
    - Normaliza Unicode NFKC
    - Estandariza saltos de línea
    - Elimina anotaciones [Chorus], [Bridge: ...]
    - Colapsa espacios múltiples
    """
    t = text.replace(START_SONG, f" {START_SONG} ").replace(END_SONG, f" {END_SONG} ")
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\[.*?\]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def read_corpus(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return normalize_and_clean_text(f.read())

# ----------------------------
# Tokenizadores
# ----------------------------
class BaseTokenizer:
    def fit(self, text: str): ...
    def encode(self, text: str) -> List[int]: ...
    def decode(self, ids: List[int]) -> str: ...
    @property
    def stoi(self) -> Dict[str, int]: ...
    @property
    def itos(self) -> List[str]: ...
    @property
    def vocab_size(self) -> int: ...


class CharTokenizer(BaseTokenizer):
    def __init__(self, keep_song_delims: bool = True):
        self.keep_song_delims = keep_song_delims
        self._itos: List[str] = []
        self._stoi: Dict[str, int] = {}

    def fit(self, text: str):
        chars = set(text)
        if self.keep_song_delims:
            # Asegurar que los caracteres de los delimitadores estén en el vocab
            chars.update(list(START_SONG + END_SONG))
        specials = [SPECIAL_TOKENS[k] for k in ["PAD", "UNK", "BOS", "EOS"]]
        self._itos = specials + sorted(chars)
        self._stoi = {ch: i for i, ch in enumerate(self._itos)}

    def encode(self, text: str) -> List[int]:
        unk_id = self._stoi[SPECIAL_TOKENS["UNK"]]
        return [self._stoi.get(ch, unk_id) for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self._itos[i] for i in ids)

    @property
    def stoi(self): return self._stoi
    @property
    def itos(self): return self._itos
    @property
    def vocab_size(self): return len(self._itos)


class WordTokenizer(BaseTokenizer):
    def __init__(
        self,
        lowercase: bool = True,
        min_freq: int = 1,
        keep_song_delims: bool = True,
        use_spacy: bool = False,
        spacy_model: str = "en_core_web_sm",
    ):
        self.lowercase = lowercase
        self.min_freq = min_freq
        self.keep_song_delims = keep_song_delims
        self.use_spacy = use_spacy
        self.spacy_model = spacy_model
        self._nlp = None
        if self.use_spacy:
            try:
                import spacy  # import local para no requerirlo si no se usa
                self._nlp = spacy.load(self.spacy_model)
            except Exception as e:
                print(f"[WARN] No se pudo cargar spaCy ({self.spacy_model}). Fallback a tokenización por espacios. Detalle: {e}")
                self._nlp = None

        self._itos: List[str] = []
        self._stoi: Dict[str, int] = {}

    def _tokenize(self, text: str) -> List[str]:
        t = normalize_and_clean_text(text)
        if not self.keep_song_delims:
            # Si no se quieren conservar, se eliminan del texto limpio
            t = t.replace(START_SONG, " ").replace(END_SONG, " ")
            t = re.sub(r"\s+", " ", t).strip()

        if self._nlp is not None:
            # spaCy
            doc = self._nlp(t)
            toks = [tok.text for tok in doc]
        else:
            # Split simple por espacios
            toks = t.split()

        if self.lowercase:
            toks = [w.lower() for w in toks]
        return toks

    def fit(self, text: str):
        words = self._tokenize(text)
        freqs = Counter(words)

        specials = [SPECIAL_TOKENS[k] for k in ["PAD", "UNK", "BOS", "EOS"]]
        if self.keep_song_delims:
            specials += [SPECIAL_TOKENS["START_SONG"], SPECIAL_TOKENS["END_SONG"]]

        vocab = list(specials)
        for w, c in freqs.most_common():
            if w in specials:
                continue
            if c >= self.min_freq:
                vocab.append(w)

        self._itos = vocab
        self._stoi = {w: i for i, w in enumerate(self._itos)}

        # Asegurar <PAD>=0 para compatibilidad con pérdidas que ignoran índice 0
        if self._itos[0] != SPECIAL_TOKENS["PAD"]:
            # Reordenar para que PAD sea 0 (raro que no lo sea, pero por si acaso)
            if SPECIAL_TOKENS["PAD"] in self._itos:
                pad_idx = self._itos.index(SPECIAL_TOKENS["PAD"])
                self._itos[0], self._itos[pad_idx] = self._itos[pad_idx], self._itos[0]
                self._stoi = {w: i for i, w in enumerate(self._itos)}

    def encode(self, text: str) -> List[int]:
        toks = self._tokenize(text)
        unk = self._stoi[SPECIAL_TOKENS["UNK"]]
        return [self._stoi.get(w, unk) for w in toks]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self._itos[i] for i in ids)

    @property
    def stoi(self): return self._stoi
    @property
    def itos(self): return self._itos
    @property
    def vocab_size(self): return len(self._itos)

# ----------------------------
# Creación de secuencias LM
# ----------------------------
def build_lm_sequences(ids: List[int], seq_len: int, stride: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construye ventanas superpuestas para LM.
    Retorna X (N, L) e y (N, L) como tensores Long.
    """
    assert seq_len > 0 and stride > 0
    n = (len(ids) - (seq_len + 1)) // stride + 1
    if n <= 0:
        return torch.empty(0, seq_len, dtype=torch.long), torch.empty(0, seq_len, dtype=torch.long)

    X = torch.empty((n, seq_len), dtype=torch.long)
    y = torch.empty((n, seq_len), dtype=torch.long)
    for i in range(n):
        start = i * stride
        chunk = ids[start : start + seq_len + 1]
        X[i] = torch.tensor(chunk[:-1], dtype=torch.long)
        y[i] = torch.tensor(chunk[1:], dtype=torch.long)
    return X, y

# ----------------------------
# Partición y guardado
# ----------------------------
def save_artifacts(
    out_dir: str,
    tokenizer: BaseTokenizer,
    X: torch.Tensor,
    y: torch.Tensor,
    train_split: float,
    seed: int,
    meta: dict,
):
    os.makedirs(out_dir, exist_ok=True)

    N = X.size(0)
    n_train = int(math.floor(N * train_split))
    n_val = N - n_train

    # Partición CONSISTENTE entre X e y (mismo orden)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    torch.save((X_train, y_train), os.path.join(out_dir, "train.pt"))
    torch.save((X_val, y_val), os.path.join(out_dir, "val.pt"))

    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"itos": tokenizer.itos}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# ----------------------------
# Pipeline principal
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Preparar dataset LM (char/word)")
    p.add_argument("--input", type=str, default="data/raw/canciones.txt")
    p.add_argument("--level", type=str, choices=["char", "word"], required=True)
    p.add_argument("--seq_len", type=int, default=128, help="longitud del contexto")
    p.add_argument("--stride", type=int, default=1, help="desplazamiento entre ventanas")
    p.add_argument("--train_split", type=float, default=0.9)
    p.add_argument("--lowercase", action="store_true", help="(solo word) minúsculas")
    p.add_argument("--min_freq", type=int, default=1, help="(solo word) frecuencia mínima")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--keep_song_delims", action="store_true", help="conservar <|startsong|>,<|endsong|>")
    p.add_argument("--use_spacy", action="store_true", help="usar spaCy para tokenizar (word-level)")
    p.add_argument("--spacy_model", type=str, default="en_core_web_sm", help="modelo spaCy")
    args = p.parse_args()

    text = read_corpus(args.input)

    # Tokenizador
    if args.level == "char":
        tokenizer = CharTokenizer(keep_song_delims=args.keep_song_delims)
    else:
        tokenizer = WordTokenizer(
            lowercase=args.lowercase,
            min_freq=args.min_freq,
            keep_song_delims=args.keep_song_delims,
            use_spacy=args.use_spacy,
            spacy_model=args.spacy_model,
        )

    tokenizer.fit(text)
    ids = tokenizer.encode(text)

    # Insertar <BOS>/<EOS> por canción en word-level si se conservaron delimitadores
    if args.keep_song_delims and args.level == "word":
        itos = tokenizer.itos
        stoi = tokenizer.stoi
        startsong_id = stoi.get(START_SONG, None)
        endsong_id = stoi.get(END_SONG, None)
        bos_id = stoi[SPECIAL_TOKENS["BOS"]]
        eos_id = stoi[SPECIAL_TOKENS["EOS"]]
        if startsong_id is not None and endsong_id is not None:
            segmented: List[int] = []
            i = 0
            while i < len(ids):
                if ids[i] == startsong_id:
                    i += 1
                    segmented.append(bos_id)
                    while i < len(ids) and ids[i] != endsong_id:
                        segmented.append(ids[i])
                        i += 1
                    segmented.append(eos_id)
                else:
                    segmented.append(ids[i])
                i += 1
            ids = segmented

    # Secuencias LM
    X, y = build_lm_sequences(ids, seq_len=args.seq_len, stride=args.stride)

    out_dir = os.path.join("data", "processed", args.level)
    meta = {
        "level": args.level,
        "vocab_size": tokenizer.vocab_size,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "train_split": args.train_split,
        "num_examples": int(X.size(0)),
        "num_tokens_total": int(len(ids)),
        "keep_song_delims": bool(args.keep_song_delims),
        "lowercase": bool(getattr(args, "lowercase", False)),
        "min_freq": int(getattr(args, "min_freq", 1)),
        "use_spacy": bool(getattr(args, "use_spacy", False)),
        "spacy_model": str(getattr(args, "spacy_model", "")),
    }

    save_artifacts(out_dir, tokenizer, X, y, args.train_split, args.seed, meta)

    # Reporte corto
    print("=== PREPARACIÓN COMPLETADA ===")
    print(f"Nivel:            {args.level}")
    print(f"Vocab size:       {tokenizer.vocab_size}")
    print(f"Ejemplos (N):     {X.size(0)}")
    print(f"seq_len:          {args.seq_len} | stride: {args.stride}")
    print(f"Tokens totales:   {len(ids)}")
    print(f"Train split:      {args.train_split}")
    print(f"Salida:           {out_dir}/(train.pt, val.pt, vocab.json, meta.json)")
    if X.size(0) == 0:
        print("⚠️  N=0. Reduce --seq_len o aumenta corpus.")
    

if __name__ == "__main__":
    main()
