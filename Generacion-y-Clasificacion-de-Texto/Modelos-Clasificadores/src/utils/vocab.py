# -*- coding: utf-8 -*-
# src/utils/vocab.py
from collections import Counter
import json

PAD, UNK = "<pad>", "<unk>"

def build_vocab(tokenized_texts, min_freq=2, specials=(PAD, UNK)):
    """
    tokenized_texts: iterable de listas de tokens
    min_freq: frecuencia mínima para incluir en vocab
    specials: tuplas de tokens especiales que van primero
    Devuelve: stoi (dict token->id), itos (lista id->token)
    """
    cnt = Counter()
    for toks in tokenized_texts:
        cnt.update(toks)

    itos = list(specials)  # specials primero
    for tok, c in cnt.most_common():
        if c >= min_freq and tok not in specials:
            itos.append(tok)
    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi, itos

def save_vocab(stoi, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False, indent=2)
