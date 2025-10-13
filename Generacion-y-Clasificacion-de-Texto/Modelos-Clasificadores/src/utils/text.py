# -*- coding: utf-8 -*-
import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")

# tokens con acentos; mantenemos signos básicos como tokens
_TOKEN_RE = re.compile(
    r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:['’\-][A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)*|[.,!?;:()\"—-]"
)

def basic_clean(text: str) -> str:
    text = text.strip()
    text = _URL_RE.sub(" URL ", text)
    text = _WS_RE.sub(" ", text)
    return text

def tokenize_es(text: str):
    text = basic_clean(text)
    text = text.lower()
    return _TOKEN_RE.findall(text)
