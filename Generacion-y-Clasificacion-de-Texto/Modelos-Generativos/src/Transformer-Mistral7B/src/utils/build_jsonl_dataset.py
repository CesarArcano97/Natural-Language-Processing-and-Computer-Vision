#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construye un dataset JSONL a partir de las canciones delimitadas con <|startsong|> y <|endsong|>.
"""

import json, os

in_file = os.path.expanduser("~/mistral-project/data/processed/twenty_one_pilots.txt")
out_file = os.path.expanduser("~/mistral-project/data/processed/lyrics_train.jsonl")

songs = []
current = []
inside_song = False

with open(in_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if "<|startsong|>" in line:
            current = []
            inside_song = True
        elif "<|endsong|>" in line:
            if current:
                songs.append("\n".join(current).strip())
            inside_song = False
        elif inside_song:
            current.append(line)

with open(out_file, "w", encoding="utf-8") as f:
    for song in songs:
        f.write(json.dumps({"text": song}) + "\n")

print(f"✅ Dataset guardado en {out_file} con {len(songs)} canciones.")
