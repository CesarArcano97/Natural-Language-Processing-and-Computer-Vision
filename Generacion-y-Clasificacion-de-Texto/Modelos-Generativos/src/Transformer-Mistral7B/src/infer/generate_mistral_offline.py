#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferencia offline con Mistral + LoRA (PEFT).
Genera letras a partir de prompts y guarda resultados con timestamp.
"""

import os, sys, datetime, torch
sys.stdout.reconfigure(line_buffering=True)

# Offline total
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# === RUTAS ===
BASE_DIR = os.path.expanduser("~/mistral-project")
MODEL_BASE = os.path.join(BASE_DIR, "models/base/Mistral-7B-Instruct-v0.2")

# Último modelo fine-tuned
finetuned_root = os.path.join(BASE_DIR, "models/finetuned")
FINETUNE_DIR = max(
    (os.path.join(finetuned_root, d) for d in os.listdir(finetuned_root)),
    key=os.path.getmtime
)
MODEL_DIR = os.path.join(FINETUNE_DIR, "final_model")

OUT_GEN = os.path.join(BASE_DIR, "results/generations")
os.makedirs(OUT_GEN, exist_ok=True)

print(f"🔹 Usando modelo base: {MODEL_BASE}")
print(f"🔹 Intentando cargar adaptadores LoRA desde: {MODEL_DIR}\n")

# === TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === MODELO BASE ===
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_BASE,
    torch_dtype=torch.float16,
    device_map="auto",
)

# === CARGA DE ADAPTADORES ===
adapter_config = os.path.join(MODEL_DIR, "adapter_config.json")
if os.path.isfile(adapter_config):
    print("Adaptadores LoRA encontrados. Cargando...")
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    print("🔹 Fusionando LoRA para inferencia...")
    model = model.merge_and_unload()
else:
    print("No se encontraron adaptadores; se usará solo el modelo base.")
    model = base_model

# === PROMPTS ===
prompts = [
    "In the city lights I find myself",
    "They say I'm broken but I'm breathing",
    "Sometimes my shadow sings louder than I do"
]

# === PARÁMETROS DE GENERACIÓN ===
gen_kwargs = dict(
    max_new_tokens=220,
    temperature=0.9,
    top_p=0.95,
    do_sample=True,
    repetition_penalty=1.1,
)

# === GENERACIÓN ===
for i, prompt in enumerate(prompts, 1):
    print(f"\nGenerando canción {i} con prompt:\n{prompt}\n{'-'*60}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Guardar resultado con timestamp
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_GEN, f"song_{i}_{ts}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Prompt: {prompt}\n\n{text}\n")

    print(f"Guardada: {out_path}\n")

print("\nGeneración completada. Revisa la carpeta results/generations/")

