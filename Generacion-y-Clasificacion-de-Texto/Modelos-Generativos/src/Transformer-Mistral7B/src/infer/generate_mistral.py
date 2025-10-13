#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generación de letras al estilo Twenty One Pilots (modelo afinado con LoRA)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os

# --- Rutas ---
base_model_path = os.path.expanduser("~/mistral-project/models/base/Mistral-7B-Instruct-v0.2")
lora_model_path = os.path.expanduser("~/mistral-project/models/finetuned/mistral_twentyonepilots")

# --- Cargar tokenizer desde el modelo base ---
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# --- Cargar modelo base y luego aplicar adaptador LoRA ---
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, lora_model_path)
model = model.merge_and_unload()  # Fusiona LoRA con el modelo base (opcional)

# --- Prompts para generación ---
prompts = [
    "In the city lights I find myself",
    "They say I'm broken but I'm breathing",
    "Sometimes my shadow sings louder than I do"
]

for i, prompt in enumerate(prompts, 1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.9,
        top_p=0.95,
        do_sample=True
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    os.makedirs("results/generations", exist_ok=True)
    with open(f"results/generations/song_{i}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n🎵 Canción {i} generada:\n{text[:500]}...\n")
