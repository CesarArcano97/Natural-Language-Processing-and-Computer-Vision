#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning de Mistral-7B en corpus de canciones (Twenty One Pilots)
Usa Unsloth + LoRA (~1% de los parámetros entrenados)
"""

import os, sys
sys.stdout.reconfigure(line_buffering=True)

# === Modo offline obligatorio ===
os.environ["UNSLOTH_FORCE_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# === Config ===
model_path = os.path.expanduser("~/mistral-project/models/base/Mistral-7B-Instruct-v0.2")
data_path  = os.path.expanduser("~/mistral-project/data/processed/lyrics_train.jsonl")
out_dir    = os.path.expanduser("~/mistral-project/models/finetuned/mistral_twentyonepilots")

max_seq_length = 1024
batch_size = 2
num_epochs = 3
lr = 2e-4

os.makedirs(out_dir, exist_ok=True)

# === Cargar modelo base ===
print("🔹 Cargando modelo base Mistral...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# Asegurar pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === Aplicar LoRA (solo ~1% de params) ===
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

# === Dataset ===
print("🔹 Cargando dataset...")
dataset = load_dataset("json", data_files={"train": data_path})["train"]

# === Argumentos de entrenamiento ===
args = TrainingArguments(
    per_device_train_batch_size = batch_size,
    gradient_accumulation_steps = 4,
    num_train_epochs = num_epochs,
    learning_rate = lr,
    logging_steps = 10,
    output_dir = out_dir,
    save_strategy = "epoch",
    fp16 = True,
    bf16 = False,
    report_to = "none",
)

# === Entrenador ===
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    args = args,
)

print("Iniciando entrenamiento...")
train_output = trainer.train()

# 🔹 Forzar guardado del adaptador LoRA
print("Guardando adaptador LoRA y tokenizer...")
trainer.save_model(out_dir)
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)

# 🔹 Guardar métricas
with open(os.path.join(out_dir, "train_results.txt"), "w") as f:
    f.write(str(train_output.metrics))

print(f"Modelo afinado guardado correctamente en {out_dir} :)")
