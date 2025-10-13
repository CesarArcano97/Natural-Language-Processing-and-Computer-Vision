#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning de Mistral-7B (Unsloth + LoRA) con evaluación automática.
Evalúa PPL y token accuracy en validación al final de cada época.
"""

import os, sys, time, math
sys.stdout.reconfigure(line_buffering=True)

# ─── MODO OFFLINE ──────────────────────────────────────────────────────────────
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["UNSLOTH_FORCE_OFFLINE"] = "1"

# ─── IMPORTS ──────────────────────────────────────────────────────────────────
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

# ─── RUTAS DEL PROYECTO ───────────────────────────────────────────────────────
BASE_DIR   = os.path.expanduser("~/mistral-project")
MODEL_BASE = os.path.join(BASE_DIR, "models/base/Mistral-7B-Instruct-v0.2")
TRAIN_PATH = os.path.join(BASE_DIR, "data/processed/lyrics_train.jsonl")
VAL_PATH   = os.path.join(BASE_DIR, "data/processed/lyrics_valid.jsonl")  # <-- Añade un val.jsonl

run_name   = time.strftime("t21p_lr2e-4_ep3_bs2x4_%Y%m%d_%H%M%S")
OUT_DIR    = os.path.join(BASE_DIR, f"models/finetuned/{run_name}")
FINAL_OUT  = os.path.join(OUT_DIR, "final_model")
os.makedirs(FINAL_OUT, exist_ok=True)

# ─── HIPERPARÁMETROS ──────────────────────────────────────────────────────────
max_seq_length = 1024
batch_size     = 2
grad_accum     = 4
num_epochs     = 3
lr             = 2e-4

# ─── CARGA DEL MODELO ─────────────────────────────────────────────────────────
print("🔹 Cargando modelo base Mistral en 4-bit (offline)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_BASE,
    max_seq_length = max_seq_length,
    dtype          = None,
    load_in_4bit   = True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─── LoRA ─────────────────────────────────────────────────────────────────────
print("🔹 Aplicando LoRA (~1 % de parámetros entrenados)...")
model = FastLanguageModel.get_peft_model(
    model,
    r                          = 16,
    target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha                 = 16,
    lora_dropout               = 0.05,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
)

# ─── DATASETS ─────────────────────────────────────────────────────────────────
print("🔹 Cargando datasets...")
data = load_dataset("json", data_files={"train": TRAIN_PATH, "validation": VAL_PATH})
train_data = data["train"]
val_data   = data["validation"]

# ─── FUNCIÓN DE MÉTRICAS ──────────────────────────────────────────────────────
import torch
from evaluate import load as load_metric

acc_metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    """
    Calcula PPL y token accuracy.
    eval_pred es un objeto EvalPrediction con logits y labels.
    """
    logits, labels = eval_pred
    # Desplazar logits a CPU
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    # Ignorar posiciones -100 (padding)
    mask = labels != -100
    preds = logits.argmax(dim=-1)
    acc = (preds[mask] == labels[mask]).float().mean().item()

    # Cross-entropy promedio
    loss = torch.nn.functional.cross_entropy(
        logits[mask].float(), labels[mask].long(), reduction="mean"
    ).item()
    ppl = math.exp(loss)
    return {"accuracy": acc, "loss": loss, "perplexity": ppl}

# ─── CONFIGURACIÓN DE ENTRENAMIENTO ───────────────────────────────────────────
args = SFTConfig(
    per_device_train_batch_size = batch_size,
    gradient_accumulation_steps = grad_accum,
    num_train_epochs            = num_epochs,
    learning_rate               = lr,
    logging_steps               = 25,
    evaluation_strategy         = "epoch",  # <-- evalúa cada época
    save_strategy               = "epoch",
    output_dir                  = OUT_DIR,
    fp16                        = True,
    report_to                   = "none",
    optim                       = "adamw_8bit",
    weight_decay                = 0.01,
    lr_scheduler_type           = "linear",
    seed                        = 3407,
)

# ─── ENTRENADOR ───────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_data,
    eval_dataset       = val_data,            # <-- ahora con validación
    dataset_text_field = "text",
    max_seq_length     = max_seq_length,
    packing            = True,
    args               = args,
    compute_metrics    = compute_metrics,     # <-- añade métricas
)

# ─── EJECUCIÓN ────────────────────────────────────────────────────────────────
print("Iniciando entrenamiento...\n")
train_output = trainer.train()

# ─── GUARDADO ─────────────────────────────────────────────────────────────────
print("Guardando adaptadores LoRA y tokenizer...")
trainer.save_model(FINAL_OUT)
tokenizer.save_pretrained(FINAL_OUT)

with open(os.path.join(OUT_DIR, "train_results.txt"), "w") as f:
    f.write(str(train_output.metrics))

print(f"\nModelo afinado guardado correctamente en:\n{FINAL_OUT}")

