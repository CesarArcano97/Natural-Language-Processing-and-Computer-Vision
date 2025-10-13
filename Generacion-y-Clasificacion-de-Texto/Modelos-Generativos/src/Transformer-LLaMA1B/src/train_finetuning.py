#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning ligero de TinyLLaMA en un corpus pequeño (~100 canciones).
Compatible con versiones antiguas de HuggingFace Transformers.
"""

import argparse
import os
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    pipeline,
)

def main(args):
    # Modelo y tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    raw_datasets = load_dataset("text", data_files={"data": args.train_file})
    split = raw_datasets["data"].train_test_split(test_size=0.1, seed=42)
    raw_train, raw_val = split["train"], split["test"]

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_train = raw_train.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = raw_val.map(tokenize_function, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = sum(examples["input_ids"], [])
        total_length = (len(concatenated) // args.block_size) * args.block_size
        result = {
            "input_ids": [concatenated[i : i + args.block_size] for i in range(0, total_length, args.block_size)],
        }
        result["attention_mask"] = [[1] * args.block_size] * len(result["input_ids"])
        return result   # <-- corregida indentación

    lm_train = tokenized_train.map(group_texts, batched=True)
    lm_val = tokenized_val.map(group_texts, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=500,
        logging_steps=100,
        save_total_limit=2,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train,
        eval_dataset=lm_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Entrenamiento
    trainer.train()

    # Evaluación final para calcular PPL en validación
    metrics = trainer.evaluate(eval_dataset=lm_val)
    if "eval_loss" in metrics:
        metrics["perplexity"] = math.exp(metrics["eval_loss"])
    print("Evaluación final:", metrics)

    # Guardar modelo
    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    # Guardar logs
    log_history = pd.DataFrame(trainer.state.log_history)
    log_path = os.path.join(args.output_dir, "training_log.csv")
    log_history.to_csv(log_path, index=False)

    # 🔹 Imprimir epoch vs PPL en consola (quedará en tu .out del SLURM)
    if "eval_perplexity" in log_history.columns:
        for _, row in log_history.iterrows():
            if "eval_perplexity" in row:
                print(f"Época {row['epoch']:.1f} | PPL validación: {row['eval_perplexity']:.2f}")

        # Curva de PPL
        plt.plot(log_history["epoch"], log_history["eval_perplexity"], marker="o")
        plt.xlabel("Época")
        plt.ylabel("Perplejidad (val)")
        plt.title("Curva de PPL vs Época")
        plt.grid()
        fig_path = os.path.join(args.output_dir, "ppl_curve.png")
        plt.savefig(fig_path)

    # Generación de prueba
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-generation", model=final_path, tokenizer=final_path, device=device)
    sample = pipe(
        "I like my soul but not my mind",
        max_new_tokens=100,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        temperature=0.9,
        repetition_penalty=1.15,
    )[0]["generated_text"]
    print(sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning TinyLLaMA en corpus pequeño")
    parser.add_argument("--model_name", type=str, required=True, help="Ruta o nombre del modelo base")
    parser.add_argument("--train_file", type=str, required=True, help="Ruta al corpus (txt)")
    parser.add_argument("--output_dir", type=str, default="./models", help="Directorio de salida")
    parser.add_argument("--epochs", type=int, default=3, help="Épocas (3-5 recomendado)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--block_size", type=int, default=256, help="Longitud máxima de secuencia")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()
    main(args)
