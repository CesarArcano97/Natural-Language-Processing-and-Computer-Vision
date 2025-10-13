import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    pipeline
)

def main(args):
    print(f"🔹 Cargando modelo desde {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Asegurar que el tokenizer tenga un token de padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"🔹 Preparando dataset desde {args.train_file}")
    raw_datasets = load_dataset("text", data_files={"train": args.train_file})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.block_size,
        )

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",  # evita que intente loguear a wandb
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
    )

    print("🚀 Iniciando entrenamiento...")
    trainer.train()

    # Guardar modelo final
    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✅ Modelo guardado en {final_path}")

    # Prueba rápida de generación
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-generation", model=final_path, tokenizer=final_path, device=device)
    print(pipe("At Dema, there is no choice.", max_length=50, do_sample=True)[0]["generated_text"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning TinyLLaMA")
    parser.add_argument("--model_name", type=str, required=True, help="Ruta al modelo base")
    parser.add_argument("--train_file", type=str, required=True, help="Ruta al corpus de entrenamiento")
    parser.add_argument("--output_dir", type=str, default="./models", help="Directorio para guardar el modelo")
    parser.add_argument("--epochs", type=int, default=1, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size por dispositivo")
    parser.add_argument("--block_size", type=int, default=512, help="Longitud máxima de secuencia")
    args = parser.parse_args()
    main(args)
