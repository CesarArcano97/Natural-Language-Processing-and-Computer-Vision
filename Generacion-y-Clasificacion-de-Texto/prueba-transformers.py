import os
import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline
)
from datasets import load_dataset

# -----------------------------------------------------------------
# PASO 1: CONFIGURACIÓN
# -----------------------------------------------------------------
model_name = "sshleifer/tiny-gpt2"

# Ajusta la ruta al corpus en tu HOME del cluster
train_file = "/home/est_posgrado_cesar.aguirre/TOP_corpus_generativo_unificado.txt"

# Directorio donde se guardará el modelo fine-tuneado
output_dir = "./top-tinygpt2-generator"

# -----------------------------------------------------------------
# PASO 2: CARGAR MODELO Y TOKENIZADOR
# -----------------------------------------------------------------
print(f"Cargando el modelo y tokenizador para '{model_name}'...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
print("Modelo y tokenizador cargados.")

# -----------------------------------------------------------------
# PASO 3: PREPARAR EL DATASET
# -----------------------------------------------------------------
if not os.path.exists(train_file):
    print(f"ERROR: no se encontró el archivo en la ruta '{train_file}'")
    sys.exit(1)

print(f"Preparando dataset desde '{train_file}'...")
dataset = load_dataset("text", data_files={"train": train_file})

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
print("✅ Dataset tokenizado y preparado.")

# -----------------------------------------------------------------
# PASO 4: CONFIGURAR Y EJECUTAR EL FINE-TUNING
# -----------------------------------------------------------------
print("Configurando los argumentos de entrenamiento...")
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=45,          # Usa pocas épocas para prueba, sube si necesitas más
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)

print("🚀 Iniciando el fine-tuning en GPU...")
trainer.train()

# -----------------------------------------------------------------
# PASO 5: GUARDAR EL MODELO Y GENERAR TEXTO
# -----------------------------------------------------------------
final_model_path = os.path.join(output_dir, "final")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"¡Modelo guardado exitosamente en '{final_model_path}'!")

print("\nCargando el modelo afinado para generación...")
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-generation", model=final_model_path, tokenizer=final_model_path, device=device)

prompt = "At Dema, there is no choice. But at Trech, I can choose."
print("\n🎶 --- Canción Generada por TØP-TinyGPT2 --- 🎶")
sequences = pipe(
    prompt,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
    max_length=200,
)
print(sequences[0]["generated_text"])