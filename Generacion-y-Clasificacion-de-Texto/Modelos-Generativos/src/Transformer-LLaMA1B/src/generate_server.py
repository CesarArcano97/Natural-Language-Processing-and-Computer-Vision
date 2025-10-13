import os
import torch
from transformers import pipeline

# --- RUTA AL MODELO AFINADO ---
model_path = "/home/est_posgrado_cesar.aguirre/llama-beta/models/tinyllama_prueba02/final"

# --- PROMPTS PARA GENERAR LETRAS ---
prompts = [
    "I will take my heart with my hands and",
    "If you ever wanna see me just",
    "Dema will fall like the"
]

# --- DIRECTORIO DE SALIDA ---
output_dir = "/home/est_posgrado_cesar.aguirre/llama-beta/results/generations"
os.makedirs(output_dir, exist_ok=True)

# --- CARGAR EL MODELO ---
print("🔹 Cargando el modelo afinado desde:", model_path)
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline(
    "text-generation",
    model=model_path,
    tokenizer=model_path,
    device=device
)

# --- GENERAR Y GUARDAR RESULTADOS ---
for i, prompt in enumerate(prompts):
    print(f"\n--- Generando para el Prompt #{i+1} ---")
    
    generation_params = {
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "num_return_sequences": 1,
        "max_new_tokens": 300,
        "repetition_penalty": 1.15,
    }

    try:
        sequences = pipe(prompt, **generation_params)
        generated_text = sequences[0]['generated_text']

        # Guardar en archivo
        file_path = os.path.join(output_dir, f"lyrics_prompt{i+1}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"PROMPT:\n{prompt}\n\nGENERATED:\n{generated_text}\n")

        print(f"✅ Letra generada guardada en {file_path}")

    except Exception as e:
        print(f"❌ Error al generar para el prompt '{prompt}': {e}")
