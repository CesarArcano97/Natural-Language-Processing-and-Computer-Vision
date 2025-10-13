#!/bin/bash
#SBATCH --job-name=finetune_tinyllama
#SBATCH --partition=GPU
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=../logs/finetune_%j.out
#SBATCH --error=../logs/finetune_%j.err

# Inicializar conda en el nodo
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate prometheus

# Directorios y parámetros
MODEL_NAME="$HOME/llama-beta/models/tiny-llama-1b"   # ruta local, no HF
CORPUS="$HOME/llama-beta/data/TOP_corpus_generativo_unificado.txt"
OUTPUT_DIR="$HOME/llama-beta/models/tinyllama_prueba02"

# Navegar a la carpeta del código
cd ~/llama-beta/src

# Ejecutar el fine-tuning
python train_finetuning.py \
  --model_name $MODEL_NAME \
  --train_file $CORPUS \
  --output_dir $OUTPUT_DIR \
  --epochs 6 \
  --batch_size 2 \
  --block_size 256 \
  --learning_rate 5e-5
