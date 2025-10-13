#!/bin/bash
#SBATCH --job-name=llama_beta_train
#SBATCH --partition=GPU
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/llama_beta_train-%j.log

echo "🔹 Inicializando entorno Conda en el nodo..."
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate prometheus

echo "🔹 Iniciando entrenamiento TinyLLaMA..."
python src/train.py \
  --model_name "./models/tiny-llama-1b" \
  --train_file "./data/TOP_corpus_generativo_unificado.txt" \
  --output_dir "./models/tiny-llama-1b-finetuned" \
  --epochs 2 \
  --batch_size 2 \
  --block_size 512
