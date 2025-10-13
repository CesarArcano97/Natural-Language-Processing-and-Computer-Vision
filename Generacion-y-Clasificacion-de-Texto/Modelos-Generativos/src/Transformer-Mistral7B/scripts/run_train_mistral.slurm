#!/bin/bash
#SBATCH --job-name=train_mistral_t21p           # Nombre del job
#SBATCH --partition=GPU                         # Cola de GPUs (ajusta según tu caso)
#SBATCH --time=04:00:00                         # Tiempo máximo (ajusta según el corpus)
#SBATCH --output=logs/train_mistral_%j.out      # Log de salida
#SBATCH --error=logs/train_mistral_%j.err       # Log de errores
#SBATCH --mem=32G                               # Memoria CPU
#SBATCH --cpus-per-task=8                       # Núcleos de CPU
#SBATCH --mail-user=cesar.aguirre@cimat.mx   # Tu correo institucional
#SBATCH --mail-type=END,FAIL         

# === Inicialización del entorno ===
echo "🔹 Iniciando entorno en $(hostname) con Slurm job ID: $SLURM_JOB_ID"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mistral-env

cd ~/mistral-project

# === Variables de ruta ===
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface
export PYTHONPATH=~/mistral-project:$PYTHONPATH

# === Lanzar entrenamiento ===
echo "Iniciando fine-tuning de Mistral-7B con corpus de Twenty One Pilots..."
python src/train/train_mistral.py

echo "Entrenamiento finalizado en $(date)"
