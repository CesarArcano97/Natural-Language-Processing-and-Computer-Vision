#!/bin/bash
#SBATCH --job-name=train_mistral_t21p
#SBATCH --partition=GPU            # Ajusta si la cola se llama distinto
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-user=cesar.aguirre@cimat.mx
#SBATCH --mail-type=END,FAIL

echo "🔹 Iniciando entrenamiento en $(hostname) - Job $SLURM_JOB_ID"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mistral-env

# 100% offline y cachés locales
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd ~/mistral-project
python src/train/train_mistral_offline.py

echo "Entrenamiento finalizado en $(date) :)"
