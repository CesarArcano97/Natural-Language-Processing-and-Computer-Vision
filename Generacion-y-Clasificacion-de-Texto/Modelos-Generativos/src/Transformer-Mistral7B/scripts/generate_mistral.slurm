#!/bin/bash
#SBATCH --job-name=infer_mistral_t21p
#SBATCH --partition=GPU
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err
#SBATCH --mail-user=est_posgrado_cesar.aguirre@cimat.mx
#SBATCH --mail-type=END,FAIL

echo "🔹 Iniciando inferencia en $(hostname) - Job $SLURM_JOB_ID"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mistral-env

export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd ~/mistral-project
python src/infer/generate_mistral_offline.py

echo "Inferencia finalizada en $(date) :)"
