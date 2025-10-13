#!/bin/bash
#SBATCH --job-name=generate_mistral
#SBATCH --partition=GPU
#SBATCH --time=00:20:00
#SBATCH --output=/home/est_posgrado_cesar.aguirre/mistral-project/logs/generate_%j.out
#SBATCH --error=/home/est_posgrado_cesar.aguirre/mistral-project/logs/generate_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

echo "🔹 Iniciando generación en nodo: $(hostname) - Job $SLURM_JOB_ID"
date

# --- Activar entorno ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mistral-env

# --- Cambiar a carpeta del proyecto ---
cd ~/mistral-project/src/infer

# --- Ejecutar generación ---
python generate_mistral_offline.py

echo "Generación completada"
date
