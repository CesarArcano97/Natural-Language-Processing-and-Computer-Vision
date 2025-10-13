#!/bin/bash
#SBATCH --job-name=generate_lyrics
#SBATCH --partition=GPU
#SBATCH --time=00:10:00
#SBATCH --output=/home/est_posgrado_cesar.aguirre/llama-beta/logs/generate_%j.out
#SBATCH --error=/home/est_posgrado_cesar.aguirre/llama-beta/logs/generate_%j.err

# Inicializar conda en nodos Slurm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate prometheus

cd ~/llama-beta/src
python generate_server.py
