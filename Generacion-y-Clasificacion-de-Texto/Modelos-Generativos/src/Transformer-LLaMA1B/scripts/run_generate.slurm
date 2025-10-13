#!/bin/bash
#SBATCH --job-name=generate_lyrics
#SBATCH --partition=GPU
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/est_posgrado_cesar.aguirre/llama-beta/logs/generate_%j.out
#SBATCH --error=/home/est_posgrado_cesar.aguirre/llama-beta/logs/generate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tu_correo@cimat.mx

# Inicializar conda en nodos Slurm
source ~/miniconda3/etc/profile.d/conda.sh
conda activate prometheus

# Navegar a la carpeta de código y ejecutar el script
cd ~/llama-beta/src
python generate_server.py
