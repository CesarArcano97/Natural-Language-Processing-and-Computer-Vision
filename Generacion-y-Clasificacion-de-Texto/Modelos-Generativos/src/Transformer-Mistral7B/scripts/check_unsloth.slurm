#!/bin/bash
#SBATCH --job-name=check_unsloth
#SBATCH --partition=GPU
#SBATCH --time=00:05:00
#SBATCH --output=logs/check_unsloth_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mistral-env

python -c "import unsloth; print('✅ Unsloth version:', unsloth.__version__)"
