#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=GPU
#SBATCH --time=00:02:00
#SBATCH --output=gpu_test_%j.out

nvidia-smi
