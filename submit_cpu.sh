#!/bin/bash
#SBATCH --job-name=yolo_bot_train
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/log_addestramento_%j.out
#SBATCH --error=logs/log_errore_%j.err

python CardGeneration.py
