#!/bin/bash
#SBATCH --job-name=yolo_bot_train
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/cpu_%j.out
#SBATCH --error=logs/cpu_%j.err

python Card_Cropper.py
