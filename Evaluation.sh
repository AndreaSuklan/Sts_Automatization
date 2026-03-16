#!/bin/bash
#SBATCH --job-name=yolo_train
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:V100:2
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/gpu_%j.out
#SBATCH --error=logs/gpu_%j.err

python Eval_ComputerVision.py
