#!/bin/bash
#SBATCH --job-name=yolo_train
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:V100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/yolo_train_%j.out
#SBATCH --error=logs/yolo_train_%j.err

python Train_CardVision.py
