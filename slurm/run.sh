#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:lgpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --account=def-furukawa

hostname
nvidia-smi
