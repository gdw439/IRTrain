#!/bin/bash
#SBATCH --partition=rag
#SBATCH --nodelist=g1003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
nvidia-smi
free -h
lscpu
# python clean_data.py
# tail -f /dev/null
