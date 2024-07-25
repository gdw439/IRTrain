#!/bin/bash
#SBATCH --partition=rag
#SBATCH --nodelist=g3013
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --output=./train.log
nvidia-smi
free -h
lscpu
deepspeed --num_gpus=1 run_train.py
