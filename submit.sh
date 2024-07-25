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
# deepspeed --num_gpus=1 run_train.py
python ./tools/index.py \
    -m /home/guodewen/research/IRTrain/models/stella_v2_large \
    -q /home/guodewen/research/IRTrain/dataset/test.jsonl \
    -c /home/guodewen/research/IRTrain/dataset/soda_stella/bge_large_0_0.jsonl
