#!/bin/bash
#SBATCH --partition=rag
#SBATCH --nodelist=g3018
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=3
#SBATCH --mem=128G
nvidia-smi
free -h
set -x
# lscpu
# deepspeed --num_gpus=1 run_train.py
# CUDA_VISIBLE_DEVICES=2 #,3,4,5,6,7
#     # -m /home/guodewen/research/IRTrain/history/soda_synth/epoch-19_globalStep-250 \
# python ./tools/index.py \
#     -m /home/guodewen/research/workspace/m3-funtune/checkpoint-405 \
#     -q /home/guodewen/research/IRTrain/dataset/soda_stella/data_ground_truth_full_0627.jsonl \
#     -c /home/guodewen/research/IRTrain/dataset/soda_stella/bge_large_0_0.jsonl \
#     -t 5

# python ./tools/index.py \
#     -m /home/guodewen/research/workspace/m3-funtune/checkpoint-405 \
#     -q /home/guodewen/research/IRTrain/dataset/soda_stella/data_ground_truth_full_0627.jsonl \
#     -c /home/guodewen/research/IRTrain/dataset/soda_stella/bge_large_0_0.jsonl \
#     -t 10

# python ./tools/index.py \
#     -m /home/guodewen/research/workspace/m3-funtune/checkpoint-405 \
#     -q /home/guodewen/research/IRTrain/dataset/soda_stella/data_ground_truth_full_0627.jsonl \
#     -c /home/guodewen/research/IRTrain/dataset/soda_stella/bge_large_0_0.jsonl \
#     -t 15

# python ./tools/index.py \
#     -m /home/guodewen/research/workspace/m3-funtune/checkpoint-405 \
#     -q /home/guodewen/research/IRTrain/dataset/soda_stella/data_ground_truth_full_0627.jsonl \
#     -c /home/guodewen/research/IRTrain/dataset/soda_stella/bge_large_0_0.jsonl \
#     -t 20

# python ./tools/index.py \
#     -m /home/guodewen/research/workspace/m3-funtune/checkpoint-405 \
#     -q /home/guodewen/research/IRTrain/dataset/soda_stella/data_ground_truth_full_0627.jsonl \
#     -c /home/guodewen/research/IRTrain/dataset/soda_stella/bge_large_0_0.jsonl \
#     -t 25

# python ./tools/dump_text_emb.py
# python ./tools/dump_text_emb_dpc.py
# python ./tools/clean_inbatch_data.py