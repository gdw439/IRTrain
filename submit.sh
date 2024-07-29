#!/bin/bash
#SBATCH --partition=rag
#SBATCH --nodelist=g3012
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=
#SBATCH --mem=128G
nvidia-smi
free -h
# lscpu
# deepspeed --num_gpus=1 run_train.py
CUDA_VISIBLE_DEVICES=2 #,3,4,5,6,7
    # -m /home/guodewen/research/IRTrain/history/soda_synth/epoch-19_globalStep-250 \
# python ./tools/index.py \
#     -m /home/guodewen/research/IRTrain/models/bge-m3 \
#     -q /home/guodewen/research/IRTrain/dataset/soda_stella/test.jsonl \
#     -c /home/guodewen/research/IRTrain/dataset/soda_stella/bge_large_0_0.jsonl

# python ./tools/dump_text_emb.py
# python ./tools/dump_text_emb_dpc.py
python ./tools/clean_inbatch_data.py