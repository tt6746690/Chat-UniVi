#!/bin/bash

set -e
set -x

CKPT=$1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

MAX_IMAGE_LENGTH=16


# MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv-endpoint=$MASTER_ADDR:1212 
torchrun --nproc_per_node=1 ChatUniVi/eval/model_video_mvbench.py \
    --model-path $CKPT \
    --video-folder /fsx/wpq/.data/chatunivi/eval/MVBench/video \
    --question-dir /fsx/wpq/.data/chatunivi/eval/MVBench/json \
    --output-dir $CKPT/eval/mvbench \
    --temperature 0.0 \
    --conv-mode v1 \
    --batch_size_per_gpu 1


# python ChatUniVi/eval/evaluate/evaluate_mvbench.py \
#     --output-dir $CKPT/eval/mvbench