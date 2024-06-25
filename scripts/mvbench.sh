#!/bin/bash

set -e
set -x

# Set up a trap to catch errors and exit all processes
trap 'echo "Error caught, exiting all processes..."; kill 0; exit 1' ERR


CKPT=$1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ChatUniVi.eval.model_video_mvbench \
        --model-path $CKPT \
        --video-folder /fsx/wpq/.data/chatunivi/eval/MVBench/video \
        --question-dir /fsx/wpq/.data/chatunivi/eval/MVBench/json_valid \
        --output-dir $CKPT/eval/mvbench \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0.0 \
        --conv-mode v1 &
done

wait


python ChatUniVi/eval/evaluate/evaluate_mvbench.py \
    --output-dir $CKPT/eval/mvbench