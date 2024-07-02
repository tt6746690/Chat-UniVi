#!/bin/bash

set -e
set -x

# Set up a trap to catch errors and exit all processes
trap 'echo "Error caught, exiting all processes..."; kill 0; exit 1' ERR


CKPT=$1
CONV_MODE=v1

MAX_IMAGE_LENGTH=64 # default=16 used by mvbench's paper. but I'll keep at 64 since it's the number used for training.
EVAL_DATA_DIR=/fsx/wpq/.data/chatunivi/eval/MVBench

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ChatUniVi.eval.model_video_mvbench \
        --model-path $CKPT \
        --video-folder $EVAL_DATA_DIR/video \
        --question-dir $EVAL_DATA_DIR/json_valid \
        --output-dir $CKPT/eval/mvbench \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0.0 \
        --conv-mode $CONV_MODE &
done

wait

python -m ChatUniVi.eval.evaluate.evaluate_mvbench \
    --output-dir $CKPT/eval/mvbench