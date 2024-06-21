#!/bin/bash

set -e


# CUDA_VISIBLE_DEVICES=0 python ChatUniVi/eval/model_video_general.py \
#     --model-path $CKPT \
#     --question-file ChatUniVi/eval/questions/video_qa/generic_qa.json \
#     --video-folder /fsx/wpq/.data/chatunivi/eval/Test_Videos \
#     --answers-file $CKPT/eval/vcgbench/answers.jsonl


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ChatUniVi.eval.model_video_general \
        --model-path $CKPT \
        --question-file ChatUniVi/eval/questions/video_qa/generic_qa.json \
        --video-folder /fsx/wpq/.data/chatunivi/eval/Test_Videos \
        --answers-file $CKPT/eval/vcgbench/answers/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0.2 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$CKPT/eval/vcgbench/answers/merge.jsonl

echo $output_file

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $CKPT/eval/vcgbench/answers/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done






# # CUDA_VISIBLE_DEVICES=0,1,2,3 \
# echo python ChatUniVi/eval/model_video_general.py \
#     --model-path $CKPT \
#     --question-file ChatUniVi/eval/questions/video_qa/temporal_qa.json \
#     --video-folder $VIDEO_FOLDER \
#     --answers-file results/answer-video-temporal.jsonl


# # CUDA_VISIBLE_DEVICES=0,1,2,3 \
# echo python ChatUniVi/eval/model_video_consistency.py \
#     --model-path $CKPT \
#     --question-file ChatUniVi/eval/questions/video_qa/consistency_qa.json \
#     --video-folder $VIDEO_FOLDER \
#     --answers-file results/answer-video-consistency.jsonl