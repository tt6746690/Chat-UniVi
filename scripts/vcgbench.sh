#!/bin/bash
# bash scripts/vcgbench.sh /fsx/wpq/.results/baselines/Chat-UniVi/Chat-UniVi-7B-v1.5

set -e


CKPT=$1
VIDEO_FOLDER=/fsx/wpq/.data/chatunivi/eval/Test_Videos


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
echo $gpu_list

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    # generic
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ChatUniVi.eval.model_video_general \
        --model-path $CKPT \
        --question-file ChatUniVi/eval/questions/video_qa/generic_qa.json \
        --video-folder $VIDEO_FOLDER \
        --answers-file $CKPT/eval/vcgbench/answers-generic/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0.2 \
        --conv-mode v1 &
    # temporal
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ChatUniVi.eval.model_video_general \
        --model-path $CKPT \
        --question-file ChatUniVi/eval/questions/video_qa/temporal_qa.json \
        --video-folder $VIDEO_FOLDER \
        --answers-file $CKPT/eval/vcgbench/answers-temporal/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0.2 \
        --conv-mode v1 &
    # consistency
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ChatUniVi.eval.model_video_consistency \
        --model-path $CKPT \
        --question-file ChatUniVi/eval/questions/video_qa/consistency_qa.json \
        --video-folder $VIDEO_FOLDER \
        --answers-file $CKPT/eval/vcgbench/answers-consistency/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0.2 \
        --conv-mode v1 &
done

wait


# generic
output_file=$CKPT/eval/vcgbench/answers-generic/merge.jsonl
echo $output_file
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $CKPT/eval/vcgbench/answers-generic/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# temporal
output_file=$CKPT/eval/vcgbench/answers-temporal/merge.jsonl
echo $output_file
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $CKPT/eval/vcgbench/answers-temporal/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# consistency
output_file=$CKPT/eval/vcgbench/answers-consistency/merge.jsonl
echo $output_file
# Clear out the output file if it exists.
> "$output_file"
# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $CKPT/eval/vcgbench/answers-consistency/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done




# NUM_TASKS=3

# echo "Category: Correctness of Information"
# python ChatUniVi/eval/evaluate/evaluate_benchmark_1_correctness.py \
#     --pred_path $CKPT/eval/vcgbench/answers-generic/merge.jsonl \
#     --output_dir $CKPT/eval/vcgbench/correctness \
#     --output_json $CKPT/eval/vcgbench/correctness/merge.jsonl \
#     --api_key $(cat ~/.openai_api_key) \
#     --num_tasks $NUM_TASKS

# echo "Category: Detail Orientation"
# python ChatUniVi/eval/evaluate/evaluate_benchmark_2_detailed_orientation.py \
#     --pred_path $CKPT/eval/vcgbench/answers-generic/merge.jsonl \
#     --output_dir $CKPT/eval/vcgbench/detailed_orientation \
#     --output_json $CKPT/eval/vcgbench/detailed_orientation/merge.jsonl \
#     --api_key $(cat ~/.openai_api_key) \
#     --num_tasks $NUM_TASKS

# echo "Category: Contextual Understanding"
# python ChatUniVi/eval/evaluate/evaluate_benchmark_3_context.py \
#     --pred_path $CKPT/eval/vcgbench/answers-generic/merge.jsonl \
#     --output_dir $CKPT/eval/vcgbench/context \
#     --output_json $CKPT/eval/vcgbench/context/merge.jsonl \
#     --api_key $(cat ~/.openai_api_key) \
#     --num_tasks $NUM_TASKS