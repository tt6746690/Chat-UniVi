#!/bin/bash

set -e
set -x

# Set up a trap to catch errors and exit all processes
trap 'echo "Error caught, exiting all processes..."; kill 0; exit 1' ERR


CKPT=$1
CONV_MODE=v1

MAX_IMAGE_LENGTH=64
CHATUNIVI_REPO_DIR=/fsx/wpq/github/metasummer2024/external/Chat-UniVi
VIDEO_FOLDER=/fsx/wpq/.data/chatunivi/eval/Chat-UniVi-Eval/Test_Videos
NUM_TASKS_GPT_EVAL=3

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}


if [[ ! -f "$CKPT/eval/vcgbench/answers-generic/merge.jsonl" || \
      ! -f "$CKPT/eval/vcgbench/answers-temporal/merge.jsonl" || \
      ! -f "$CKPT/eval/vcgbench/answers-consistency/merge.jsonl" ]]; then

    for IDX in $(seq 0 $((CHUNKS-1))); do
        # generic
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ChatUniVi.eval.model_video_general \
            --model-path $CKPT \
            --question-file $CHATUNIVI_REPO_DIR/ChatUniVi/eval/questions/video_qa/generic_qa.json \
            --video-folder $VIDEO_FOLDER \
            --answers-file $CKPT/eval/vcgbench/answers-generic/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0.2 \
            --conv-mode $CONV_MODE &
        # temporal
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ChatUniVi.eval.model_video_general \
            --model-path $CKPT \
            --question-file $CHATUNIVI_REPO_DIR/ChatUniVi/eval/questions/video_qa/temporal_qa.json \
            --video-folder $VIDEO_FOLDER \
            --answers-file $CKPT/eval/vcgbench/answers-temporal/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0.2 \
            --conv-mode $CONV_MODE &
        # consistency
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ChatUniVi.eval.model_video_consistency \
            --model-path $CKPT \
            --question-file $CHATUNIVI_REPO_DIR/ChatUniVi/eval/questions/video_qa/consistency_qa.json \
            --video-folder $VIDEO_FOLDER \
            --answers-file $CKPT/eval/vcgbench/answers-consistency/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0.2 \
            --conv-mode $CONV_MODE &
    done

    wait

    # generic
    output_file=$CKPT/eval/vcgbench/answers-generic/merge.jsonl
    echo $output_file
    > "$output_file"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $CKPT/eval/vcgbench/answers-generic/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    # temporal
    output_file=$CKPT/eval/vcgbench/answers-temporal/merge.jsonl
    echo $output_file
    > "$output_file"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $CKPT/eval/vcgbench/answers-temporal/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

    # consistency
    output_file=$CKPT/eval/vcgbench/answers-consistency/merge.jsonl
    echo $output_file
    > "$output_file"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $CKPT/eval/vcgbench/answers-consistency/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done
else
    echo "$CKPT/eval/vcgbench/answers-{generic, temporal, consistency}/merge.jsonl exists!"
fi



echo "Category: Correctness of Information"
python -m ChatUniVi.eval.evaluate.benchmark_1_correctness \
    --pred_path $CKPT/eval/vcgbench/answers-generic/merge.jsonl \
    --output_dir $CKPT/eval/vcgbench/correctness \
    --output_json $CKPT/eval/vcgbench/correctness/merge.jsonl \
    --num_tasks $NUM_TASKS_GPT_EVAL

echo "Category: Detail Orientation"
python -m ChatUniVi.eval.evaluate.benchmark_2_detailed_orientation \
    --pred_path $CKPT/eval/vcgbench/answers-generic/merge.jsonl \
    --output_dir $CKPT/eval/vcgbench/detailed_orientation \
    --output_json $CKPT/eval/vcgbench/detailed_orientation/merge.jsonl \
    --num_tasks $NUM_TASKS_GPT_EVAL

echo "Category: Contextual Understanding"
python -m ChatUniVi.eval.evaluate.benchmark_3_context \
    --pred_path $CKPT/eval/vcgbench/answers-generic/merge.jsonl \
    --output_dir $CKPT/eval/vcgbench/context \
    --output_json $CKPT/eval/vcgbench/context/merge.jsonl \
    --num_tasks $NUM_TASKS_GPT_EVAL

echo "Category: Temporal Understanding"
python -m ChatUniVi.eval.evaluate.benchmark_4_temporal \
    --pred_path $CKPT/eval/vcgbench/answers-temporal/merge.jsonl \
    --output_dir $CKPT/eval/vcgbench/temporal \
    --output_json $CKPT/eval/vcgbench/temporal/merge.jsonl \
    --num_tasks $NUM_TASKS_GPT_EVAL

echo "Category: Consistency"
python -m ChatUniVi.eval.evaluate.benchmark_5_consistency \
    --pred_path $CKPT/eval/vcgbench/answers-consistency/merge.jsonl \
    --output_dir $CKPT/eval/vcgbench/consistency \
    --output_json $CKPT/eval/vcgbench/consistency/merge.jsonl \
    --num_tasks $NUM_TASKS_GPT_EVAL