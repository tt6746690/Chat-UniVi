#!/bin/bash

set -e
set -x


CKPT=$1
VIDEO_FOLDER=/fsx/wpq/.data/chatunivi/eval/Test_Videos
NUM_TASKS_GPT_EVAL=3

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
echo $gpu_list

CHUNKS=${#GPULIST[@]}



if [[ ! -f "$CKPT/eval/vcgbench/answers-generic/merge.jsonl" || \
      ! -f "$CKPT/eval/vcgbench/answers-temporal/merge.jsonl" || \
      ! -f "$CKPT/eval/vcgbench/answers-consistency/merge.jsonl" ]]; then

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

    # wait

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
python ChatUniVi/eval/evaluate/evaluate_benchmark_1_correctness.py \
    --pred_path $CKPT/eval/vcgbench/answers-generic/merge.jsonl \
    --output_dir $CKPT/eval/vcgbench/correctness \
    --output_json $CKPT/eval/vcgbench/correctness/merge.jsonl \
    --num_tasks $NUM_TASKS_GPT_EVAL

echo "Category: Detail Orientation"
python ChatUniVi/eval/evaluate/evaluate_benchmark_2_detailed_orientation.py \
    --pred_path $CKPT/eval/vcgbench/answers-generic/merge.jsonl \
    --output_dir $CKPT/eval/vcgbench/detailed_orientation \
    --output_json $CKPT/eval/vcgbench/detailed_orientation/merge.jsonl \
    --num_tasks $NUM_TASKS_GPT_EVAL

echo "Category: Contextual Understanding"
python ChatUniVi/eval/evaluate/evaluate_benchmark_3_context.py \
    --pred_path $CKPT/eval/vcgbench/answers-generic/merge.jsonl \
    --output_dir $CKPT/eval/vcgbench/context \
    --output_json $CKPT/eval/vcgbench/context/merge.jsonl \
    --num_tasks $NUM_TASKS_GPT_EVAL

echo "Category: Temporal Understanding"
python ChatUniVi/eval/evaluate/evaluate_benchmark_4_temporal.py \
    --pred_path $CKPT/eval/vcgbench/answers-temporal/merge.jsonl \
    --output_dir $CKPT/eval/vcgbench/temporal \
    --output_json $CKPT/eval/vcgbench/temporal/merge.jsonl \
    --num_tasks $NUM_TASKS_GPT_EVAL

echo "Category: Consistency"
python ChatUniVi/eval/evaluate/evaluate_benchmark_5_consistency.py \
    --pred_path $CKPT/eval/vcgbench/answers-consistency/merge.jsonl \
    --output_dir $CKPT/eval/vcgbench/consistency \
    --output_json $CKPT/eval/vcgbench/consistency/merge.jsonl \
    --num_tasks $NUM_TASKS_GPT_EVAL