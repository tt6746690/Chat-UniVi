#!/bin/bash

set -e
set -x



# Set up a trap to catch errors and exit all processes
trap 'echo "Error caught, exiting all processes..."; kill 0; exit 1' ERR


CKPT=$1
TOKEN_SCALE=$2
SAVE_DIR=$3
CONV_MODE=$4

MAX_IMAGE_LENGTH=64
CHATUNIVI_REPO_DIR=/fsx/wpq/github/metasummer2024/external/Chat-UniVi
VIDEO_FOLDER=/fsx/wpq/.data/chatunivi/eval/Chat-UniVi-Eval/Test_Videos
NUM_TASKS_GPT_EVAL=3
# LLM_JUDGE_MODEL=gpt-3.5-turbo
LLM_JUDGE_MODEL=gpt-4o-mini-2024-07-18


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}


# generic
if [[ ! -f "$SAVE_DIR/answers-generic/merge.jsonl" ]]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ChatUniVi.eval.model_video_general \
            --model-path $CKPT \
            --question-file $CHATUNIVI_REPO_DIR/ChatUniVi/eval/questions/video_qa/generic_qa.json \
            --video-folder $VIDEO_FOLDER \
            --answers-file $SAVE_DIR/answers-generic/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0.2 \
            --conv-mode $CONV_MODE \
            $(if [ -n "$TOKEN_SCALE" ]; then echo "--matryoshka_vis_token_scale $TOKEN_SCALE"; fi) &
    done
    wait
    output_file=$SAVE_DIR/answers-generic/merge.jsonl
    echo $output_file
    > "$output_file"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $SAVE_DIR/answers-generic/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done
fi


# temporal
if [[ ! -f "$SAVE_DIR/answers-temporal/merge.jsonl" ]]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ChatUniVi.eval.model_video_general \
            --model-path $CKPT \
            --question-file $CHATUNIVI_REPO_DIR/ChatUniVi/eval/questions/video_qa/temporal_qa.json \
            --video-folder $VIDEO_FOLDER \
            --answers-file $SAVE_DIR/answers-temporal/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0.2 \
            --conv-mode $CONV_MODE \
            $(if [ -n "$TOKEN_SCALE" ]; then echo "--matryoshka_vis_token_scale $TOKEN_SCALE"; fi) &
    done
    wait
    output_file=$SAVE_DIR/answers-temporal/merge.jsonl
    echo $output_file
    > "$output_file"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $SAVE_DIR/answers-temporal/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done
fi


# consistency
if [[ ! -f "$SAVE_DIR/answers-consistency/merge.jsonl" ]]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ChatUniVi.eval.model_video_consistency \
            --model-path $CKPT \
            --question-file $CHATUNIVI_REPO_DIR/ChatUniVi/eval/questions/video_qa/consistency_qa.json \
            --video-folder $VIDEO_FOLDER \
            --answers-file $SAVE_DIR/answers-consistency/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0.2 \
            --conv-mode $CONV_MODE \
            $(if [ -n "$TOKEN_SCALE" ]; then echo "--matryoshka_vis_token_scale $TOKEN_SCALE"; fi) &
    done
    wait
    output_file=$SAVE_DIR/answers-consistency/merge.jsonl
    echo $output_file
    > "$output_file"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $SAVE_DIR/answers-consistency/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done
fi


echo "Category: Correctness of Information"
python -m ChatUniVi.eval.evaluate.evaluate_benchmark_1_correctness \
    --pred_path $SAVE_DIR/answers-generic/merge.jsonl \
    --output_dir $SAVE_DIR/correctness \
    --output_json $SAVE_DIR/correctness/merge.jsonl \
    --num_tasks $NUM_TASKS_GPT_EVAL \
    --model $LLM_JUDGE_MODEL

echo "Category: Detail Orientation"
python -m ChatUniVi.eval.evaluate.evaluate_benchmark_2_detailed_orientation \
    --pred_path $SAVE_DIR/answers-generic/merge.jsonl \
    --output_dir $SAVE_DIR/detailed_orientation \
    --output_json $SAVE_DIR/detailed_orientation/merge.jsonl \
    --num_tasks $NUM_TASKS_GPT_EVAL \
    --model $LLM_JUDGE_MODEL

echo "Category: Contextual Understanding"
python -m ChatUniVi.eval.evaluate.evaluate_benchmark_3_context \
    --pred_path $SAVE_DIR/answers-generic/merge.jsonl \
    --output_dir $SAVE_DIR/context \
    --output_json $SAVE_DIR/context/merge.jsonl \
    --num_tasks $NUM_TASKS_GPT_EVAL \
    --model $LLM_JUDGE_MODEL

echo "Category: Temporal Understanding"
python -m ChatUniVi.eval.evaluate.evaluate_benchmark_4_temporal \
    --pred_path $SAVE_DIR/answers-temporal/merge.jsonl \
    --output_dir $SAVE_DIR/temporal \
    --output_json $SAVE_DIR/temporal/merge.jsonl \
    --num_tasks $NUM_TASKS_GPT_EVAL \
    --model $LLM_JUDGE_MODEL

echo "Category: Consistency"
python -m ChatUniVi.eval.evaluate.evaluate_benchmark_5_consistency \
    --pred_path $SAVE_DIR/answers-consistency/merge.jsonl \
    --output_dir $SAVE_DIR/consistency \
    --output_json $SAVE_DIR/consistency/merge.jsonl \
    --num_tasks $NUM_TASKS_GPT_EVAL \
    --model $LLM_JUDGE_MODEL