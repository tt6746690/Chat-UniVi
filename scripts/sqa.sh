#!/bin/bash

set -e
set -x

CKPT=$1
TOKEN_SCALE=$2
CONV_MODE=v1
EVAL_DATA_DIR=/fsx/wpq/.data/eval/scienceqa

python -m ChatUniVi.eval.model_vqa_science \
    --model-path $1 \
    --question-file $EVAL_DATA_DIR/llava_test_CQM-A.json \
    --image-folder $EVAL_DATA_DIR/images/test \
    --answers-file $CKPT/eval/scienceqa/answers.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE  \
    --matryoshka_vis_token_scale $TOKEN_SCALE

python -m ChatUniVi.eval.eval_science_qa \
    --base-dir $EVAL_DATA_DIR \
    --result-file $CKPT/eval/scienceqa/answers.jsonl \
    --output-file $CKPT/eval/scienceqa/outputs.jsonl \
    --output-result $CKPT/eval/scienceqa/results.jsonl
