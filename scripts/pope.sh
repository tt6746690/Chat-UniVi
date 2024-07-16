#!/bin/bash

set -e
set -x

CKPT=$1
TOKEN_SCALE=$2
CONV_MODE=v1
EVAL_DATA_DIR=/fsx/wpq/.data/eval/pope

python -m ChatUniVi.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file $EVAL_DATA_DIR/llava_pope_test.jsonl \
    --image-folder $EVAL_DATA_DIR/val2014 \
    --answers-file $CKPT/eval/pope/answers.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE  \
    --matryoshka_vis_token_scale $TOKEN_SCALE

python -m ChatUniVi.eval.eval_pope \
    --annotation-dir $EVAL_DATA_DIR/coco \
    --question-file $EVAL_DATA_DIR/llava_pope_test.jsonl \
    --result-file $CKPT/eval/pope/answers.jsonl