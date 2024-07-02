#!/bin/bash

set -e
set -x

CKPT=$1
CONV_MODE=v1
EVAL_DATA_DIR=/fsx/wpq/.data/eval/mm-vet

python -m ChatUniVi.eval.model_vqa \
    --model-path $CKPT \
    --question-file $EVAL_DATA_DIR/llava-mm-vet.jsonl \
    --image-folder $EVAL_DATA_DIR/images \
    --answers-file $CKPT/eval/mmvet/answers.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

python -m ChatUniVi.eval.convert_mmvet_for_eval \
    --src $CKPT/eval/mmvet/answers.jsonl \
    --dst $CKPT/eval/mmvet/results.json

python $EVAL_DATA_DIR/mm-vet_evaluator.py \
    --mmvet_path $EVAL_DATA_DIR \
    --result_file $CKPT/eval/mmvet/results.json \
    --result_path $CKPT/eval/mmvet \
    --gpt_model gpt-4-0613
    # --use_sub_set to debug