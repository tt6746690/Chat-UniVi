#!/bin/bash

set -e
set -x

CKPT=$1
CONV_MODE=v1
EVAL_DATA_DIR=/fsx/wpq/.data/eval/mmbench
SPLIT="mmbench_dev_20230712"
ANSWER_UPLOAD_DIR=/fsx/wpq/github/metasummer2024/external/LLaVA/playground/answers_upload

python -m ChatUniVi.eval.model_vqa_mmbench \
    --model-path $CKPT \
    --question-file $EVAL_DATA_DIR/$SPLIT.tsv \
    --answers-file $CKPT/eval/mmbench/$SPLIT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE

python -m ChatUniVi.eval.convert_mmbench_for_submission \
    --annotation-file $EVAL_DATA_DIR/$SPLIT.tsv \
    --result-dir $CKPT/eval/mmbench \
    --upload-dir $CKPT/eval/mmbench \
    --experiment $SPLIT


python -m ChatUniVi.eval.copy_predictions $CKPT $ANSWER_UPLOAD_DIR