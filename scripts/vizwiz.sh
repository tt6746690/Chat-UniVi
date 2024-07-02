#!/bin/bash

set -e

CKPT=$1
CONV_MODE=v1

EVAL_DATA_DIR=/fsx/wpq/.data/eval/vizwiz

python -m ChatUniVi.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file $EVAL_DATA_DIR/llava_test.jsonl \
    --image-folder $EVAL_DATA_DIR/test \
    --answers-file $CKPT/eval/vizwiz/answers.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

python -m ChatUniVi.eval.convert_vizwiz_for_submission \
    --annotation-file $EVAL_DATA_DIR/llava_test.jsonl \
    --result-file $CKPT/eval/vizwiz/answers.jsonl \
    --result-upload-file $CKPT/eval/vizwiz/answers_upload.json


# submit with evalai-cli
source /fsx/wpq/.profile_local.sh
conda activate evalai-cli
echo -e "y\n$CKPT\n\n\n\n" | evalai challenge 2185 phase 4336 submit --file $CKPT/eval/vizwiz/answers_upload.json  --large --private

