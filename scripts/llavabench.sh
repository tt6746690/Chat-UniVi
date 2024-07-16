#!/bin/bash

set -e
set -x

CKPT=$1
TOKEN_SCALE=$2
CONV_MODE=v1

EVAL_DATA_DIR=/fsx/wpq/.data/eval/llava-bench-in-the-wild
LLAVA_REPO_DIR=/fsx/wpq/github/metasummer2024/external/LLaVA

python -m ChatUniVi.eval.model_vqa \
    --model-path $CKPT \
    --question-file $EVAL_DATA_DIR/questions.jsonl \
    --image-folder $EVAL_DATA_DIR/images \
    --answers-file $CKPT/eval/llavabench/answers.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE  \
    --matryoshka_vis_token_scale $TOKEN_SCALE

mkdir -p $EVAL_DATA_DIR/reviews

python -m ChatUniVi.eval.eval_gpt_review_bench \
    --question $EVAL_DATA_DIR/questions.jsonl \
    --context $EVAL_DATA_DIR/context.jsonl \
    --rule $LLAVA_REPO_DIR/llava/eval/table/rule.json \
    --answer-list \
        $EVAL_DATA_DIR/answers_gpt4.jsonl \
        $CKPT/eval/llavabench/answers.jsonl \
    --output \
        $CKPT/eval/llavabench/reviews.jsonl

python -m ChatUniVi.eval.summarize_gpt_review -f $CKPT/eval/llavabench/reviews.jsonl