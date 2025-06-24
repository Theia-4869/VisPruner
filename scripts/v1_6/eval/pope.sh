#!/bin/bash

CKPT_DIR="/path/to/checkpoint"
DATA_DIR="/path/to/dataset"

CKPT="llava-v1.6-vicuna-7b"
SPLIT="llava_pope_test"

TOKEN=${1}
RATIO=${2}

python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.jsonl \
    --visual_token_num ${TOKEN} \
    --important_ratio ${RATIO} \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ${DATA_DIR}/pope/coco \
    --question-file ./playground/data/eval/pope/${SPLIT}.jsonl \
    --result-file ./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.jsonl
