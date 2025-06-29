#!/bin/bash

CKPT_DIR="/path/to/checkpoint"
DATA_DIR="/path/to/dataset"

CKPT="llava-v1.5-7b"
SPLIT="llava-mm-vet"

TOKEN=${1}
RATIO=${2}

python -m llava.eval.model_vqa \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/mm-vet/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.jsonl \
    --visual_token_num ${TOKEN} \
    --important_ratio ${RATIO} \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/answers_upload/${SPLIT}/${CKPT}/n_${TOKEN}

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.jsonl \
    --dst ./playground/data/eval/mm-vet/answers_upload/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.json

