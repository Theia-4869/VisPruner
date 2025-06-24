#!/bin/bash

CKPT_DIR="/path/to/checkpoint"
DATA_DIR="/path/to/dataset"

CKPT="llava-v1.6-vicuna-7b"
SPLIT="llava_test"

TOKEN=${1}
RATIO=${2}

python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/vizwiz/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.jsonl \
    --visual_token_num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/${SPLIT}.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.json
