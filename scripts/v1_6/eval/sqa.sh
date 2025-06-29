#!/bin/bash

CKPT_DIR="/path/to/checkpoint"
DATA_DIR="/path/to/dataset"

CKPT="llava-v1.6-vicuna-7b"
SPLIT="llava_test_CQM-I"

TOKEN=${1}
RATIO=${2}

python -m llava.eval.model_vqa_science \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/scienceqa/${SPLIT}.json \
    --image-folder ${DATA_DIR}/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.jsonl \
    --visual_token_num ${TOKEN} \
    --important_ratio ${RATIO} \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_science_qa \
    --base-dir ${DATA_DIR}/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}_result.json
