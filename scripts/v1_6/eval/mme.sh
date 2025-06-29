#!/bin/bash

CKPT_DIR="/path/to/checkpoint"
DATA_DIR="/path/to/dataset"

CKPT="llava-v1.6-vicuna-7b"
SPLIT="llava_mme"

TOKEN=${1}
RATIO=${2}

python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/MME/${SPLIT}.jsonl \
    --image-folder ${DATA_DIR}/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.jsonl \
    --visual_token_num ${TOKEN} \
    --important_ratio ${RATIO} \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py \
    --data_path ${DATA_DIR}/MME \
    --experiment ${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}

cd eval_tool

python calculation.py --results_dir answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}
