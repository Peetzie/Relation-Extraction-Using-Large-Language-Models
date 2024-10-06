#!/bin/bash

# Capture the command-line arguments
CMD=$1
TEACHER_DIR=$2
LAMBDA=$3
SEED=$4
SAVE_PATH=$5
META_DIR=$6
DATA_DIR=$7
NUM_CLASS=$8
EPOCHS=$9
LR=${10}
MAX_GRAD_NORM=${11}

# Generate a descriptive name for the run
NAME="${CMD}_lambda${LAMBDA}_${SEED}"

# Echo the command and arguments for logging/debugging purposes
echo "Running: 3.sh"
echo "Arguments: CMD=$CMD TEACHER_DIR=$TEACHER_DIR LAMBDA=$LAMBDA SEED=$SEED SAVE_PATH=$SAVE_PATH META_DIR=$META_DIR DATA_DIR=$DATA_DIR NUM_CLASS=$NUM_CLASS EPOCHS=$EPOCHS LR=$LR MAX_GRAD_NORM=$MAX_GRAD_NORM"

# Execute the Python script with the provided arguments
python /work3/s174159/LLM_Thesis/SOTA/DREEAM/run.py --do_train \
    --meta_dir "${META_DIR}" \
    --data_dir "${DATA_DIR}" \
    --transformer_type bert \
    --model_name_or_path bert-base-cased \
    --display_name "${NAME}" \
    --train_file train_distant.json \
    --dev_file dev.json \
    --teacher_sig_path "${TEACHER_DIR}" \
    --save_path "${SAVE_PATH}" \
    --train_batch_size 4 \
    --test_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_steps 5000 \
    --num_labels 4 \
    --lr_transformer "${LR}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --evi_thresh 0.2 \
    --attn_lambda "${LAMBDA}" \
    --warmup_ratio 0.06 \
    --num_train_epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --num_class "${NUM_CLASS}"
