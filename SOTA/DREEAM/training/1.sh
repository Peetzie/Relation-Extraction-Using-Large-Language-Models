#!/bin/bash

# Capture the command-line arguments
CMD=$1
LAMBDA=$2
SEED=$3
SAVE_PATH=$4
META_DIR=$5
DATA_DIR=$6
NUM_CLASS=$7
EPOCHS=$8
LR=$9
MAX_GRAD_NORM=${10}

# Generate a descriptive name for the run
NAME="${CMD}_${LAMBDA}_${EPOCHS}_${LR}_${MAX_GRAD_NORM}_${LAMBDA}"

# Echo the command and arguments for logging/debugging purposes
echo "Running: 1.sh"
echo "Arguments: CMD=$CMD LAMBDA=$LAMBDA SEED=$SEED SAVE_PATH=$SAVE_PATH META_DIR=$META_DIR DATA_DIR=$DATA_DIR NUM_CLASS=$NUM_CLASS EPOCHS=$EPOCHS LR=$LR MAX_GRAD_NORM=$MAX_GRAD_NORM"

# Execute the Python script with the provided arguments
python /work3/s174159/LLM_Thesis/SOTA/DREEAM/run.py --do_train \
    --wandb_project BASELINE \
    --meta_dir "${META_DIR}" \
    --data_dir "${DATA_DIR}" \
    --transformer_type bert \
    --model_name_or_path bert-base-cased \
    --display_name "${NAME}" \
    --save_path "${SAVE_PATH}" \
    --train_file train_annotated.json \
    --dev_file dev.json \
    --train_batch_size 4 \
    --test_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_labels 4 \
    --lr_transformer "${LR}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --evi_thresh "${LAMBDA}" \
    --evi_lambda 0.2 \
    --warmup_ratio 0.06 \
    --num_train_epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --num_class "${NUM_CLASS}"
