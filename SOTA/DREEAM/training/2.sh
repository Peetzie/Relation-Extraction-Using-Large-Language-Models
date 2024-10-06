#!/bin/bash

# Capture the command-line arguments
CMD=$1
LOAD_DIR=$2
LAMBDA=$3
SEED=$4
META_DIR=$5
DATA_DIR=$6
NUM_CLASS=$7


# Generate a descriptive name for the run
NAME="${CMD}_${LAMBDA}_${THRESHOLD}"

# Echo the command and arguments for logging/debugging purposes
echo "Running: 2.sh"
echo "Arguments: CMD=$CMD LOAD_DIR=$LOAD_DIR LAMBDA=$LAMBDA SEED=$SEED META_DIR=$META_DIR DATA_DIR=$DATA_DIR NUM_CLASS=$NUM_CLASS"

# Execute the Python script with the provided arguments
python /work3/s174159/LLM_Thesis/SOTA/DREEAM/run.py \
    --meta_dir "${META_DIR}" \
    --data_dir "${DATA_DIR}" \
    --wandb_project BASELINE \
    --transformer_type bert \
    --model_name_or_path bert-base-cased \
    --display_name "${NAME}" \
    --load_path "${LOAD_DIR}" \
    --eval_mode single \
    --test_file train_distant.json \
    --test_batch_size 4 \
    --evi_thresh 0.2 \
    --num_labels 4 \
    --num_class "${NUM_CLASS}" \
    --save_attn
