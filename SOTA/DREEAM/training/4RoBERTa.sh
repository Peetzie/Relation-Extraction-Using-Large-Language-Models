#!/bin/bash

# Capture the command-line arguments
CMD=$1
LOAD_DIR=$2
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
echo "Running: 4.sh"
echo "Arguments: CMD=$CMD LOAD_DIR=$LOAD_DIR LAMBDA=$LAMBDA SEED=$SEED SAVE_PATH=$SAVE_PATH META_DIR=$META_DIR DATA_DIR=$DATA_DIR NUM_CLASS=$NUM_CLASS EPOCHS=$EPOCHS LR=$LR MAX_GRAD_NORM=$MAX_GRAD_NORM"

# Execute the Python script with the provided arguments
python /work3/s174159/LLM_Thesis/SOTA/DREEAM/run.py --do_train \
    --meta_dir "${META_DIR}" \
    --data_dir "${DATA_DIR}" \
    --transformer_type roberta \
    --model_name_or_path roberta-large \
    --display_name "${NAME}" \
    --train_file train_annotated.json \
    --dev_file dev.json \
    --save_path "${SAVE_PATH}" \
    --load_path "${LOAD_DIR}" \
    --train_batch_size 4 \
    --test_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_labels 4 \
    --lr_transformer "${LR}" \
    --lr_added 3e-6 \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --evi_thresh 0.2 \
    --evi_lambda "${LAMBDA}" \
    --warmup_ratio 0.06 \
    --num_train_epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --num_class "${NUM_CLASS}"
