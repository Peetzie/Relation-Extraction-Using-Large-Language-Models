#!/bin/bash

# Capture the command-line arguments
NAME=$1
MODEL_DIR=$2 # Should be the student model directory
SPLIT=$3
NUM_CLASS=$4
DATA_DIR=$5
META_DIR=$6

# Echo the command and arguments for logging/debugging purposes
echo "Running: eval.sh"
echo "Arguments: NAME=$NAME MODEL_DIR=$MODEL_DIR SPLIT=$SPLIT NUM_CLASS=$NUM_CLASS DATA_DIR=$DATA_DIR META_DIR=$META_DIR"

# Execute the Python script with single evaluation mode
python /work3/s174159/LLM_Thesis/SOTA/DREEAM/run.py --data_dir "${DATA_DIR}" \
--meta_dir "${META_DIR}" \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--display_name "${NAME}" \
--load_path "${MODEL_DIR}" \
--eval_mode single \
--test_file "${SPLIT}.json" \
--test_batch_size 8 \
--num_labels 4 \
--evi_thresh 0.2 \
--num_class "${NUM_CLASS}" \
--do_test

# Execute the Python script with fusion evaluation mode
python /work3/s174159/LLM_Thesis/SOTA/DREEAM/run.py --data_dir "${DATA_DIR}" \
--meta_dir "${META_DIR}" \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--display_name "${NAME}" \
--load_path "${MODEL_DIR}" \
--eval_mode fushion \
--test_file "${SPLIT}.json" \
--test_batch_size 8 \
--num_labels 4 \
--evi_thresh 0.2 \
--num_class "${NUM_CLASS}"\
--do_test
