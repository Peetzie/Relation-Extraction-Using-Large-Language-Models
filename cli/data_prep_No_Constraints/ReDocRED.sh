#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J VERIFY
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 06:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=10GB]"
### -- set the email address --
##SUB -u s174159@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o "/work3/s174159/LLM_Thesis/out/prepping/Verified_No_Constraint/ReDocRED_%J.out" 
#BSUB -e "/work3/s174159/LLM_Thesis/out/prepping/Verified_No_Constraint/ReDocRED%J.err"
# -- end of LSF options --

source /work3/s174159/LLM_Thesis/databuilding/.venv/bin/activate

python /work3/s174159/LLM_Thesis/databuilding/components/scripts/verify.py --input_file "/work3/s174159/LLM_Thesis/data/No_Constraints/ReDocRED/combined_dataset_corrected.json" --output_dir "/work3/s174159/LLM_Thesis/data/No_Constraints/ReDocRED" --model_name bert-base-cased --max_seq_length 1024
