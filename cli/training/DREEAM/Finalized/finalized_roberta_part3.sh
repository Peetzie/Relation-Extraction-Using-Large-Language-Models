#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J Finalized_Roberta
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=90GB]"
### -- set the email address --
##SUB -u s174159@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o "/work3/s174159/LLM_Thesis/out/training_DREEAM/roberta/Finalized_%J.out" 
#BSUB -e "/work3/s174159/LLM_Thesis/out/training_DREEAM/roberta/Finalized_%J.err"
# -- end of LSF options --


source /work3/s174159/LLM_Thesis/SOTA/DREEAM/.venv/bin/activate
cd /work3/s174159/LLM_Thesis/SOTA/DREEAM/training

python /work3/s174159/LLM_Thesis/SOTA/DREEAM/training/finalized/finalized_RoBERTa.py -p 3


