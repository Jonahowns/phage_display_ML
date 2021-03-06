#!/bin/bash

#SBATCH -A ACCOUNT
#SBATCH -n 1
#SBATCH -c CORES
#SBATCH -t WALLTIME
#SBATCH -p PARTITION
#SBATCH -q QUEUE
#SBATCH --gres=gpu:GPUS
#SBATCH -o OUT.%j.out
#SBATCH -e ERR.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=EMAIL     #send to my email
#SBATCH --chdir=WDIR

source activate exmachina3

python MODEL_train.py DATATYPE DATAPATH EPOCHS GPUS WEIGHTS PRECISION

