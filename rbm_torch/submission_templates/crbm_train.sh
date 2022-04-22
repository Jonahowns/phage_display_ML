#!/bin/bash
 
#SBATCH -A jprocyk
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 7-00:00               # wall time (D-HH:MM)
#SBATCH -p PARTITION
#SBATCH -q QUEUE
#SBATCH --gres=gpu:GPU_NUM
#SBATCH -o slurm_NAME.%j.out          # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err          # STDERR (%j = JobId)
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=jprocyk@asu.edu     #send to my email
#SBATCH --chdir=/scratch/jprocyk/machine_learning/phage_display_ML/rbm_torch/

source activate exmachina3

python crbm_train.py FOCUS DATA_PATH EPOCHS GPU_NUM WEIGHTS

#python crbm_train.sh focus Data_Path Epochs Gpus Weights
