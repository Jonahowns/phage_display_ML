#!/bin/bash

#SBATCH -A jprocyk
#SBATCH -n 1                  # number of cores
#SBATCH -c 24
#SBATCH -t 0-04:00               # wall time (D-HH:MM)
#SBATCH -o slurm_rbm_hyperparam_hidden_opt_cov_r1.%j.out          # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err          # STDERR (%j = JobId)
#SBATCH -p htcgpu                            #sulcgpu2
#SBATCH -q normal                            #sulcgpu1
#SBATCH -C A100
#SBATCH --gres=gpu:2
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=jprocyk@asu.edu     #send to my email
#SBATCH --chdir=/scratch/jprocyk/machine_learning/phage_display_ML/rbm_torch/

source activate exmachina3

python rbm_hyperparam_optimization.py pig /scratch/jprocyk/machine_learning/phage_display_ML/pig_tissue/np3_c2.fasta 2 100 1 12 6 False

# python rbm_hyperparam_optimization.py FOCUS DATASET_PATH SAMPLES EPOCHS GPUS CPUS DATAWORKERS WEIGHTS
