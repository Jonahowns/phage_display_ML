#!/bin/bash

#SBATCH -A jprocyk
#SBATCH -n 48                  # number of cores
#SBATCH -t 6-00:00               # wall time (D-HH:MM)
#SBATCH -o slurm_rbm_hyperparam_hidden_opt_cpu.%j.out          # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err          # STDERR (%j = JobId)
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=jprocyk@asu.edu     #send to my email
#SBATCH --chdir=/scratch/jprocyk/machine_learning/phage_display_ML/rbm_torch/

source activate exmachina3

python rbm_hyperparam_optimization.py cov /scratch/jprocyk/machine_learning/phage_display_ML/cov/r1.fasta 3 100 0 24 12 False

# python rbm_hyperparam_optimization.py FOCUS DATASET_PATH SAMPLES EPOCHS GPUS CPUS DATAWORKERS WEIGHTS
