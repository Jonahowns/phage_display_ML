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

python rbm_hyperparam_optimization.py /scratch/jprocyk/machine_learning/phage_display_ML/pig_tissue/b3_c1.fasta 22 protein 3 100 0 12 6

# python rbm_hyperparam_optimization.py DATASET_PATH VISIBLE_NUM MOLECULE SAMPLES EPOCHS GPUS CPUS DATAWORKERS
