#!/bin/bash

#SBATCH -A jprocyk
#SBATCH -n 6                  # number of cores
#SBATCH -t 4-00:00               # wall time (D-HH:MM)
#SBATCH -o slurm_rbm_hyperparam_1.%j.out          # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err          # STDERR (%j = JobId)
#SBATCH -p sulcgpu2
#SBATCH -q sulcgpu1
#SBATCH --gres=gpu:3
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=jprocyk@asu.edu     #send to my email
#SBATCH --chdir=/scratch/jprocyk/machine_learning/ML_for_aptamers/rbm_torch/


module load cuda/10.2.89
module load anaconda3/4.4.0

source activate exmachina2

python rbm_hyperparam_optimization.py
