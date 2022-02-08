#!/bin/bash

#SBATCH -A jprocyk
#SBATCH -n 8                  # number of cores
#SBATCH -t 0-04:00               # wall time (D-HH:MM)
#SBATCH -o slurm_rbm_hyperparam_hidden_opt.%j.out          # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err          # STDERR (%j = JobId)
#SBATCH -p htcgpu                            #sulcgpu2
#SBATCH -q normal                            #sulcgpu1
#SBATCH -C A100
#SBATCH --gres=gpu:4
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=jprocyk@asu.edu     #send to my email
#SBATCH --chdir=/scratch/jprocyk/machine_learning/phage_display_ML/rbm_torch/

module purge
module load cuda/11.3.0
module load anaconda3/4.4.0

conda activate machina

python rbm_hyperparam_optimization.py
