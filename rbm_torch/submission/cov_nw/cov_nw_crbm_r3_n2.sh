#!/bin/bash

#SBATCH -A jprocyk
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 7-00:00
#SBATCH -p wzhengpu1
#SBATCH -q wildfire
#SBATCH --gres=gpu:2
#SBATCH -o cov_nw_crbm_r3_n2.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=jprocyk@asu.edu     #send to my email
#SBATCH --chdir=/scratch/jprocyk/machine_learning/phage_display_ML/rbm_torch/

source activate exmachina3

python crbm_train.py cov_nw ../datasets/cov/nw/r3.fasta 200 2 r3_n2_weights.json single

