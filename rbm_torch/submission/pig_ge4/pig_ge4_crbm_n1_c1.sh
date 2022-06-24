#!/bin/bash

#SBATCH -A jprocyk
#SBATCH -n 1
#SBATCH -c 6
#SBATCH -t 4-00:00
#SBATCH -p wzhengpu1
#SBATCH -q wildfire
#SBATCH --gres=gpu:1
#SBATCH -o pig_ge4_crbm_n1_c1.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=jprocyk@asu.edu     #send to my email
#SBATCH --chdir=/scratch/jprocyk/machine_learning/phage_display_ML/rbm_torch/

source activate exmachina3

python crbm_train.py pig_ge4 ../datasets/pig/ge4/n1_c1.fasta 200 1 False single

