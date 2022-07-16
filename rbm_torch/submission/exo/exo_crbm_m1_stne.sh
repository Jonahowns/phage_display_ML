#!/bin/bash

#SBATCH -A jprocyk
#SBATCH -n 1
#SBATCH -c 6
#SBATCH -t 7-00:00
#SBATCH -p wzhengpu1
#SBATCH -q wildfire
#SBATCH --gres=gpu:2
#SBATCH -o ./submission/exo/out/exo_crbm_m1_stne.%j.out
#SBATCH -e ./submission/exo/err/slurm.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=jprocyk@asu.edu     #send to my email
#SBATCH --chdir=/scratch/jprocyk/machine_learning/phage_display_ML/rbm_torch/

source activate exmachina3

python crbm_train.py exo ../datasets/exo/m1.fasta 1000 2 m1_stne.json single

