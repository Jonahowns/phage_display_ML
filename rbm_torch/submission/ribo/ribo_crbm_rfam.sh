#!/bin/bash

#SBATCH -A jprocyk
#SBATCH -n 1
#SBATCH -c 6
#SBATCH -t 7-00:00
#SBATCH -p sjayasurgpu1
#SBATCH -q wildfire
#SBATCH --gres=gpu:2
#SBATCH -o ribo_crbm_rfam.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=jprocyk@asu.edu     #send to my email
#SBATCH --chdir=/scratch/jprocyk/machine_learning/phage_display_ML/rbm_torch/

source activate exmachina3

python crbm_train.py ribo ../datasets/ribo/rfam.fasta 200 2 False double

