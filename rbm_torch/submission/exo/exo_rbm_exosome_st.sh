#!/bin/bash

#SBATCH -A jprocyk
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 7-00:00
#SBATCH -p wzhengpu1
#SBATCH -q wildfire
#SBATCH --gres=gpu:1
#SBATCH -o exo_rbm_exosome_st.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=jprocyk@asu.edu     #send to my email
#SBATCH --chdir=/scratch/jprocyk/machine_learning/phage_display_ML/rbm_torch/

source activate exmachina3

python rbm_train.py exo ../datasets/exo/exosome.fasta 200 1 exosome_st_weights.json single

