#!/bin/bash

#SBATCH -A jprocyk
#SBATCH -n 1
#SBATCH -c 6
#SBATCH -t 4-00:00
#SBATCH -p amciigpu1
#SBATCH -q wildfire
#SBATCH --gres=gpu:2
#SBATCH -o pig_ge2_crbm_np2_c2_w.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=jprocyk@asu.edu     #send to my email
#SBATCH --chdir=/scratch/jprocyk/machine_learning/phage_display_ML/rbm_torch/

source activate exmachina3

python crbm_train.py pig_ge2 ../pig_tissue/gaps_end_2_clusters/np2_c2.fasta 200 2 True
