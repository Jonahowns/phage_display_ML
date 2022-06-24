#!/bin/bash

#SBATCH -A jprocyk
#SBATCH -n 1
#SBATCH -c 6
#SBATCH -t 4-00:00
#SBATCH -p amciigpu1
#SBATCH -q wildfire
#SBATCH --gres=gpu:1
#SBATCH -o pig_ge2_crbm_b3_c2_w.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=jprocyk@asu.edu     #send to my email
#SBATCH --chdir=/scratch/jprocyk/machine_learning/phage_display_ML/rbm_torch/

source activate exmachina3

python crbm_train.py pig_ge2 ../datasets/pig/ge2/b3_c2.fasta 200 1 True single

