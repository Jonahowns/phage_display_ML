#!/bin/bash

#SBATCH -A jprocyk
#SBATCH -n 1
#SBATCH -c 6
#SBATCH -t 7-00:00
#SBATCH -p wzhengpu1
#SBATCH -q wildfire
#SBATCH --gres=gpu:1
#SBATCH -o ./submission/pig_ge2/out/pig_ge2_rbm_m1_c1_m2.%j.out
#SBATCH -e ./submission/pig_ge2/err/slurm.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=jprocyk@asu.edu     #send to my email
#SBATCH --chdir=/scratch/jprocyk/machine_learning/phage_display_ML/rbm_torch/

source activate exmachina3

python rbm_train.py pig_ge2 ../datasets/pig/ge2/m1_c1.fasta 200 1 m1_c1_m2.json double

