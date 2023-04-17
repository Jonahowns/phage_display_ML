#!/bin/bash

#SBATCH -A jprocyk
#SBATCH -n 1
#SBATCH -c 6
#SBATCH -t 7-00:00
#SBATCH -p wzhengpu1
#SBATCH -q wildfire
#SBATCH --gres=gpu:1
#SBATCH -o ./datasets/exo/submission/out/pcrbm_cluster_exo_reconst.%j.out
#SBATCH -e ./datasets/exo/submission/err/slurm.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=jprocyk@asu.edu     #send to my email
#SBATCH --chdir=/scratch/jprocyk/machine_learning/phage_display_ML/

source activate aptamer

python train.py ./datasets/exo/run_files/exo_pcrbm_cluster_reconst.json

