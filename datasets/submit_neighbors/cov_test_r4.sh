#!/bin/bash
 
#SBATCH -A jprocyk
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 7-00:00               # wall time (D-HH:MM)
#SBATCH -p serial
#SBATCH -q normal
#SBATCH -o slurm.%j.out          # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err          # STDERR (%j = JobId)
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=jprocyk@asu.edu     #send to my email
#SBATCH --chdir=/scratch/jprocyk/machine_learning/phage_display_ML/datasets/

source activate exmachina3

python pairwise_distances.py ./cov/ r4.fasta dna 24 0.15 0.25
