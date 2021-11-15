#!/bin/bash
 
#SBATCH -A jprocyk
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 7-00:00               # wall time (D-HH:MM)
#SBATCH -o slurm_NAME.%j.out          # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err          # STDERR (%j = JobId)
#SBATCH --mail-type=BEGIN,END,FAIL     # notifications for job done & fail
#SBATCH --mail-user=jprocyk@asu.edu     #send to my email
#SBATCH --chdir=/scratch/jprocyk/machine_learning/phage_display_ML/ProteinMotifRBM/

#module load anaconda2/5.2.0
source activate rbm

python RBM_trainer.py NAME DATA_PATH DESTINATION HIDDEN
