#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=60G               # memory (per node)
#SBATCH --time=0-04:00            # time (DD-HH:MM)

###########cluster information above this line


###load environment 
module load anaconda/3
module load cuda/10.1
conda activate phyre



python GenerateData.py