#!/bin/bash
#SBATCH      --job-name="case_study"
#SBATCH      --mail-user="weiqingh@sas.upenn.edu" # Change to your email
#SBATCH      --time=12:00:00
#SBATCH      --mem=80G
#SBATCH      --gpus=p100  # 22x A100-80, 26x A40-48, 4x V100, 72x P100
#SBATCH      --mail-type=ALL
#SBATCH      --output=slurm_output/output_%A.log

source activate sdoh

cd /cbica/home/hewei/projects/Stq

python case_study.py --epoch 250 --times 15
