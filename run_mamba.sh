#!/bin/bash
#SBATCH --job-name=mamba
#SBATCH -c 36
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maharnav.singhal@research.iiit.ac.in

source activate base
cd ~/ANLP-Project

# models=(bert-tiny bert-mini bert-small bert-medium bert-base roberta-base distilbert-base albert-base xlm-roberta-base bert-base-multilingual)
models=("mamba-tiny" "mamba-mini" "mamba-small" "mamba-medium" "mamba-large")

# Loop through each model
for model in ${models[@]}; do
    python3 mamba.py --model $model --num_epochs 20
done