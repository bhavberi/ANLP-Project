#!/bin/bash
#SBATCH --job-name=transformers
#SBATCH -c 9
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maharnav.singhal@research.iiit.ac.in

source activate base
cd ~/ANLP-Project

models=(bert-base distilbert-base)

# Loop through each model
for model in ${models[@]}; do
    python3 transformer.py --model $model --num_epochs 20 --freeze
done

for model in ${models[@]}; do
    python3 transformer.py --model $model --num_epochs 20 --lora
done
