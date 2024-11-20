#!/bin/bash
#SBATCH --job-name=transformers
#SBATCH -c 9
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bhav.beri@research.iiit.ac.in

cd ~
source activate anlp
cd ~/anlp/project

models=(bert-tiny bert-mini bert-small bert-medium bert-base bert-large roberta-base roberta-large distilbert-base albert-base albert-large)

# Loop through each model
for model in ${models[@]}; do
    python3 transformer.py --model $model --num_epochs 20
done

