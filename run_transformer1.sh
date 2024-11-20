#!/bin/bash
#SBATCH --job-name=transformers
#SBATCH -c 9
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bhav.beri@research.iiit.ac.in

source activate base
cd ~/ANLP-Project

# models=(bert-tiny bert-mini bert-small bert-medium bert-base roberta-base distilbert-base albert-base xlm-roberta-base bert-base-multilingual)
# models=("distilbert/distilbert-base-multilingual-cased" "distilbert/distilbert-base-cased")
models=(bert-base-multilingual distilbert/distilbert-base-multilingual-cased)

# Loop through each model
for model in ${models[@]}; do
    python3 transformer.py --model $model --num_epochs 20 --csv_path edos_labelled_aggregated_translated_french.csv --translated_and_normal --save_path_suffix "_fr-en"
done

python3 transformer.py --model "google-bert/bert-base-multilingual-uncased" --num_epochs 20