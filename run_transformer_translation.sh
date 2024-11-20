#!/bin/bash
#SBATCH --job-name=translation
#SBATCH -c 9
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=transformer_new_languages_job_uncased.txt
#SBATCH --mail-user=bhav.beri@research.iiit.ac.in

source activate base
cd ~/ANLP-Project

declare -A languages=(
    ["hindi"]="_hin"
    ["marathi"]="_mar"
    ["kyrgyz"]="_kir"
    ["polish"]="_pol"
    ["kazakh"]="_kaz"
    ["french"]="_fre"
    ["spanish"]="_spa"
)

model="google-bert/bert-base-multilingual-uncased"

# Iterate over each language fullname and code
for fullname in "${!languages[@]}"; do
    langcode="${languages[$fullname]}"
    echo "Running for $fullname ($langcode)"
    
    python3 transformer.py --model $model --num_epochs 20 --csv_path "edos_labelled_aggregated_translated_${fullname}.csv" --translated_text --save_path_suffix "${langcode}"
done

