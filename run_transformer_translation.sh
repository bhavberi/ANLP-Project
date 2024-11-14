#!/bin/bash
#SBATCH --job-name=translation
#SBATCH -c 9
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=transformer_new_languages_job.txt
#SBATCH --mail-user=bhav.beri@research.iiit.ac.in

source activate base
cd ~/ANLP-Project

declare -A languages=(
    ["hindi"]="hin_Deva"
    ["marathi"]="mar_Deva"
    ["kyrgyz"]="kir_Cyrl"
    ["kazakh"]="kaz_Cyrl"
)

model="bert-base-multilingual"

# Iterate over each language fullname and code
for fullname in "${!languages[@]}"; do
    code="${languages[$fullname]}"
    echo "Running for $fullname ($code)"

    # split the code to get the language code
    langcode=$(echo $code | cut -d'_' -f 1)
    
    python3 transformer.py --model $model --num_epochs 20 --csv_path "edos_labelled_aggregated_translated_${fullname}.csv" --translated_text --save_path_suffix "_${langcode}"
done

