#!/bin/bash
#SBATCH --job-name=setfit
#SBATCH -c 9
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maharnav.singhal@research.iiit.ac.in
#SBATCH --output=setfit.log

source activate base
mkdir -p /scratch/maharnav.singhal/setfit/
cd /scratch/maharnav.singhal/setfit/
scp maharnav.singhal@ada.iiit.ac.in:~/ANLP-Project/setfit.py .
scp maharnav.singhal@ada.iiit.ac.in:~/ANLP-Project/edos_labelled_aggregated.csv .
scp -r maharnav.singhal@ada.iiit.ac.in:~/ANLP-Project/utils .

models=(paraphrase-mpnet-base-v2 paraphrase-MiniLM-L3-v2 paraphrase-MiniLM-L6-v2 all-mpnet-base-v2 all-MiniLM-L6-v2 all-MiniLM-L12-v2 all-distilroberta-v1 all-roberta-large-v1)

# Loop through each model
for model in ${models[@]}; do
    python3 setfit.py --model $model --num_epochs 20 --num_iterations 20
done

scp -r /scratch/maharnav.singhal/setfit/ maharnav.singhal@ada.iiit.ac.in:/share1/maharnav.singhal