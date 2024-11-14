#!/bin/bash
#SBATCH --job-name=translation
#SBATCH -c 9
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=translation_job.txt
#SBATCH --mail-user=bhav.beri@research.iiit.ac.in

source activate base
cd ~/ANLP-Project

declare -A languages=(
    ["hindi"]="hin_Deva"
    ["marathi"]="mar_Deva"
    ["kyrgyz"]="kir_Cyrl"
    ["kazakh"]="kaz_Cyrl"
)

# Path to your Python script
PYTHON_SCRIPT="translation.py"

# Iterate over each language fullname and code
for fullname in "${!languages[@]}"; do
    code="${languages[$fullname]}"
    echo "Translating to $fullname ($code)"
    
    # Run the Python script with the language code and fullname
    python3 "$PYTHON_SCRIPT" --to "$code" --fullname "$fullname"
done

