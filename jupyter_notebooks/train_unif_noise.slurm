#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -t 1-0
#SBATCH --mem=256G
#SBATCH -c 12
#SBATCH -n 1

# Define the values you want to test
x_values=("0" "5" "10" "20" "50" "100")

# Loop over all combinations
for x in "${x_values[@]}"; do
  echo "Running with x=$x"
  python3 -m train_xgboost_unif_aerob.py \
    --x_noisy_samples "$x" 
  done
done

