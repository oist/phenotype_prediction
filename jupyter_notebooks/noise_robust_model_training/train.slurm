#!/bin/bash -l
#SBATCH -p compute
#SBATCH -t 1-0
#SBATCH --mem=256G
#SBATCH -c 8
#SBATCH -n 1

# Define the values you want to test
noise_types=("exp" "unif" "gamma")

# Loop over all combinations
for noise_type in "${noise_types[@]}"; do
  echo "Running with noise_type=$noise_type"
  python3 -m train_xgboost_on_noisy_data.py \
    --x_noisy_samples 400 \
    --noise_type "$noise_type"
  done
done

