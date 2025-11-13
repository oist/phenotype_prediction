#!/bin/bash -l
#SBATCH -p compute
#SBATCH -t 1-0
#SBATCH --mem=128G
#SBATCH -c 8
#SBATCH -n 1

python3 -m aerob_shap_curves.py
