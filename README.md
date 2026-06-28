# Models trained with noisy genomes extend bacterial phenotype prediction into deep time

## Overview
Predicting bacterial phenotypes from gene content using noise-robust machine learning, applied to ancestral genome reconstruction. Includes models for oxygen use, cell-wall type, sporulation, optimal growth temperature, and GC content, with inference back to the last bacterial common ancestor (LBCA).

## Repository structure

### `jupyter_notebooks/`
Contains the full prediction pipeline for each phenotype. Each phenotype has two notebooks:

- `*_extant.ipynb` — model training and evaluation on extant genomes
- `*_ancestral.ipynb` — applying noise-robust models to reconstructed ancestral genomes

The following utility modules are shared across notebooks:
- `feature_selection/feature_selection_utils.py` — feature selection methods (SHAP, mutual information, Markov blankets)
- `utils/utils.py` — data loading, model training, and evaluation utilities
- `utils/utils_ancestral_predict.py` — utilities specific to ancestral prediction

### `train_xgboost_on_noisy_data_scripts/`
Batch training scripts for running model training on noise-augmented data across different noise regimes — varying the number of noisy genome copies per training sample and the false-positive/false-negative rate distributions applied to COG count profiles.

## Data

All the jupyter notebooks and the ba

## Installation
git clone + pip install, any non-obvious setup steps.

## Usage
The 3–5 most common things someone would actually run, with minimal working examples.

## Methods
Brief descriptions of the key algorithmic components 
(e.g. feature selection methods, model architecture, CV strategy).

## Results
Optional — a table or figure of main results if the paper isn't out yet.

## Citation
BibTeX block or "preprint coming soon".

## License
One line.
