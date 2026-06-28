# Models trained with noisy genomes extend bacterial phenotype prediction into deep time

## Overview
Predicting bacterial phenotypes from gene content using noise-robust machine learning, applied to ancestral genome reconstruction. Includes models for oxygen use, cell-wall type, sporulation, optimal growth temperature, and GC content with inference back to the last bacterial common ancestor (LBCA).

## Repository structure
The repo has two main two main directories

- `jupyter_notebooks`

This directory contains all jupyter notebooks with the pipeline for the model trainin on the extant genomes and  noise-robust machine learning, applied to ancestral genomes for each phenotype. In particular, each phenotype has two notebooks:

- `*_extant.ipynb`
- `*_ancestral.ipynb`

- `train_xgboost_on_noisy_data_scripts`

Contains `.py` scripts for batch model training on noise augmented train data in different noise regimes defined by different number of noisy genome copies, and different distributions for the false positive and false negative rates that are applied to the true genome COG counts.

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
