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

### `tree_stats_visualization/`
A script for visualising tree statistics across node ages. It plots the prediction confidence for binary classifiers, and the distributions of the predicted values (OGT, GC content) for the regressors.

How to run:

```bash
$ python3 plot_tree_stats_over_time.py
```

The input `.tree` files are in the `tree/` subdirectory.

## Data

All the jupyter notebooks and the ba

Specify all the inputs/outputs!

## Installation
git clone + pip install, any non-obvious setup steps...

## Methods

### Data representation
Each genome is represented as a vector of COG (Clusters of Orthologous Groups) copy numbers, assigned to GTDB genomes using eggNOG-mapper. This representation provides a compact summary of functional gene content and is directly compatible with ancestral genome reconstruction methods based on phylogenetic reconciliation.

### Taxonomy-aware train/test splits
To evaluate generalization across evolutionary distances, genomes are split into training and test sets at five taxonomic levels (family, order, class, phylum, and domain), ensuring that all genomes from a given taxonomic group are assigned exclusively to either the training or the test set. Thirty independent splits are generated per taxonomic level. Performance is reported as Matthews Correlation Coefficient (MCC), supplemented by balanced accuracy, recall, and F1 score.

### Models
Binary phenotypes (oxygen use, cell envelope, sporulation) are predicted using an XGBoost classifier. Continuous phenotypes (optimal growth temperature and GC content) use a two-stage mixture-of-experts approach: an XGBoost classifier first partitions genomes into two groups (mesophile/thermophile for OGT; low/high GC), and two separate XGBoost regressors are trained on each group. The final prediction is a probability-weighted combination of the two regressors.

### Feature selection
Two complementary feature selection methods are applied to each phenotype:

- **SHAP** (SHapley Additive exPlanations): quantifies each feature's contribution to the model's predictions by measuring its effect across all possible feature subsets. Features are ranked by mean absolute SHAP value across the training set.
- **Markov Blanket (IAMB algorithm)**: identifies the minimal set of features that renders the target phenotype conditionally independent of all remaining features. This is a model-agnostic method based on conditional mutual information, implemented using the Incremental Association Markov Blanket (IAMB) algorithm.

### Noise-robust model training
Ancestral genomes are reconstructed probabilistically and carry increasing uncertainty toward the root. To train models robust to this noise, each training genome is augmented with *x* noisy copies, where noise is introduced by:

- **False negatives**: subtracting Poisson(*g* · *r*_FN) counts from each gene count *g*, simulating gene loss.
- **False positives**: adding Poisson(*r*_FP) counts to all genes, simulating spurious gene presence.

Noise rates are sampled from predefined distributions (uniform, exponential, or gamma) parameterized by their mean values λ_FN and λ_FP. For each phenotype, augmentation configurations are evaluated across a grid of (*x*, λ_FN, λ_FP) values and the scheme maximizing robustness on noisy test data is selected.

### Ancestral phenotype prediction
Noise-robust models trained on extant genomes are applied to reconstructed ancestral genomes across a 1,007-species bacterial phylogeny. Ancestral COG copy numbers are inferred by phylogenetic reconciliation. Model confidence is tracked as a function of node age (from extant tips to the root) to assess how far back each phenotype can be reliably predicted.


## Citation
preprint coming soon...

