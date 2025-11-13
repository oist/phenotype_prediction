import pandas as pd
import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import defaultdict
import argparse
from xgboost import XGBClassifier, XGBRegressor
from sklearn.pipeline import make_pipeline
import torch
from joblib import Parallel, delayed
import joblib, os
from tqdm import tqdm
from utils.utils import read_diderm_data, pca_run_and_plot, tsne_plot

TAX_LEVEL = "phylum" # <----------------- taxonomy level for train/test split is here!!
DATA_DIRECTORY = "data_diderm"  # <-------- input data directory is here!!
NUM_SPLITS_TO_READ = 30  # <-------------- number of splits to read and process is here!!

print(f"NUM_SPLITS_TO_READ = {NUM_SPLITS_TO_READ}")

RANDOM_SEED = 42
OUTPUT_DIRECTORY = f"../{DATA_DIRECTORY}/outputs/{TAX_LEVEL}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Params for the noisy training
parser = argparse.ArgumentParser(description="Noise sampling experiment")
parser.add_argument("--x_noisy_samples", type=int, default=10,
                    help="Number of noisy samples per clean sample")

args = parser.parse_args()

# --- use args ---
x_noisy_samples = args.x_noisy_samples

noise_std = 0.3
hard_fn_flag = True

print(f"Uniform Noise! Calculating for x_noisy_samples = {x_noisy_samples}; noise_std = {noise_std}, and hard_fn_flag = {hard_fn_flag}")

mn_fp_arr = [0.0, 0.05, 0.1, 0.15, 0.2]
mn_fn_arr = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

mean_rem_add_rates_tuples = [(add, rem) for add in mn_fp_arr for rem in mn_fn_arr]


def flip_with_fractional_noise(X: torch.Tensor, fp_rate: float, fn_rate: float,
                               noise_std = 0.3, filename=None, hard_fn_flag = False):
    X_noisy = X.float().clone()
    n_rows, _ = X_noisy.shape
    for i in range(n_rows):

        fp_rate_sampled =  np.random.uniform(0, fp_rate)
        fn_rate_sampled =  np.random.uniform(0, fn_rate)
        
        with open(f"{OUTPUT_DIRECTORY}/trained_models/{filename}", "a") as f:
            f.write(f"{(fp_rate_sampled, fn_rate_sampled)}\n")

        
        # False negatives: subtract (1 + noise) from fraction of positives
        pos_idx = torch.nonzero(X[i] > 0).flatten()

        n_fn = int(round(fn_rate_sampled * len(pos_idx)))
        
        if n_fn > 0:
            fn_idx = pos_idx[torch.randperm(len(pos_idx))[:n_fn]]
            noise = torch.randn(len(fn_idx)) * noise_std
            if hard_fn_flag == False:
                X_noisy[i, fn_idx] -= (1.0 + noise)
            else:
                X_noisy[i, fn_idx] = 0 

        # False positives: add (1 + noise) to fraction of zeros
        zero_idx = torch.nonzero(X[i] == 0).flatten()
        n_fp = int(round(fp_rate_sampled * len(zero_idx)))
        if n_fp > 0:
            fp_idx = zero_idx[torch.randperm(len(zero_idx))[:n_fp]]
            noise = torch.randn(len(fp_idx)) * noise_std
            X_noisy[i, fp_idx] += (1.0 + noise)

    # Clamp to ensure no negatives
    X_noisy = torch.clamp(X_noisy, min=0.0)    
    return X_noisy

def augment_data_at_rate_and_length_uniform(X_val_train, y_label_train, n_clones, fp_rate_max, fn_rate_max, hard_fn_flag = False, noise_std=0.3, filename=None):
    
    X_augmented = X_val_train.clone()
    y_augmented = y_label_train.clone()
    
    if n_clones == 0:
        return X_augmented, y_augmented
        
    for _ in range(n_clones):
        
        y_clone = y_label_train.clone()
        X_noisy = flip_with_fractional_noise(X_val_train, fp_rate_max, fn_rate_max, noise_std = noise_std, filename=filename, hard_fn_flag = hard_fn_flag)
        X_augmented = torch.cat([X_augmented, X_noisy], dim=0)
        y_augmented = torch.cat([y_augmented, y_label_train], dim=0)

    # shuffle the augmented dataset
    idx = torch.randperm(len(X_augmented))
    X_augmented, y_augmented = X_augmented[idx], y_augmented[idx]
    return X_augmented, y_augmented


# --- GPU / CPU setup ---
use_gpu = torch.cuda.is_available()
tree_method = "gpu_hist" if use_gpu else "hist"
predictor = "gpu_predictor" if use_gpu else "auto"

# --- output dir ---
if not os.path.exists(f"{OUTPUT_DIRECTORY}/trained_models"):
    os.makedirs(f"{OUTPUT_DIRECTORY}/trained_models")

# --- model factory ---
def make_xgb():
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        min_child_weight=5,
        tree_method=tree_method,
       # predictor=predictor,
        n_jobs=-1
    )

# --- training function for one split ---
def train_one_split(split_id, fp_rate_mean, fn_rate_mean, x_noisy_samples):
    X_val_train = all_splits_dict[split_id]["X_train"]
    y_label_train = all_splits_dict[split_id]["y_train"]

    filename = f"sampled_unif_fps_fns_fp_mean_{fp_rate_mean}_fn_mean_{fn_rate_mean}_x_{x_noisy_samples}_split_{split_id}.txt"

    
    # augment training set
    X_augmented, y_augmented = augment_data_at_rate_and_length_uniform(
        X_val_train, y_label_train,
        n_clones=x_noisy_samples,
        fp_rate_max=fp_rate_mean, 
        fn_rate_max=fn_rate_mean, 
        hard_fn_flag=hard_fn_flag, noise_std=noise_std, filename = filename
    )

    model = make_xgb()
    pipe_with_noise = make_pipeline(model)
    pipe_with_noise.fit(X_augmented, y_augmented)

    return split_id, pipe_with_noise


# 1. Read the data
all_splits_dict = defaultdict(int)
for split_id in range(NUM_SPLITS_TO_READ):
    # Read train data
    data_filename_train = f"../{DATA_DIRECTORY}/input_data/{TAX_LEVEL}/train_data_{TAX_LEVEL}_tax_level_split_{split_id}"
    y_filename_train = f"../{DATA_DIRECTORY}/input_data/{TAX_LEVEL}/train_annot_{TAX_LEVEL}_tax_level_split_{split_id}"
    taxa_names_filename_train = f"../{DATA_DIRECTORY}/input_data/{TAX_LEVEL}/train_taxa_names_{TAX_LEVEL}_tax_level_split_{split_id}" if TAX_LEVEL != "random" else None
    X_val_train, y_label_train, X_column_names, taxa_group_names_train = read_diderm_data(data_filename_train, y_filename_train, taxa_names_filename_train, DEVICE)
   # X_val_train = (X_val_train > 0).int()
    # Read test data
    data_filename_test = f"../{DATA_DIRECTORY}/input_data/{TAX_LEVEL}/test_data_{TAX_LEVEL}_tax_level_split_{split_id}"
    y_filename_test = f"../{DATA_DIRECTORY}/input_data/{TAX_LEVEL}/test_annot_{TAX_LEVEL}_tax_level_split_{split_id}"
    taxa_names_filename_test = f"../{DATA_DIRECTORY}/input_data/{TAX_LEVEL}/test_taxa_names_{TAX_LEVEL}_tax_level_split_{split_id}" if TAX_LEVEL != "random" else None
    X_val_test, y_label_test, X_column_names, taxa_group_names_test = read_diderm_data(data_filename_test, y_filename_test, taxa_names_filename_test, DEVICE)
   # X_val_test = (X_val_test > 0).int()

    if sum(y_label_test) == 0 or sum(y_label_train) == 0:
        continue

    if TAX_LEVEL == "random":
        taxa_group_names_train = None
        taxa_group_names_test = None
        
    
    curr_split_dict = defaultdict(str)
    curr_split_dict["X_train"] = X_val_train
    curr_split_dict["y_train"] = y_label_train
    curr_split_dict["taxa_group_names_train"] = taxa_group_names_train
    curr_split_dict["X_test"] = X_val_test
    curr_split_dict["y_test"] = y_label_test
    curr_split_dict["taxa_group_names_test"] = taxa_group_names_test
    curr_split_dict["feature_names"] = X_column_names

    all_splits_dict[split_id] = curr_split_dict


    
# --- main loop ---
trained_models_all_noise_rates = {}

for fp_rate_max, fn_rate_max in tqdm(mean_rem_add_rates_tuples, desc="Noise configs..."):
    noise_rates = (fp_rate_max, fn_rate_max)
    print(f"noise_rates = {noise_rates}")

    # Create the txt files for 
    for split_id in all_splits_dict.keys():
        filename = f"sampled_fps_fns_fp_mean_{fp_rate_mean}_fn_mean_{fn_rate_mean}_sigma_{sigma_fp}_x_{x_noisy_samples}_split_{split_id}.txt"
        open(f"{OUTPUT_DIRECTORY}/trained_models/{filename}", "w").close()

    print(f"filename = {filename}")
    
    # parallel over splits
    results = Parallel(n_jobs=min(4, os.cpu_count()), backend="threading")(
        delayed(train_one_split)(
            split_id, fp_rate_max, fn_rate_max, x_noisy_samples
        )
        for split_id in all_splits_dict.keys()
    )

    # collect models
    trained_models_local = {split_id: model for split_id, model in results}
    trained_models_all_noise_rates[noise_rates] = trained_models_local

    # save per noise config
    filename = f"trained_models_unif_fp_{fp_rate_max}_fn_{fn_rate_max}_x_{x_noisy_samples}.pkl"
    joblib.dump(trained_models_local, f"{OUTPUT_DIRECTORY}/trained_models/{filename}")

    print(f"✅ Saved {len(trained_models_local)} models for noise {noise_rates}")





    