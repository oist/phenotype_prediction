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
NUM_SPLITS_TO_READ = 5#30  # <-------------- number of splits to read and process is here!!

print(f"NUM_SPLITS_TO_READ = {NUM_SPLITS_TO_READ}")

RANDOM_SEED = 42
OUTPUT_DIRECTORY = f"../{DATA_DIRECTORY}/outputs/{TAX_LEVEL}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINED_MOD_DIR = "trained_models_train_val_test"

# Params for the noisy training

parser = argparse.ArgumentParser(description="Noise sampling experiment")
parser.add_argument("--x_noisy_samples", type=int, default=10,
                    help="Number of noisy samples per clean sample")
parser.add_argument("--noise_type", type=str, default=None,
                    help="Noise distribution")
# parser.add_argument("--sigma_fn", type=float, default=0.2,
#                     help="Std deviation for FN sampling distribution")
# parser.add_argument("--sigma_fp", type=float, default=0.2,
#                     help="Std deviation for FP sampling distribution")

args = parser.parse_args()

# --- use args ---
x_noisy_samples = args.x_noisy_samples
noise_type = args.noise_type
# sigma_fn = args.sigma_fn
# sigma_fp = args.sigma_fp

noise_std = 0.3

print(f"Calculating for x_noisy_samples = {x_noisy_samples}; noise_type = {noise_type}")

mn_fp_arr =  [0.0, 0.05, 0.1, 0.15, 0.2, 0.5]
mn_fn_arr = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0] 



mean_rem_add_rates_tuples = [(add, rem) for add in mn_fp_arr for rem in mn_fn_arr]



# def sample_rate(mean, sigma, high_lim = 1, size=1):
#     if sigma == 0:
#         return np.full(size, mean)[0]
    
#     # Define boundaries: [0, 2*mean]
#     low, high = 0, high_lim#2 * mean
    
#     # Draw normal around mean, truncate to [0, 2*mean]
#     samples = np.random.normal(mean, sigma, size * 10)  # oversample
#     samples = samples[(samples >= low) & (samples <= high)]
    
#     if len(samples) < size:
#        # fallback to uniform if truncation fails
#         return np.random.uniform(low, high, size)[0]
    
#     return np.random.choice(samples, size=size, replace=False)[0]

# def flip_with_fractional_noise(X: torch.Tensor, fp_rate_mean: float, sigma_fp: float, fn_rate_mean: float, sigma_fn: float,
#                                noise_std = 0.3, filename=None, hard_fn_flag = False):
#     X_noisy = X.float().clone()
#     n_rows, _ = X_noisy.shape
#     for i in range(n_rows):

#         fp_rate_sampled = sample_rate(fp_rate_mean, sigma_fp, high_lim = 0.2)
#         fn_rate_sampled = sample_rate(fn_rate_mean, sigma_fn, high_lim = 1)

#         with open(f"{OUTPUT_DIRECTORY}/trained_models/{filename}", "a") as f:
#             f.write(f"{(fp_rate_sampled, fn_rate_sampled)}\n")

        
#         # False negatives: subtract (1 + noise) from fraction of positives
#         pos_idx = torch.nonzero(X[i] > 0).flatten()

#         n_fn = int(round(fn_rate_sampled * len(pos_idx)))
        
#         if n_fn > 0:
#             fn_idx = pos_idx[torch.randperm(len(pos_idx))[:n_fn]]
#             noise = torch.randn(len(fn_idx)) * noise_std
#             if hard_fn_flag == False:
#                 X_noisy[i, fn_idx] -= (1.0 + noise)
#             else:
#                 X_noisy[i, fn_idx] = 0 

#         # False positives: add (1 + noise) to fraction of zeros
#         zero_idx = torch.nonzero(X[i] == 0).flatten()
#         n_fp = int(round(fp_rate_sampled * len(zero_idx)))
#         if n_fp > 0:
#             fp_idx = zero_idx[torch.randperm(len(zero_idx))[:n_fp]]
#             noise = torch.randn(len(fp_idx)) * noise_std
#             X_noisy[i, fp_idx] += (1.0 + noise)

#     # Clamp to ensure no negatives
#     X_noisy = torch.clamp(X_noisy, min=0.0)    
#     return X_noisy

# def augment_data_at_rate_and_length(X_val_train, y_label_train, n_clones, fp_rate_mean, sigma_fp, fn_rate_mean, sigma_fn, hard_fn_flag = False, noise_std=0.3, filename=None):
    
#     X_augmented = X_val_train.clone()
#     y_augmented = y_label_train.clone()
    
#     if n_clones == 0:
#         return X_augmented, y_augmented
        
#     for _ in range(n_clones):
    
#         y_clone = y_label_train.clone()
#         X_noisy = flip_with_fractional_noise(X_val_train, fp_rate_mean, sigma_fp, fn_rate_mean, sigma_fn, noise_std = noise_std, filename=filename, hard_fn_flag = hard_fn_flag)
#         X_augmented = torch.cat([X_augmented, X_noisy], dim=0)
#         y_augmented = torch.cat([y_augmented, y_label_train], dim=0)

#     # shuffle the augmented dataset
#     idx = torch.randperm(len(X_augmented))
#     X_augmented, y_augmented = X_augmented[idx], y_augmented[idx]
#     return X_augmented, y_augmented
    
def apply_noise(genome, fp_rate=0.2, fn_rate=0.5):
    genome_noisy = genome.float().clone()

    # False negatives (multiple hits per count, Poisson distributed)
    losses = torch.poisson(genome.float() * fn_rate)
    genome_noisy = torch.clamp(genome_noisy - losses.int(), min=0)

    # False positives (Poisson noise added to zeros only)
    fp_add = torch.zeros_like(genome)
    zero_mask = genome > -1
    fp_add[zero_mask] = torch.poisson(torch.full((zero_mask.sum(),), fp_rate))
    genome_noisy = genome_noisy + fp_add

    return genome_noisy

def sample_exp(mean):
    sample = np.random.exponential(scale=mean)  # scale = 1/λ
    return sample

def sample_unif(mean):
    sample = np.random.uniform(0, 2*mean)
    return sample

def sample_gamma(mean):#(shape, scale):
    scale = 0.5
    shape = mean/scale
    sample = np.random.gamma(shape, scale)
    return sample
    
def augment_data_with_noise(X_val_train, y_label_train, n_clones, mean_fp, mean_fn, noise_type=None, filename=None):
    # Start with the original data as lists
    X_augmented_list = [X_val_train]
    y_augmented_list = [y_label_train]

    if n_clones == 0:
        return X_val_train.clone(), y_label_train.clone()

    n_rows, _ = X_val_train.shape
    for i in range(n_rows):   
        genome = X_val_train[i]
        label = y_label_train[i]
        for _ in range(n_clones):

            # Sample FP/FN rates based on noise_type
            if noise_type == "exp":
                fp_rate_sampled = sample_exp(mean_fp)
                fn_rate_sampled = sample_exp(mean_fn)
            elif noise_type == "gamma":  
                fp_rate_sampled = sample_gamma(mean_fp)
                fn_rate_sampled = sample_gamma(mean_fn)
            elif noise_type == "unif":     
                fp_rate_sampled = sample_unif(mean_fp)
                fn_rate_sampled = sample_unif(mean_fn) 
            else:
                print(f"Incorrect noise type!")

            # Apply noise
            genome_noisy = apply_noise(genome, fp_rate=fp_rate_sampled, fn_rate=fn_rate_sampled)
            
            # Append to lists
            X_augmented_list.append(genome_noisy.unsqueeze(0))
            y_augmented_list.append(label.unsqueeze(0))

    # Stack / concatenate once at the end
    X_augmented = torch.cat(X_augmented_list, dim=0)
    y_augmented = torch.cat(y_augmented_list, dim=0)

    # Shuffle the augmented dataset
    idx = torch.randperm(len(X_augmented))
    X_augmented, y_augmented = X_augmented[idx], y_augmented[idx]

    return X_augmented, y_augmented



# --- GPU / CPU setup ---
use_gpu = torch.cuda.is_available()
tree_method = "gpu_hist" if use_gpu else "hist"
predictor = "gpu_predictor" if use_gpu else "auto"

# --- output dir ---
if not os.path.exists(f"{OUTPUT_DIRECTORY}/{TRAINED_MOD_DIR}"):
    os.makedirs(f"{OUTPUT_DIRECTORY}/{TRAINED_MOD_DIR}")

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

    filename = f"sampled_fps_fns_fp_mean_{fp_rate_mean}_fn_mean_{fn_rate_mean}_noise_type_{noise_type}_x_{x_noisy_samples}_split_{split_id}.txt"
   # f"sampled_fps_fns_fp_mean_{fp_rate_mean}_fn_mean_{fn_rate_mean}_sigma_{sigma_fp}_x_{x_noisy_samples}_split_{split_id}.txt"

    # augment training set
    X_augmented, y_augmented = augment_data_with_noise(X_val_train, y_label_train, n_clones=x_noisy_samples, mean_fp=fp_rate_mean, mean_fn=fn_rate_mean, noise_type=noise_type, filename=filename)
    


    model = make_xgb()
    pipe_with_noise = make_pipeline(model)
    pipe_with_noise.fit(X_augmented, y_augmented)

    return split_id, pipe_with_noise


# 1. Read the data
from collections import defaultdict

all_splits_dict = defaultdict(int)
for split_id in tqdm(range(NUM_SPLITS_TO_READ), desc="Processing splits..."):
    # Read train data
    data_filename_train = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/train_data_{TAX_LEVEL}_split_{split_id}"
    y_filename_train = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/train_annot_{TAX_LEVEL}_split_{split_id}"
    taxa_names_filename_train = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/train_taxa_{TAX_LEVEL}_split_{split_id}" if TAX_LEVEL != "random" else None
    X_val_train, y_label_train, X_column_names, taxa_group_names_train = read_diderm_data(data_filename_train, y_filename_train, taxa_names_filename_train, DEVICE)

    # Read validation data
    data_filename_val = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/val_data_{TAX_LEVEL}_split_{split_id}"
    y_filename_val = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/val_annot_{TAX_LEVEL}_split_{split_id}"
    taxa_names_filename_val = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/val_taxa_{TAX_LEVEL}_split_{split_id}" if TAX_LEVEL != "random" else None
    X_val_val, y_label_val, X_column_names, taxa_group_names_val = read_diderm_data(data_filename_val, y_filename_val, taxa_names_filename_val, DEVICE)  
    
    # Read test data
    data_filename_test = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/test_data_{TAX_LEVEL}_split_{split_id}"
    y_filename_test = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/test_annot_{TAX_LEVEL}_split_{split_id}"
    taxa_names_filename_test = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/test_taxa_{TAX_LEVEL}_split_{split_id}" if TAX_LEVEL != "random" else None
    X_val_test, y_label_test, X_column_names, taxa_group_names_test = read_diderm_data(data_filename_test, y_filename_test, taxa_names_filename_test, DEVICE)

    if sum(y_label_test) == 0 or sum(y_label_train) == 0:
        continue

    if TAX_LEVEL == "random":
        taxa_group_names_train = None
        taxa_group_names_test = None

    if X_val_train is not None and y_label_train is not None and X_val_test is not None and y_label_test is not None and X_val_val is not None and y_label_val is not None:
        curr_split_dict = defaultdict(str)
        curr_split_dict["X_train"] = X_val_train
        curr_split_dict["y_train"] = y_label_train
        curr_split_dict["taxa_group_names_train"] = taxa_group_names_train
        curr_split_dict["X_test"] = X_val_test
        curr_split_dict["y_test"] = y_label_test
        curr_split_dict["taxa_group_names_test"] = taxa_group_names_test
        curr_split_dict["X_val"] = X_val_val
        curr_split_dict["y_val"] = y_label_val
        curr_split_dict["taxa_group_names_val"] = taxa_group_names_val
        curr_split_dict["feature_names"] = X_column_names

        all_splits_dict[split_id] = curr_split_dict



# --- main loop ---
trained_models_all_noise_rates = {}

for fp_rate_mean, fn_rate_mean in tqdm(mean_rem_add_rates_tuples, desc="Noise configs..."):
    noise_rates = (fp_rate_mean, fn_rate_mean)
    print(f"noise_rates = {noise_rates}")

    # Create the txt files for 
    for split_id in all_splits_dict.keys():
        filename = f"sampled_fps_fns_fp_mean_{fp_rate_mean}_fn_mean_{fn_rate_mean}_noise_type_{noise_type}_x_{x_noisy_samples}_split_{split_id}.txt"
        open(f"{OUTPUT_DIRECTORY}/{TRAINED_MOD_DIR}/{filename}", "w").close()

    print(f"filename = {filename}")

    # parallel over splits
    results = Parallel(n_jobs=min(4, os.cpu_count()), backend="threading")(
        delayed(train_one_split)(
            split_id, fp_rate_mean, fn_rate_mean, x_noisy_samples
        )
        for split_id in all_splits_dict.keys()
    )

    # collect models
    trained_models_local = {split_id: model for split_id, model in results}
    trained_models_all_noise_rates[noise_rates] = trained_models_local

    # save per noise config
    filename = f"trained_models_fp_{fp_rate_mean}_fn_{fn_rate_mean}_noise_type_{noise_type}_x_{x_noisy_samples}.pkl"
    joblib.dump(trained_models_local, f"{OUTPUT_DIRECTORY}/{TRAINED_MOD_DIR}/{filename}")

    print(f"✅ Saved {len(trained_models_local)} models for noise {noise_rates}")


