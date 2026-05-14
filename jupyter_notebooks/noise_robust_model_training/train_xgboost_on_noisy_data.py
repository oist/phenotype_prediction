import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier, XGBRegressor

import joblib
from joblib import Parallel, delayed

# Define constant params used below
TAX_LEVEL = "phylum" # <----------------- taxonomy level for train/test split is here!!
DATA_DIRECTORY = "data_diderm"  # <-------- input data directory is here!!
NUM_SPLITS_TO_READ = 5#30  # <-------------- number of splits to read and process is here!!

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIRECTORY = f"outputs/{TAX_LEVEL}"
TRAINED_MOD_DIR = "trained_models_train_val_test"
os.makedirs(f"{OUTPUT_DIRECTORY}/{TRAINED_MOD_DIR}", exist_ok=True) # Make sure the output dir exists

# GPU / CPU setup 
use_gpu = torch.cuda.is_available()
tree_method = "gpu_hist" if use_gpu else "hist"
predictor = "gpu_predictor" if use_gpu else "auto"

# Define input variable params for the noisy training
parser = argparse.ArgumentParser(description="Noise sampling experiment")
parser.add_argument("--x_noisy_samples", type=int, default=10,
                    help="Number of noisy samples per clean sample")
parser.add_argument("--noise_type", type=str, default=None,
                    help="Noise distribution")
args = parser.parse_args()

# Read the input params
x_noisy_samples = args.x_noisy_samples
noise_type = args.noise_type

# Define mean values for the FP and FN distributions
mn_fp_arr =  [0.0, 0.05, 0.1, 0.15, 0.2, 0.5]
mn_fn_arr = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0] 
# Create a list of the FP and FN mean tuples to define their sampling distributions
mean_rem_add_rates_tuples = [(add, rem) for add in mn_fp_arr for rem in mn_fn_arr]


def sample_exp(mean):
    """
    Samples a value from the exponential distribution with the specified mean value 
    """
    sample = np.random.exponential(scale=mean) 
    return sample

def sample_unif(mean):
    """
    Samples a value from the uniform distribution with the specified mean value 
    """
    sample = np.random.uniform(0, 2*mean)
    return sample

def sample_gamma(mean):
    """
    Samples a value from the gamma distribution with the specified mean value (the sacle value is fixed and = 0.5)
    """
    scale = 0.5
    shape = mean/scale
    sample = np.random.gamma(shape, scale)
    return sample

def apply_noise(genome, fp_rate=0.2, fn_rate=0.5):
    """
    Applies noise at the specified FP anf FN rates to a genome count vector. The noise is implemented in two steps:
        1. False negatives implemented as the poisson distributed counts subtracted from the original non-zero counts 
        2. False positives are implemented as the poisson distributed counts added to the results of step (1)

    Args:
        - genome (tensor): genome count vector 
        - fp_rate (float): FP rate of the Poisson distribution for the false positive count sampling, default = 0.2
        - fn_rate (float): FN rate of the Poisson distribution for the false negative count sampling, default = 0.5
    Returns:
        - genome_noisy (tensor): noisy genome count vector 
    """
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

def augment_data_with_noise(X_val_train, y_label_train, n_clones, mean_fp, mean_fn, noise_type=None):
    """
    Augments the train data set with noisy genome clones. The noise scheme for teh train data augmentation is specified by the argumnets of the function.
    The noisy genomes are annotated in the same way as the original genome. 

    Args:
        - X_val_train (tensor): X train (count table)
        - y_label_train (tensor): y train (annotation vector)
        - n_clones (int): a number of genome copies to create for each original genome 
        - mean_fp (float): mean value of the FP rate distribution 
        - mean_fn (float): mean value of the FN rate distribution 
        - noise_type (str): a name of the distribution from which FPs and FNs are sampled ["exp"|"gamma":"unif"]
    Returns:
        - X_augmented (tensor): augmented X train (count table)
        - y_augmented (tensor): augmented y train (annotation vector)
    """
    # Start with the original data as lists
    X_augmented_list = [X_val_train]
    y_augmented_list = [y_label_train]

    if n_clones == 0:
        return X_val_train.clone(), y_label_train.clone()

    n_rows, _ = X_val_train.shape
    # Iterate over all original genomes and add n_clones number of noisy copies
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
                print(f"Incorrect noise type! Please choose either 'exp', or 'gamma', or 'unif'.")

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

def train_one_split(split_id, fp_rate_mean, fn_rate_mean, x_noisy_samples):
    """
    Train a model on a noise augmented data split.
    """
    X_val_train = all_splits_dict[split_id]["X_train"]
    y_label_train = all_splits_dict[split_id]["y_train"]

    filename = f"sampled_fps_fns_fp_mean_{fp_rate_mean}_fn_mean_{fn_rate_mean}_noise_type_{noise_type}_x_{x_noisy_samples}_split_{split_id}.txt"

    # Augment training set
    X_augmented, y_augmented = augment_data_with_noise(X_val_train, y_label_train, n_clones=x_noisy_samples, mean_fp=fp_rate_mean, mean_fn=fn_rate_mean, noise_type=noise_type)

    # Train a model
    model = make_xgb()
    pipe_with_noise = make_pipeline(model)
    pipe_with_noise.fit(X_augmented, y_augmented)

    return split_id, pipe_with_noise


# 1. Read the data
all_splits_dict = defaultdict(int)
for split_id in tqdm(range(NUM_SPLITS_TO_READ), desc="Processing splits..."):
    # Read train data
    data_filename_train = f"../../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/train_data_{TAX_LEVEL}_split_{split_id}"
    y_filename_train = f"../../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/train_annot_{TAX_LEVEL}_split_{split_id}"
    taxa_names_filename_train = f"../../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/train_taxa_{TAX_LEVEL}_split_{split_id}" if TAX_LEVEL != "random" else None
    X_val_train = torch.load(data_filename_train)
    y_label_train = torch.load(y_filename_train)
    with open(taxa_names_filename_train, "r") as f:
        taxa_group_names_train = [line.strip() for line in f]

    # Read validation data
    data_filename_val = f"../../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/val_data_{TAX_LEVEL}_split_{split_id}"
    y_filename_val = f"../../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/val_annot_{TAX_LEVEL}_split_{split_id}"
    taxa_names_filename_val = f"../../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/val_taxa_{TAX_LEVEL}_split_{split_id}" if TAX_LEVEL != "random" else None
    X_val_val = torch.load(data_filename_val)
    y_label_val = torch.load(y_filename_val)
    with open(taxa_names_filename_val, "r") as f:
        taxa_group_names_val = [line.strip() for line in f]

    # Read test data
    data_filename_test = f"../../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/test_data_{TAX_LEVEL}_split_{split_id}"
    y_filename_test = f"../../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/test_annot_{TAX_LEVEL}_split_{split_id}"
    taxa_names_filename_test = f"../../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/test_taxa_{TAX_LEVEL}_split_{split_id}" if TAX_LEVEL != "random" else None
    X_val_test = torch.load(data_filename_train)
    y_label_test = torch.load(y_filename_train)
    with open(taxa_names_filename_test, "r") as f:
        taxa_group_names_test = [line.strip() for line in f]

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
        all_splits_dict[split_id] = curr_split_dict


# 2. Main loop: for each combination of FP and FN rates, fixed x (number of genome noisy copies), and fixed noise_type (FP, FN rate distributions), train XGBoost models on the augmented train datasets, and save the results

# Print some logs
print(f"Processing {NUM_SPLITS_TO_READ} number of data splits for training...")
print(f"Training XGBoost models for the data with x = {x_noisy_samples} number of clones, and the FPs and FNs sampled from {noise_type} distribution...")

# Doct for the trained models
trained_models_all_noise_rates = {}
for fp_rate_mean, fn_rate_mean in tqdm(mean_rem_add_rates_tuples, desc="Noise configs progress..."):
    noise_rates = (fp_rate_mean, fn_rate_mean)
    print(f"noise_rates = {noise_rates}")

    # Optional: save sampled FNs and FPs as a txt (just for a sanity check)
    for split_id in all_splits_dict.keys():
        filename = f"sampled_fps_fns_fp_mean_{fp_rate_mean}_fn_mean_{fn_rate_mean}_noise_type_{noise_type}_x_{x_noisy_samples}_split_{split_id}.txt"
        open(f"{OUTPUT_DIRECTORY}/{TRAINED_MOD_DIR}/{filename}", "w").close()

    # Parallel model training over splits
    results = Parallel(n_jobs=min(4, os.cpu_count()), backend="threading")(
        delayed(train_one_split)(
            split_id, fp_rate_mean, fn_rate_mean, x_noisy_samples
        )
        for split_id in all_splits_dict.keys()
    )

    # Collect models
    trained_models_local = {split_id: model for split_id, model in results}
    trained_models_all_noise_rates[noise_rates] = trained_models_local

    # Save per noise config
    filename = f"trained_models_fp_{fp_rate_mean}_fn_{fn_rate_mean}_noise_type_{noise_type}_x_{x_noisy_samples}.pkl"
    joblib.dump(trained_models_local, f"{OUTPUT_DIRECTORY}/{TRAINED_MOD_DIR}/{filename}")

    print(f"Saved {len(trained_models_local)} models for noise {noise_rates}!")


