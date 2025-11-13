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
from utils.utils import read_ogt_data, pca_run_and_plot, tsne_plot

TAX_LEVEL = "phylum" # <--------------- taxonomy level for train/test split is here!!
DATA_DIRECTORY = "data_ogt" # <-------- input data directory is here!!
FEATURES = "COG_aa" # <------------------- data features is here!!
NUM_SPLITS_TO_READ = 3  # <----------- number of splits to read and process is here!!!

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIRECTORY = f"../data_ogt/outputs/{FEATURES}/{TAX_LEVEL}"

TRAINED_MOD_DIR = "trained_models_train_val_test"

# Params for the noisy training

parser = argparse.ArgumentParser(description="Noise sampling experiment")
parser.add_argument("--x_noisy_samples", type=int, default=10,
                    help="Number of noisy samples per clean sample")
parser.add_argument("--noise_type", type=str, default=None,
                    help="Noise distribution")

# --- use args ---
args = parser.parse_args()
x_noisy_samples = args.x_noisy_samples
noise_type = args.noise_type

noise_std = 0.3

print(f"Calculating for x_noisy_samples = {x_noisy_samples}; noise_type = {noise_type}")

mn_fp_arr =  [0.0, 0.05, 0.1, 0.15, 0.2, 0.5]
mn_fn_arr = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0] 


mean_rem_add_rates_tuples = [(add, rem) for add in mn_fp_arr for rem in mn_fn_arr]


def apply_noise(genome, fp_rate=0.2, fn_rate=0.5):
    genome_noisy = genome.float().clone()

    # False negatives (multiple hits per count, Poisson distributed)
    losses = torch.poisson(genome.float() * fn_rate)
    genome_noisy = torch.clamp(genome_noisy - losses.int(), min=0)

    # False positives (Poisson noise added to zeros only)
  #  fp_add = torch.zeros_like(genome)
    fp_add = torch.zeros_like(genome, dtype=torch.float)
    zero_mask = genome > -1
   # fp_add = torch.poisson(torch.full((zero_mask.sum(),), fp_rate)).to(genome_noisy.dtype)
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

    
def label_ogt_range(y,high_thresh=45):
    labels = []
    for val in y:
        if val < high_thresh:
            labels.append('low')
        else:
            labels.append('high')
    return np.array(labels)



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import KFold, GroupKFold
from xgboost import XGBClassifier, XGBRegressor
import numpy as np

from sklearn.preprocessing import LabelEncoder


from sklearn.utils.class_weight import compute_class_weight
import subprocess

def is_gpu_available():
    try:
        subprocess.check_output(["nvidia-smi"])
        return True
    except Exception:
        return False

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


def make_xgb_regressor(tree_method="auto"):
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        min_child_weight=5,
        tree_method=tree_method,
        n_jobs=-1,
        objective="reg:squarederror",  # standard regression
        eval_metric="rmse"              # optional
    )  

def train_one_split(split_id, fp_rate_mean, fn_rate_mean, x_noisy_samples):
    num_classes = 2
    temp_bound = 45

    X_val_train = all_splits_dict[split_id]["X_train"]
    y_label_train = all_splits_dict[split_id]["y_train"]

    filename = f"sampled_fps_fns_fp_mean_{fp_rate_mean}_fn_mean_{fn_rate_mean}_noise_type_{noise_type}_x_{x_noisy_samples}_split_{split_id}.txt"


    # augment training set
    X_augmented, y_augmented = augment_data_with_noise(X_val_train, y_label_train, n_clones=x_noisy_samples, mean_fp=fp_rate_mean, mean_fn=fn_rate_mean, noise_type=noise_type, filename=filename)

    range_labels = label_ogt_range(y_augmented)
    le = LabelEncoder()
    range_ids = le.fit_transform(range_labels)  # Converts to 0,1,2
    label_to_int = {'low': 0, 'high': 1}
    range_ids = np.vectorize(label_to_int.get)(range_labels)

    classes = np.unique(range_ids)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=range_ids)
    class_weights = dict(zip(classes, weights))
    sample_weights = np.array([class_weights[c] for c in range_ids])


    # Device-aware XGBoost config
    if is_gpu_available():
        tree_method = "hist"
        device = "cuda"
       # print("GPU detected: using CUDA device.")
    else:
        tree_method = "hist"
        device = "cpu"
      #  print("No GPU: using CPU.")


    # Convert to numpy
    X_train_np = X_augmented.cpu().numpy()
    y_train_np = y_augmented.cpu().numpy().flatten()
    
    sample_weights_np = sample_weights if isinstance(sample_weights, np.ndarray) else sample_weights.cpu().numpy()
    range_ids_np = range_ids if isinstance(range_ids, np.ndarray) else range_ids.cpu().numpy()

    # 1. Gating model
    gating_model_with_noise = make_xgb()
    gating_model_with_noise.fit(X_augmented, range_ids, sample_weight=sample_weights)

    # Right and left regressors
    low_mask = y_train_np < temp_bound
    high_mask = y_train_np >= temp_bound


    model_low_with_noise = make_xgb_regressor()
    model_high_with_noise = make_xgb_regressor()
    model_low_with_noise.fit(X_train_np[low_mask], y_train_np[low_mask])
    model_high_with_noise.fit(X_train_np[high_mask], y_train_np[high_mask])
    
    return split_id, gating_model_with_noise, model_low_with_noise, model_high_with_noise



from collections import defaultdict

all_splits_dict = defaultdict(int)
for split_id in range(NUM_SPLITS_TO_READ):
    # Read train data
    data_filename_train = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/train_data_{TAX_LEVEL}_split_{split_id}"
    y_filename_train = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/train_annot_{TAX_LEVEL}_split_{split_id}"
    taxa_names_filename_train = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/train_taxa_{TAX_LEVEL}_split_{split_id}" if TAX_LEVEL != "random" else None
    X_test, y_test, X_column_names, taxa_group_names_test = read_ogt_data(data_filename_train, y_filename_train, taxa_names_filename_train, DEVICE)
    
    # Read validation data
    data_filename_val = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/val_data_{TAX_LEVEL}_split_{split_id}"
    y_filename_val = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/val_annot_{TAX_LEVEL}_split_{split_id}"
    taxa_names_filename_val = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/val_taxa_{TAX_LEVEL}_split_{split_id}" if TAX_LEVEL != "random" else None
    X_val, y_val, X_column_names, taxa_group_names_val = read_ogt_data(data_filename_val, y_filename_val, taxa_names_filename_val, DEVICE)  
    
    # Read test data
    data_filename_test = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/test_data_{TAX_LEVEL}_split_{split_id}"
    y_filename_test = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/test_annot_{TAX_LEVEL}_split_{split_id}"
    taxa_names_filename_test = f"../{DATA_DIRECTORY}/input_data_train_val_test/{TAX_LEVEL}/test_taxa_{TAX_LEVEL}_split_{split_id}" if TAX_LEVEL != "random" else None
    X_train, y_train, X_column_names, taxa_group_names_train = read_ogt_data(data_filename_test, y_filename_test, taxa_names_filename_test, DEVICE)


    
    if sum(y_train)/len(y_train) < 0.01 or  sum(y_test)/len(y_test) < 0.01:
        print(f"Skipping split_id = {split_id}")
        continue
    if TAX_LEVEL == "random":
        taxa_group_names_train = None
        taxa_group_names_test = None
        
    if X_train is not None and y_train is not None and X_test is not None and y_test is not None and X_val is not None and y_val is not None:    
        X_test = X_test[:, :-20]
        X_val = X_val[:, :-20]
        X_train = X_train[:, :-20]

        X_column_names = X_column_names[:-20]
        curr_split_dict = defaultdict(str)
        curr_split_dict["X_train"] = X_train
        curr_split_dict["y_train"] = y_train
        curr_split_dict["taxa_group_names_train"] = taxa_group_names_train
        curr_split_dict["X_test"] = X_test
        curr_split_dict["y_test"] = y_test
        curr_split_dict["taxa_group_names_test"] = taxa_group_names_test
        curr_split_dict["X_val"] = X_val
        curr_split_dict["y_val"] = y_val
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
        open(f"{OUTPUT_DIRECTORY}/trained_models/{filename}", "w").close()

    print(f"filename = {filename}")

    # parallel over splits
    results = Parallel(n_jobs=min(4, os.cpu_count()), backend="threading")(
        delayed(train_one_split)(
            split_id, fp_rate_mean, fn_rate_mean, x_noisy_samples
        )
        for split_id in all_splits_dict.keys()
    )

    # collect models
    trained_models_local = {split_id: (pipe_with_noise, model_low_with_noise, model_high_with_noise) for split_id, pipe_with_noise, model_low_with_noise, model_high_with_noise in results}
    trained_models_all_noise_rates[noise_rates] = trained_models_local

    # save per noise config
    filename = f"trained_models_fp_{fp_rate_mean}_fn_{fn_rate_mean}_noise_type_{noise_type}_x_{x_noisy_samples}.pkl"
    joblib.dump(trained_models_local, f"{OUTPUT_DIRECTORY}/{TRAINED_MOD_DIR}/{filename}")

    print(f"✅ Saved {len(trained_models_local)} models for noise {noise_rates}")






















    
    