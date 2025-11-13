import pandas as pd
import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.colors import ListedColormap

from utils.utils import read_aerob_data, pca_run_and_plot, tsne_plot


print("Starting the script...")

TAX_LEVEL = "phylum"  # <----------------- taxonomy level for train/test split is here!!
DATA_DIRECTORY = "data_aerob"  # <-------- input data directory is here!!
NUM_SPLITS_TO_READ = 30  # <-------------- number of splits to read and process is here!!

RANDOM_SEED = 42
OUTPUT_DIRECTORY = f"../{DATA_DIRECTORY}/outputs/{TAX_LEVEL}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FONTSIZE = 13

FEATURE_STEP = 100

all_splits_dict = defaultdict(int)
for split_id in range(NUM_SPLITS_TO_READ):
    # Read train data
    data_filename_train = f"../{DATA_DIRECTORY}/input_data/{TAX_LEVEL}/train_data_{TAX_LEVEL}_tax_level_split_{split_id}"
    y_filename_train = f"../{DATA_DIRECTORY}/input_data/{TAX_LEVEL}/train_annot_{TAX_LEVEL}_tax_level_split_{split_id}"
    taxa_names_filename_train = f"../{DATA_DIRECTORY}/input_data/{TAX_LEVEL}/train_taxa_names_{TAX_LEVEL}_tax_level_split_{split_id}" if TAX_LEVEL != "random" else None
    X_val_train, y_label_train, X_column_names, taxa_group_names_train = read_aerob_data(data_filename_train, y_filename_train, taxa_names_filename_train, DEVICE)  
   # X_val_train = (X_val_train > 0).int()
    
    # Read test data
    data_filename_test = f"../{DATA_DIRECTORY}/input_data/{TAX_LEVEL}/test_data_{TAX_LEVEL}_tax_level_split_{split_id}"
    y_filename_test = f"../{DATA_DIRECTORY}/input_data/{TAX_LEVEL}/test_annot_{TAX_LEVEL}_tax_level_split_{split_id}"
    taxa_names_filename_test = f"../{DATA_DIRECTORY}/input_data/{TAX_LEVEL}/test_taxa_names_{TAX_LEVEL}_tax_level_split_{split_id}"  if TAX_LEVEL != "random" else None
    X_val_test, y_label_test, X_column_names, taxa_group_names_test = read_aerob_data(data_filename_test, y_filename_test, taxa_names_filename_test, DEVICE)  
  #  X_val_test = (X_val_test > 0).int()

    if sum(y_label_train)/len(y_label_train) < 0.01 or  sum(y_label_test)/len(y_label_test) < 0.01:
        print(f"Skipping split_id = {split_id}")
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
    
print(f"Number of added splits = {len(all_splits_dict.keys())}")    

# Concatenate train and test for the 2D visualization
y_label = torch.cat((y_label_train, y_label_test), dim=0)
X_val = torch.cat((X_val_train, X_val_test), dim=0)
if TAX_LEVEL != "random":
    taxa_group_names = taxa_group_names_train + taxa_group_names_test
else:    
    taxa_group_names = None

# READ/FIND SHAP values
from feature_selection.feature_selection_utils import iamb
from feature_selection.feature_selection_utils import shap_features
import warnings
from sklearn.preprocessing import KBinsDiscretizer
import json

filename = f"{OUTPUT_DIRECTORY}/shap_list.json"
all_shap_lists_dict = defaultdict(int)
device = 'cpu'


if os.path.isfile(filename):
    print("Reading the existing file...")
    with open(filename, "rb") as f:
        all_shap_lists_dict = json.load(f)

else:
    for split_id in range(NUM_SPLITS_TO_READ):
        print(f"Processing split {split_id}")
        X_val_train =  all_splits_dict[split_id]["X_train"]
        y_label_train =  all_splits_dict[split_id]["y_train"]
        X_column_names =  all_splits_dict[split_id]["feature_names"]
 
        X_np = (X_val_train > 0).int().cpu()

        sorted_cog_idx_by_shap, sorted_shap, sorted_names, shap_vals  = shap_features(X_np, y_label_train.cpu(), X_column_names, device) 
        print("SHAP list:", len(sorted_names))
        all_shap_lists_dict[split_id] = sorted_names[:]#[:N]
    with open(filename, "w") as f:
        json.dump(dict(all_shap_lists_dict), f, indent=2)


import os
import pickle
from collections import defaultdict
from feature_selection.feature_selection_utils import xgboost_train_accur, xgboost_accur_select_features

train_test_feat_apply_flag = True
filename = f"{OUTPUT_DIRECTORY}/accuracy_curves_all_splits_add_feat_train_test.pkl"

# Load if file exists
if False:#os.path.exists(filename):
    with open(filename, "rb") as f:
        accuracy_curves_all_splits_add_feat_train_test = pickle.load(f)
    print(f"Loaded existing results from {filename}")
else:
    accuracy_curves_all_splits_add_feat_train_test = defaultdict(dict)
    print("No existing file found. Starting fresh...")

    feat_step = FEATURE_STEP
    feat_removal = False
    accuracy_curves_all_splits_add_feat_train_test = defaultdict(dict) 
    for split_id in all_shap_lists_dict.keys():

        X_val_train =  all_splits_dict[int(split_id)]["X_train"]
        y_label_train =  all_splits_dict[int(split_id)]["y_train"]
        X_column_names =  all_splits_dict[int(split_id)]["feature_names"]
        taxa_group_names_train = all_splits_dict[int(split_id)]["taxa_group_names_train"]
        
        shap_list = all_shap_lists_dict[split_id]
        X_column_names = list(X_column_names)
        indices = [X_column_names.index(f) for f in shap_list if f in X_column_names]
        cv_accur_arr, test_accur_arr, num_feat = xgboost_accur_select_features(X_val_train.cpu(), X_val_test.cpu(), y_label_train.cpu(), y_label_test.cpu(), indices, feat_step, DEVICE, feat_removal,  train_test_feat_apply_flag, groups = taxa_group_names_train)    
        accuracy_curves_all_splits_add_feat_train_test[split_id]['cv_accur'] = cv_accur_arr
        accuracy_curves_all_splits_add_feat_train_test[split_id]['test_accur'] = test_accur_arr
        accuracy_curves_all_splits_add_feat_train_test[split_id]['num_feat'] = num_feat
        print(f"Split {split_id} done")
    
    with open(filename, "wb") as f:
        pickle.dump(accuracy_curves_all_splits_add_feat_train_test, f)



train_test_feat_apply_flag = False
filename = f"{OUTPUT_DIRECTORY}/accuracy_curves_all_splits_add_feat_test.pkl"

# Load if file exists
if False:#os.path.exists(filename):
    with open(filename, "rb") as f:
        accuracy_curves_all_splits_add_feat_test = pickle.load(f)
    print(f"Loaded existing results from {filename}")
else:
    accuracy_curves_all_splits_add_feat_test = defaultdict(dict)
    print("No existing file found. Starting fresh...")

    feat_step = FEATURE_STEP
    feat_removal = False
    accuracy_curves_all_splits_add_feat_test = defaultdict(dict) 
    for split_id in all_shap_lists_dict.keys():

        X_val_train =  all_splits_dict[int(split_id)]["X_train"]
        y_label_train =  all_splits_dict[int(split_id)]["y_train"]
        X_column_names =  all_splits_dict[int(split_id)]["feature_names"]
        taxa_group_names_train = all_splits_dict[int(split_id)]["taxa_group_names_train"]
        
        shap_list = all_shap_lists_dict[split_id]
        X_column_names = list(X_column_names)
        indices = [X_column_names.index(f) for f in shap_list if f in X_column_names]
        cv_accur_arr, test_accur_arr, num_feat = xgboost_accur_select_features(X_val_train.cpu(), X_val_test.cpu(), y_label_train.cpu(), y_label_test.cpu(), indices, feat_step, DEVICE, feat_removal, train_test_feat_apply_flag, groups = taxa_group_names_train)    
        accuracy_curves_all_splits_add_feat_test[split_id]['cv_accur'] = cv_accur_arr
        accuracy_curves_all_splits_add_feat_test[split_id]['test_accur'] = test_accur_arr
        accuracy_curves_all_splits_add_feat_test[split_id]['num_feat'] = num_feat
        print(f"Split {split_id} done")
    
    with open(filename, "wb") as f:
        pickle.dump(accuracy_curves_all_splits_add_feat_test, f)

train_test_feat_apply_flag = True
filename = f"{OUTPUT_DIRECTORY}/accuracy_curves_all_splits_rem_feat_train_test.pkl"

# Load if file exists
if False:#os.path.exists(filename):
    with open(filename, "rb") as f:
        accuracy_curves_all_splits_rem_feat_train_test = pickle.load(f)
    print(f"Loaded existing results from {filename}")
else:
    print("No existing file found. Starting fresh...")
    accuracy_curves_all_splits_rem_feat_train_test = defaultdict(dict) 

    feat_removal = True
    feat_step = FEATURE_STEP
    for split_id in all_shap_lists_dict.keys():

        X_val_train =  all_splits_dict[int(split_id)]["X_train"]
        y_label_train =  all_splits_dict[int(split_id)]["y_train"]
        X_column_names =  all_splits_dict[int(split_id)]["feature_names"]
        taxa_group_names_train = all_splits_dict[int(split_id)]["taxa_group_names_train"]
        
        shap_list = all_shap_lists_dict[split_id]
        X_column_names = list(X_column_names)
        indices = [X_column_names.index(f) for f in shap_list if f in X_column_names]
        cv_accur_arr, test_accur_arr, num_feat = xgboost_accur_select_features(X_val_train.cpu(), X_val_test.cpu(), y_label_train.cpu(), y_label_test.cpu(), indices, feat_step, DEVICE, feat_removal, train_test_feat_apply_flag, groups = taxa_group_names_train)    
        accuracy_curves_all_splits_rem_feat_train_test[split_id]['cv_accur'] = cv_accur_arr
        accuracy_curves_all_splits_rem_feat_train_test[split_id]['test_accur'] = test_accur_arr
        accuracy_curves_all_splits_rem_feat_train_test[split_id]['num_feat'] = num_feat
        print(f"Split {split_id} done")
    with open(filename, "wb") as f:
        pickle.dump(accuracy_curves_all_splits_rem_feat_train_test, f)        

train_test_feat_apply_flag = False
filename = f"{OUTPUT_DIRECTORY}/accuracy_curves_all_splits_rem_feat_test.pkl"

# Load if file exists
if False:# os.path.exists(filename):
    with open(filename, "rb") as f:
        accuracy_curves_all_splits_rem_feat_test = pickle.load(f)
    print(f"Loaded existing results from {filename}")
else:
    print("No existing file found. Starting fresh...")
    accuracy_curves_all_splits_rem_feat_test = defaultdict(dict) 

    feat_removal = True
    feat_step = FEATURE_STEP
    for split_id in all_shap_lists_dict.keys():

        X_val_train =  all_splits_dict[int(split_id)]["X_train"]
        y_label_train =  all_splits_dict[int(split_id)]["y_train"]
        X_column_names =  all_splits_dict[int(split_id)]["feature_names"]
        taxa_group_names_train = all_splits_dict[int(split_id)]["taxa_group_names_train"]
        
        shap_list = all_shap_lists_dict[split_id]
        X_column_names = list(X_column_names)
        indices = [X_column_names.index(f) for f in shap_list if f in X_column_names]
        cv_accur_arr, test_accur_arr, num_feat = xgboost_accur_select_features(X_val_train.cpu(), X_val_test.cpu(), y_label_train.cpu(), y_label_test.cpu(), indices, feat_step, DEVICE, feat_removal, feat_removal, groups = taxa_group_names_train)    
        accuracy_curves_all_splits_rem_feat_test[split_id]['cv_accur'] = cv_accur_arr
        accuracy_curves_all_splits_rem_feat_test[split_id]['test_accur'] = test_accur_arr
        accuracy_curves_all_splits_rem_feat_test[split_id]['num_feat'] = num_feat
        print(f"Split {split_id} done")
    with open(filename, "wb") as f:
        pickle.dump(accuracy_curves_all_splits_rem_feat_test, f)            