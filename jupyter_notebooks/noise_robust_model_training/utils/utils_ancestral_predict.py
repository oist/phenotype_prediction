import os
import sys
import warnings
import contextlib
from collections import defaultdict

import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
import xgboost

from tqdm import tqdm, TqdmExperimentalWarning
from joblib import Parallel, delayed

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from xgboost import XGBClassifier, XGBRegressor


# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")


@contextlib.contextmanager
def suppress_xgb_warnings():
    with open(os.devnull, "w") as fnull:
        old_stderr = sys.stderr
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

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

def sample_gamma(mean, scale = 0.5):
    """
    Samples a value from the gamma distribution with the specified mean value (the sacle value is fixed and = 0.5)
    """
    shape = mean/scale
    sample = np.random.gamma(shape, scale)
    return sample   


def fp_curve_areas_one_model(metric, max_fp, max_fn, cog_remov_add_accuracies):
    """
    Calculates the integrals of the FN curves for each FP value. 
    """
    fn_space = []
    fp_curves = defaultdict(list)
    for (fn, fp) in cog_remov_add_accuracies.keys():   
        if fp <= max_fp:
            if fn <= max_fn:
                fp_curves[fp].append(cog_remov_add_accuracies[(fn, fp)][metric][0])
                if fn not in fn_space:
                    fn_space.append(fn)
    fp_curve_areas = []
    for fp in fp_curves.keys():
        fp_curve_areas.append(np.trapezoid(fp_curves[fp], fn_space))
    return  fp_curve_areas       


def augment_data_with_noise(X_val_train, y_label_train, n_clones, mean_fp, mean_fn, noise_type=None, filename=None):
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

def apply_noise(genome, fp_rate, fn_rate):
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
    fp_add = torch.zeros_like(genome).float()
    zero_mask = genome > -1
    fp_add[zero_mask] = torch.poisson(torch.full((zero_mask.sum(),), fp_rate))
    genome_noisy = genome_noisy + fp_add

    return genome_noisy

def flip_with_fractional_noise(X: torch.Tensor, fp_rate: float, fn_rate: float,
                               noise_std = 0.3, hard_fn_flag = False):
    """
    Creates a noisy X dataset by flipping the true counts at false positive and false negative rates.
    
    Args:
        - X (tensor): original count table,
        - fn_rate (float): FN rate, i.e., the fraction of non-zero counts that are either reduced by 1 or set to 0 (specified by hard_fn_flag).
        - fp_rate (float): FP rate, i.e. the fraction of all counts that are incteased by 1 
        - hard_fn_flag (float): the flag specifying if a count should be reduced by 1 or set to 0
    Returns:
        - X_noisy (tensor): noisy count table.
    """
    
    X_noisy = X.float().clone()
    n_rows, _ = X_noisy.shape
    for i in range(n_rows):
        # False negatives: subtract (1 + noise) from fraction of positives
        pos_idx = torch.nonzero(X[i] > 0).flatten()
        n_fn = int(round(fn_rate * len(pos_idx)))

        if n_fn > 0:
            fn_idx = pos_idx[torch.randperm(len(pos_idx))[:n_fn]]
            noise = torch.randn(len(fn_idx)) * noise_std
            if hard_fn_flag == False:
                X_noisy[i, fn_idx] -= 1#(1.0 + noise)
            else:
                X_noisy[i, fn_idx] = 0 

        # False positives: add (1 + noise) to fraction of zeros
        zero_idx = torch.nonzero(X[i] > -1).flatten()
        n_fp = int(round(fp_rate * len(zero_idx)))
        if n_fp > 0:
            fp_idx = zero_idx[torch.randperm(len(zero_idx))[:n_fp]]
            noise = torch.randn(len(fp_idx)) * noise_std
            X_noisy[i, fp_idx] += 1 #(1.0 + noise)

    # Clamp to ensure no negatives
    X_noisy = torch.clamp(X_noisy, min=0.0)    

    return X_noisy

def eval_trained_models_on_noisy_data(
    all_splits_dict,
    trained_models,
    hard_fn_flag=False,
    max_fp=0.2,
    max_fn=1.0,
    n_jobs=-1,
    truncated_feature_set = None,
    test_or_val = "test"
):
    """
    Evaluates the performance of the trained models on the test data with different noise levels.
    The noise is added to the train data by flipping a fraction of non-zero counts to zeros, and adding one count to some fraction of all counts. 
    Those fractions are increased gradually in the test data.

    Args:
        - all_splits_dict (dict): a dictionary with all data splits
        - trained_models (dict): a dictionary with the trained models on all_splits_dict data splits (the keys of these dicts = split IDs and are consistent between each other)
        - hard_fn_flag (bool): a boolian flag for whether the non-zero counts should be flipped to 0 (True), or -1 (False), default = False
        - max_fp (float): max fraction of FP count fraction to run inside the function and return the results for  
        - max_fn (float): : max fraction of FN count fraction to run inside the function and return the results for  
        - truncated_feature_set (array): an array of features to run the analysis for (= None if the full set of features should be used)
        - test_or_val (str): a string spesifying whether the models should be tested on the validation or test dataset, default = "test"; ["test"|"val"]
    Returns:
        - cog_remov_add_accuracies (dict): a dictionary with the train model accuracies, where the key is (FN_rate, FP_rate) for the test data
        
    """
    cog_adding_rates = [fp for fp in [0.0, 0.05, 0.1, 0.15, 0.2] if fp <= max_fp]
    cog_removal_rates = [fn for fn in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] if fn <= max_fn]
    total = len(cog_removal_rates) * len(cog_adding_rates)

    # Cache test splits on CPU once
    if test_or_val == "test":
        test_data = {
            sid: (v["X_test"].cpu(), v["y_test"].cpu())
            for sid, v in all_splits_dict.items()
        }
    else:
        test_data = {
            sid: (v["X_val"].cpu(), v["y_val"].cpu())
            for sid, v in all_splits_dict.items()
        }        

    cog_remov_add_accuracies = {}

    # --- define evaluation function for parallel use ---
    def eval_one_noise(rem_rate, add_rate):
        mcc_arr, acc_arr, bacc_arr, prec_arr, rec_arr, f1_arr = [], [], [], [], [], []

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*glibc.*", category=FutureWarning)

            for split_id, pipe in trained_models.items():
                X_val_test, y_label_test = test_data[int(split_id)]

                if truncated_feature_set is not None:
                    X_val_test = X_val_test[:, truncated_feature_set]
    
                # Apply noise once per model-split
                X_val_test_noisy = flip_with_fractional_noise(
                    X_val_test, add_rate, rem_rate, noise_std=0.3, hard_fn_flag=hard_fn_flag
                )
                with suppress_xgb_warnings():
                    y_pred = pipe.predict(X_val_test_noisy)
    
                mcc_arr.append(matthews_corrcoef(y_label_test, y_pred))
                acc_arr.append(accuracy_score(y_label_test, y_pred))
                bacc_arr.append(balanced_accuracy_score(y_label_test, y_pred))
                prec_arr.append(precision_score(y_label_test, y_pred, zero_division=0))
                rec_arr.append(recall_score(y_label_test, y_pred, zero_division=0))
                f1_arr.append(f1_score(y_label_test, y_pred, zero_division=0))
    
            # Aggregate metrics
            return (rem_rate, add_rate, {
                "mcc": (np.mean(mcc_arr), np.std(mcc_arr)),
                "accuracy": (np.mean(acc_arr), np.std(acc_arr)),
                "balanced_accuracy": (np.mean(bacc_arr), np.std(bacc_arr)),
                "precision": (np.mean(prec_arr), np.std(prec_arr)),
                "recall": (np.mean(rec_arr), np.std(rec_arr)),
                "f1": (np.mean(f1_arr), np.std(f1_arr)),
            })
            
    with suppress_xgb_warnings():
        warnings.filterwarnings("ignore", message=".*glibc.*", category=FutureWarning)
    
        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_one_noise)(rem_rate, add_rate)
            for rem_rate, add_rate in tqdm(
                [
                    (r, a)
                    for r in cog_removal_rates
                    for a in cog_adding_rates
                ],
                total=total,
                desc="Evaluating noise settings..."
            )
        )

    # Convert to dictionary
    cog_remov_add_accuracies = {(r, a): metrics for (r, a, metrics) in results}

    return cog_remov_add_accuracies



def read_and_evaluate_models_for_x_and_sigma(trained_models_dir, x_noisy_samples, noise_type, metric, all_splits_dict, output_dir, clean_test_flag = True, add_rate = None, rem_rate = None, noise_std = 0.3, hard_fn_flag = None):
    mn_fn_arr = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0]
    mn_fp_arr = [0.0, 0.05, 0.1, 0.15, 0.2, 0.5]


    mean_rem_add_rates_tuples = [(add, rem) for add in mn_fp_arr for rem in mn_fn_arr]
    
    
    noise_increase_accuracy = {}
    for i in tqdm(range(len(mean_rem_add_rates_tuples)), desc="Processing noise rates..."):    
        (fp_rate_mean, fn_rate_mean) = mean_rem_add_rates_tuples[i]
        noise_rates = (fp_rate_mean, fn_rate_mean)

        filename = f"trained_models_fp_{fp_rate_mean}_fn_{fn_rate_mean}_noise_type_{noise_type}_x_{x_noisy_samples}.pkl"
    
       # filename = f"trained_models_fp_{fp_rate_mean}_fn_{fn_rate_mean}_sigma_fp_{sigma_fp}_sigma_fn_{sigma_fn}_x_{x_noisy_samples}.pkl"
        filepath = f"{output_dir}/{trained_models_dir}/{filename}"
        metrics_accum = {key: [] for key in ["mcc", "accuracy", "balanced_accuracy", "precision", "recall", "f1"]}
        if os.path.exists(filepath):
            loaded_models_dict = joblib.load(filepath)
            
            for split_id in loaded_models_dict.keys():#range(3):#all_splits_dict.keys(): #all_splits_dict.keys()
                trained_model = loaded_models_dict[split_id]
                X_val_test = all_splits_dict[split_id]["X_val"]
                y_label_test = all_splits_dict[split_id]["y_val"]
               
                if clean_test_flag == True:
                    y_pred = trained_model.predict(X_val_test)

                else:  
                    # Apply noise
                    X_val_test_noisy = flip_with_fractional_noise(
                        X_val_test.cpu(), add_rate, rem_rate, noise_std , hard_fn_flag = hard_fn_flag)
                    y_pred = trained_model.predict(X_val_test_noisy)

                metrics_accum["mcc"].append(matthews_corrcoef(y_label_test.cpu(), y_pred))
                metrics_accum["accuracy"].append(accuracy_score(y_label_test.cpu(), y_pred))
                metrics_accum["balanced_accuracy"].append(balanced_accuracy_score(y_label_test.cpu(), y_pred))
                metrics_accum["precision"].append(precision_score(y_label_test.cpu(), y_pred, zero_division=0))
                metrics_accum["recall"].append(recall_score(y_label_test.cpu(), y_pred, zero_division=0))
                metrics_accum["f1"].append(f1_score(y_label_test.cpu(), y_pred, zero_division=0))
            
            test_accuracy_scores = {k: [np.mean(v), np.std(v)] for k,v in metrics_accum.items()}    
        else:
            test_accuracy_scores = {k: [None, None] for k,v in metrics_accum.items()}
        noise_increase_accuracy[tuple(noise_rates)] = test_accuracy_scores
    
    noise_increase_accuracy_one_metric = {}
    for key in noise_increase_accuracy.keys():
        noise_increase_accuracy_one_metric[key] = noise_increase_accuracy[key] [metric]
        

    return noise_increase_accuracy_one_metric




def read_and_evaluate_models_for_x_and_sigma_regress(trained_models_dir, x_noisy_samples, noise_type, metric, all_splits_dict, output_directory, clean_test_flag = True, add_rate = None, rem_rate = None, noise_std = 0.3, hard_fn_flag = None):
    mn_fn_arr = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0]
    mn_fp_arr = [0.0, 0.05, 0.1, 0.15, 0.2, 0.5]
    # mn_fp_arr = [0.0, 0.05, 0.1, 0.15, 0.2]
    # mn_fn_arr = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    mean_rem_add_rates_tuples = [(add, rem) for add in mn_fp_arr for rem in mn_fn_arr]
    mean_rem_add_rates_tuples
    
    noise_increase_accuracy = {}
    for i in tqdm(range(len(mean_rem_add_rates_tuples)), desc="Processing noise rates..."):    
        (fp_rate_mean, fn_rate_mean) = mean_rem_add_rates_tuples[i]
        noise_rates = (fp_rate_mean, fn_rate_mean)

        filename = f"trained_models_fp_{fp_rate_mean}_fn_{fn_rate_mean}_noise_type_{noise_type}_x_{x_noisy_samples}.pkl"
    
       # filename = f"trained_models_fp_{fp_rate_mean}_fn_{fn_rate_mean}_sigma_fp_{sigma_fp}_sigma_fn_{sigma_fn}_x_{x_noisy_samples}.pkl"
        filepath = f"{output_directory}/{trained_models_dir}/{filename}"
        metrics_accum = {key: [] for key in ["mcc", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "rmse", "r2"]}
        if os.path.exists(filepath):
            loaded_models_dict = joblib.load(filepath)
            
            
            for split_id in loaded_models_dict.keys():#range(3):#all_splits_dict.keys(): #all_splits_dict.keys()
                trained_model = loaded_models_dict[split_id]
                classifier, regressor_low, regressor_high = trained_model
                
                X_val_test = all_splits_dict[split_id]["X_test"]
                y_label_test = all_splits_dict[split_id]["y_test"]

                # Convert to numpy
                y_train_np = y_label_test.cpu().numpy().flatten()
                range_labels = label_ogt_range(y_label_test)
                le = LabelEncoder()
                range_ids = le.fit_transform(range_labels)  # Converts to 0,1,2
                label_to_int = {'low': 0, 'high': 1}
                range_ids = np.vectorize(label_to_int.get)(range_labels)
                
                if clean_test_flag == True:
                    classifier_pred = classifier.predict(X_val_test)
                    classifier_probs = classifier.predict_proba(X_val_test)

                    # Final prediction
                    pred_low  = regressor_low.predict(X_val_test)
                    pred_high  = regressor_high.predict(X_val_test)
                    final_pred = (classifier_probs[:, 0]  * pred_low +classifier_probs[:, 1] * pred_high)
                    
                else:  
                    # Apply noise
                    X_val_test_noisy = flip_with_fractional_noise(
                        X_val_test.cpu(), add_rate, rem_rate, noise_std , hard_fn_flag = hard_fn_flag)
                    classifier_pred = classifier.predict(X_val_test_noisy)
                    classifier_probs = classifier.predict_proba(X_val_test_noisy)

                    # Final prediction
                    pred_low  = regressor_low.predict(X_val_test_noisy)
                    pred_high  = regressor_high.predict(X_val_test_noisy)
                    final_pred = (classifier_probs[:, 0]  * pred_low +classifier_probs[:, 1] * pred_high)

                metrics_accum["mcc"].append(matthews_corrcoef(range_ids, classifier_pred))
                metrics_accum["accuracy"].append(accuracy_score(range_ids, classifier_pred))
                metrics_accum["balanced_accuracy"].append(balanced_accuracy_score(range_ids, classifier_pred))
                metrics_accum["precision"].append(precision_score(range_ids, classifier_pred, zero_division=0))
                metrics_accum["recall"].append(recall_score(range_ids, classifier_pred, zero_division=0))
                metrics_accum["f1"].append(f1_score(range_ids, classifier_pred, zero_division=0))
                metrics_accum["rmse"].append(np.sqrt(mean_squared_error(y_label_test.cpu(), final_pred))),
                metrics_accum["r2"].append(r2_score(y_label_test.cpu(), final_pred))
            
            test_accuracy_scores = {k: [np.mean(v), np.std(v)] for k,v in metrics_accum.items()}    
        else:
            test_accuracy_scores = {k: [None, None] for k,v in metrics_accum.items()}
        noise_increase_accuracy[tuple(noise_rates)] = test_accuracy_scores
    
    noise_increase_accuracy_one_metric = {}
    for key in noise_increase_accuracy.keys():
        noise_increase_accuracy_one_metric[key] = noise_increase_accuracy[key] [metric]
        
  #  areas_mn_std=areas_across_fps(noise_increase_accuracy_one_metric)     
    
    return noise_increase_accuracy_one_metric#, areas_mn_std


def label_ogt_range(y,high_thresh=45):
    labels = []
    for val in y:
        if val < high_thresh:
            labels.append('low')
        else:
            labels.append('high')
    return np.array(labels)


def eval_trained_models_on_noisy_data_classif_and_regress(
    trained_models, all_splits_dict, hard_fn_flag=False, max_fp=0.2, max_fn=1, n_jobs=-1, truncated_feature_set=None
):
    cog_adding_rates = [r for r in [0.0, 0.05, 0.1, 0.15, 0.2] if r <= max_fp]
    cog_removal_rates = [r for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] if r <= max_fn]
    noise_std = 0.3

    # --- Pre-cache data ---
    split_cache = {}
    for split_id, models in trained_models.items():
        X_test = all_splits_dict[split_id]["X_test"]
        y_test = all_splits_dict[split_id]["y_test"]

        if truncated_feature_set is not None:
            X_test = X_test[:, truncated_feature_set]

        range_labels = label_ogt_range(y_test)
        label_to_int = {'low': 0, 'high': 1}
        range_ids = np.vectorize(label_to_int.get)(range_labels)

        split_cache[split_id] = (X_test, y_test, range_ids, models)

    # --- Define worker function ---
    def evaluate_one_condition(rem_rate, add_rate):
        mcc_arr, acc_arr, bal_acc_arr = [], [], []
        prec_arr, rec_arr, f1_arr = [], [], []
        rmse_arr, r2_arr = [], []

        for split_id, (X_test, y_test, range_ids, models) in split_cache.items():
            classifier, reg_low, reg_high = models

            # Apply noise
            X_noisy = flip_with_fractional_noise(
                X_test, add_rate, rem_rate, noise_std, hard_fn_flag=True
            )

            # --- Classification ---
            y_pred = classifier.predict(X_noisy)
            y_proba = classifier.predict_proba(X_noisy)

            mcc_arr.append(matthews_corrcoef(range_ids, y_pred))
            acc_arr.append(accuracy_score(range_ids, y_pred))
            bal_acc_arr.append(balanced_accuracy_score(range_ids, y_pred))
            prec_arr.append(precision_score(range_ids, y_pred, zero_division=0))
            rec_arr.append(recall_score(range_ids, y_pred, zero_division=0))
            f1_arr.append(f1_score(range_ids, y_pred, zero_division=0))

            # --- Regression ---
            pred_low = reg_low.predict(X_noisy)
            pred_high = reg_high.predict(X_noisy)
            final_pred = y_proba[:, 0] * pred_low + y_proba[:, 1] * pred_high

            rmse_arr.append(np.sqrt(mean_squared_error(y_test, final_pred)))
            r2_arr.append(r2_score(y_test, final_pred))

        # Return aggregated results
        return (rem_rate, add_rate, {
            "mcc": (np.mean(mcc_arr), np.std(mcc_arr)),
            "accuracy": (np.mean(acc_arr), np.std(acc_arr)),
            "balanced_accuracy": (np.mean(bal_acc_arr), np.std(bal_acc_arr)),
            "precision": (np.mean(prec_arr), np.std(prec_arr)),
            "recall": (np.mean(rec_arr), np.std(rec_arr)),
            "f1": (np.mean(f1_arr), np.std(f1_arr)),
            "rmse": (np.mean(rmse_arr), np.std(rmse_arr)),
            "r2": (np.mean(r2_arr), np.std(r2_arr))
        })

    # --- Run in parallel ---
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_one_condition)(rem, add)
        for rem in cog_removal_rates#tqdm(cog_removal_rates, desc="removal rates")
        for add in cog_adding_rates
    )

    # --- Collect results into dict ---
    cog_remov_add_accuracies = {(rem, add): metrics for rem, add, metrics in results}
    return cog_remov_add_accuracies

    
