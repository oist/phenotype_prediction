import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests

from sklearn.model_selection import GroupKFold

import shap
from xgboost import XGBClassifier, XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef, make_scorer

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
from joblib import Parallel, delayed


THREADS = 64

def xgboost_train_accur(X_train, y_train, X_test, y_test, device, groups=None, n_splits = 5):
    """
    Trains XGBoost for the specified X/y train and test data.
    Returns dictionaries with training accuracy measures calculated for cross-validation and test.
    """

    
    # Initialize training pipelina
    pipe = make_pipeline(XGBClassifier(n_jobs=THREADS if device == "cpu" else None, tree_method="gpu_hist" if device == "cpu" else "hist"))

    
    
    # create a scorer for MCC
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    # then pass it in your scoring dict
    scoring = {
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'mcc': mcc_scorer,
        'f1': mcc_scorer
    }

    # Choose CV strategy
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits)
        cv_results = cross_validate(
            pipe,
            X_train.cpu(), y_train.cpu(),
            cv=cv,
            groups=groups, 
            scoring=scoring,
            return_train_score=False
        )

    else:
        cv = n_splits
        cv_results = cross_validate(pipe, X_train.cpu(), y_train.cpu(), cv=cv, scoring=scoring, return_train_score=False)
    
    cv_accuracy_scores = {
        'mcc': np.mean(cv_results['test_mcc']),
        'balanced_accuracy': np.mean(cv_results['test_balanced_accuracy']),
        'accuracy': np.mean(cv_results['test_accuracy']),
        'precision': np.mean(cv_results['test_precision']),
        'recall': np.mean(cv_results['test_recall']),
        'f1': np.mean(cv_results['test_f1']),
    }

    # Fit on full training set
    pipe.fit(X_train.cpu(), y_train.cpu())

    # Test set predictions
    y_pred = pipe.predict(X_test.cpu())
    y_prob = pipe.predict_proba(X_test.cpu())[:, 1] if len(np.unique(y_train.cpu())) == 2 else None  # binary case

    # Collect final metrics on test set
    test_accuracy_scores = {
        'mcc': matthews_corrcoef(y_test.cpu(), y_pred),
        'accuracy': accuracy_score(y_test.cpu(), y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test.cpu(), y_pred),
        'precision': precision_score(y_test.cpu(), y_pred, zero_division=0),
        'recall': recall_score(y_test.cpu(), y_pred, zero_division=0),
        'f1': f1_score(y_test.cpu(), y_pred, zero_division=0),
    }
    return cv_accuracy_scores, test_accuracy_scores

import random
def xgboost_accur_select_features(X_train, X_test, y_train, y_test, sorted_indices, feat_step, device, feat_removal = False, train_test_feat_apply = True, groups=None):
    cv_accur_arr = []
    test_accur_arr = []
 
    num_feat = range(1,len(sorted_indices),feat_step)
    num_feat_plot = []
    for N in num_feat:
        
        if feat_removal == False:
            select_feat = list(sorted_indices[:N])
        else:
            select_feat = list(sorted_indices[N:])
      #  select_feat = random.sample(sorted_indices, N)  
       # print(select_feat)
        num_feat_plot.append(N) 

        if train_test_feat_apply == True:
            X_train_select_feat = X_train[:, select_feat] 
            X_test_select_feat = X_test[:, select_feat]
        else:
            X_train_select_feat = X_train.clone()  
            X_test_select_feat = X_test.clone()
            X_test_select_feat[:, select_feat] = 0


       


        cv_accuracy_scores, test_accuracy_scores = xgboost_train_accur(X_train_select_feat, y_train, X_test_select_feat, y_test, device, groups=groups)
        cv_accur_arr.append(cv_accuracy_scores)
        test_accur_arr.append(test_accuracy_scores)
        # if N==1:
        #     print(f"feat = {select_feat}")
        #     print(f"shape = {X_train_select_feat.shape}")
        #     print(f"sum = {sum(X_train_select_feat)}")
        #     print(X_train_select_feat)

        #     print(test_accur_arr)
    return cv_accur_arr,  test_accur_arr, num_feat_plot 


def mutual_info_features(X_train, y_train, X_train_column_names, random_state, contin_flag = False):
    if contin_flag == False:
        mutual_info = mutual_info_classif(X_train, y_train, random_state=random_state)
    else:
        mutual_info = mutual_info_regression(X_train, y_train, random_state=random_state)
    
    sorted_indices = np.argsort(mutual_info)[::-1] 
    sorted_mi = [mutual_info[i] for i in sorted_indices]
    sorted_names = [X_train_column_names[i] for i in sorted_indices]

    return sorted_indices, sorted_mi, sorted_names

def random_forest_features(X_train, y_train, X_train_column_names, random_state, contin_flag = False):

    if contin_flag == False:
        # Train a Random Forest model
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:    
        rf = RandomForestRegressor(n_estimators=100, random_state=random_state)

    rf.fit(X_train, y_train)

    # Get feature importances
    importances = rf.feature_importances_
    # print("Feature importances:", importances)
    # print(len(importances))

    # Select features based on importance threshold
    selector = SelectFromModel(rf, threshold='mean', prefit=True)
    X_selected = selector.transform(X_train)

    print(f"Original feature count: {X_train.shape[1]}, Selected feature count: {X_selected.shape[1]}")
    

    # plt.figure(figsize=(10, 2))
    # plt.bar(range(X_train.shape[1]), importances)
    # plt.xlabel("Feature Index")
    # plt.ylabel("Importance")
    # plt.ylim([0, max(importances)])
    # plt.show()

    sorted_indices = np.argsort(importances)[::-1]  # Reverse the order to get descending sort

    # Step 2: Use the sorted indices to get the sorted importances and corresponding names
    sorted_importances = [importances[i] for i in sorted_indices]
    sorted_names = [X_train_column_names[i] for i in sorted_indices]

    return sorted_indices, sorted_importances, sorted_names

# --- 1. Label each sample as low, mid, or high OGT group
def label_ogt_range(y,high_thresh=45):
    labels = []
    for val in y:
        if val < high_thresh:
            labels.append('low')
        else:
            labels.append('high')
    return np.array(labels)

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import shap
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.pipeline import make_pipeline

def shap_features(X_train, y_train, X_column_names, device, contin_flag=False):
    if not contin_flag:
        model = XGBClassifier(
            n_jobs=None if device != "cpu" else -1,
            tree_method="gpu_hist" if device != "cpu" else "hist",
            eval_metric="logloss"
        )
    else:
        model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            n_jobs=-1,
            tree_method="gpu_hist" if device != "cpu" else "hist",
        )

    # convert to numpy if torch
    if hasattr(X_train, "cpu"):
        X_np = X_train.cpu().numpy()
        y_np = y_train.cpu().numpy()
    else:
        X_np, y_np = X_train, y_train

    # fit model
    model.fit(X_np, y_np)

    # use TreeExplainer (much faster & correct for XGB)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_np)

    # if classifier, shap_values is a list (per class)
    if not contin_flag and isinstance(shap_values, list):
        shap_values = shap_values[1]  # take class 1 SHAP values

    # feature importances
    abs_shap_vals = np.abs(shap_values)
    mean_abs_shap_vals = np.mean(abs_shap_vals, axis=0)

    sorted_indices = np.argsort(mean_abs_shap_vals)[::-1]
    sorted_importances = mean_abs_shap_vals[sorted_indices]
    sorted_names = [X_column_names[i] for i in sorted_indices]

    return sorted_indices, sorted_importances, sorted_names, shap_values


       #  shap_vals_contin = defaultdict(list)

       #  X_np = X_train.cpu().numpy()

       #  range_labels = label_ogt_range(y_train)
       #  le = LabelEncoder()
       #  range_ids = le.fit_transform(range_labels)  # Converts to 0,1,2
       #  range_ids_np = range_ids if isinstance(range_ids, np.ndarray) else range_ids.cpu().numpy()
        
       #  gating_model = XGBClassifier(
       #      n_jobs=-1,
       #      tree_method=tree_method,
       #      device=device,
       #      #predictor=predictor,
       #      objective="binary:logistic",
       #      eval_metric="logloss"
       #  )
       #  label_to_int = {'low': 0, 'high': 1}
       #  range_ids = np.vectorize(label_to_int.get)(range_labels)
    
       #  classes = np.unique(range_ids)
       #  weights = compute_class_weight(class_weight='balanced', classes=classes, y=range_ids)
       #  class_weights = dict(zip(classes, weights))
       #  #class_weights[0]=1
       # # class_weights[1]=10
       #  sample_weights = np.array([class_weights[c] for c in range_ids])
        
       #  gating_model.fit(X_np, range_ids_np, sample_weight=sample_weights)
       #  # SHAP
       #  explainer = shap.Explainer(model, X_np)  # can also use shap.TreeExplainer(model)
       #  shap_values = explainer(X_np)
    
       #  # 5. Extract SHAP values and compute mean absolute SHAP value per feature
       #  shap_vals = shap_values.values  # shape: [n_samples, n_features]
       #  abs_shap_vals = np.abs(shap_vals)
       #  mean_abs_shap_vals = np.mean(abs_shap_vals, axis=0)
    
       #  # 6. Sort features by importance
       #  sorted_indices = np.argsort(mean_abs_shap_vals)[::-1]
       #  sorted_importances = mean_abs_shap_vals[sorted_indices]
       #  sorted_names = [X_column_names[i] for i in sorted_indices]
       #  local_dict = defaultdict(list) 
       #  local_dict["sorted_indices"] = sorted_indices
       #  local_dict["sorted_importances"] = sorted_importances
       #  local_dict["sorted_names"] = sorted_names
       #  shap_vals_contin["gating_model"] = local_dict
       #  #REgression
       #  y_train = y_train.squeeze()
        
       #  temp_bound = 45
       #  # Define masks (all 1D)
       #  low_mask  = y_train < temp_bound
       #  high_mask = y_train >= temp_bound
    
       #  # Apply masks correctly
       #  X_low, y_low   = X_train[low_mask].cpu(), y_train[low_mask].cpu()
       #  X_high, y_high = X_train[high_mask].cpu(), y_train[high_mask].cpu()
    
       #  model_low  = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05).fit(X_low, y_low)
       #  model_high =  XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05).fit(X_high, y_high)

       #  explainer_low = shap.Explainer(model_low)
       #  shap_values_low = explainer_low(X_val_low)

       #  explainer_high = shap.Explainer(model_high)
       #  shap_values_high = explainer_high(X_val_high)

        

        
def svc_features(X_train, y_train, X_train_column_names):
    scaler = MaxAbsScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    # Step 3: Fit the LinearSVC model with L1 penalty (for feature selection)
    svm = LinearSVC(C=0.01, penalty='l1', dual=False, max_iter=5000)
    svm.fit(X_train, y_train)

    # Step 4: Extract the absolute feature importance (model coefficients)
    feature_importance = np.abs(svm.coef_.ravel())

    # Step 5: Sort the indices by feature importance (from highest to lowest)
    sorted_indices = np.argsort(feature_importance)[::-1]  # Reverse order for descending

    sorted_importances = [feature_importance[i] for i in sorted_indices]
    sorted_names = [X_train_column_names[i] for i in sorted_indices]

    return sorted_indices, sorted_importances, sorted_names


def plot_accuracy_metric11(metric, test_accuracy_scores, cv_accuracy_scores, test_accur_arr, test_accur_arr_rem, cv_accur_arr, cv_accur_arr_rem, num_feat, n_cols):
    plt.axhline(y=test_accuracy_scores[metric], color='darkred', linestyle='--', linewidth=1.5, label='baseline test')
    plt.axhline(y=cv_accuracy_scores[metric], color='darkblue', linestyle='--', linewidth=1.5, label='baseline CV')

    plt.plot(num_feat, [scores[metric] for scores in test_accur_arr], c = "tab:red", label = "test | add")
    plt.plot(num_feat, [scores[metric] for scores in cv_accur_arr], c = "tab:blue", label = "cv | add")

    plt.plot([n_cols - n_feat for n_feat in num_feat],  [scores[metric] for scores in test_accur_arr_rem], c = "tab:red", label = "test | remove", alpha = 0.5)
    plt.plot([n_cols - n_feat for n_feat in num_feat], [scores[metric] for scores in cv_accur_arr_rem], c = "tab:blue", label = "cv | remove", alpha = 0.5)

    plt.xlabel("number of features added/removed")
    plt.ylabel(metric)
    plt.ylim(0.0, 1.1)

def make_cog_descr(df):
    """
    Get descriptions for COGs from NCBI.
    
    :param pd.DataFrame cogs_df: pandas dataframe with top COGs from MI, RandomForest, and SHAP feature selection
    :return: pandas.DataFrame
    """
    cogs_df = df.copy()

    url = 'https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2024/data/cog-24.def.tab'
    response = requests.get(url)
    
    data_lines = response.content.decode('utf-8').splitlines()
    cogs_descr = pd.DataFrame([line.split('\t') for line in data_lines if len(line.split('\t')) == 7],
                              columns=['COG_ID', 'Category', 'Description', 'Gene', 'Function', 'Gene_IDs', 'PDB_ID'])

    for column in ['MI', 'RandomForest', 'SHAP']:
        df_descr = pd.merge(cogs_df, cogs_descr[['COG_ID', 'Description']], 
                            how='left', left_on=[column], right_on=['COG_ID'], 
                            suffixes=('', f'_{column}'))
        cogs_df[column] = cogs_df[column] + ': ' + df_descr[f'Description'].fillna('')

    return cogs_df[['MI', 'RandomForest', 'SHAP']]

import copy
def random_feat_removal_curves(X_train, X_test, y_train, y_test, num_runs, feat_step, device, feat_removal, groups=None):
    tot_num_feat = X_train.cpu().shape[1]
    num_feat = range(1,tot_num_feat,feat_step)
    accuracy_one_point_arr = {
        'mcc': [],
        'balanced_accuracy':  [],
        'accuracy':  [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
    }
    cv_accur_arr_all_runs = [copy.deepcopy(accuracy_one_point_arr) for _ in num_feat]
    test_accur_arr_all_runs = [copy.deepcopy(accuracy_one_point_arr) for _ in num_feat]

    for i in range(num_runs):
        print(f"Processing random feature combo {i}")
       # shuffled_indices = np.random.permutation(tot_num_feat)
        cv_accur_arr, test_accur_arr, num_feat = xgboost_accur_select_features(X_train, X_test, y_train, y_test, list(range(X_train.shape[1])), feat_step, device, feat_removal, groups=groups)
        for j in range(len(num_feat)):
            for metric in cv_accur_arr[j].keys():
                cv_accur_arr_all_runs[j][metric].append(cv_accur_arr[j][metric])
                test_accur_arr_all_runs[j][metric].append(test_accur_arr[j][metric])
                
    accuracy_one_point_val = {
        'mcc': 0,
        'balanced_accuracy':  0,
        'accuracy':  0,
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'roc_auc': 0,
    }
    cv_accur_arr_all_runs_mn = [copy.deepcopy(accuracy_one_point_val) for _ in num_feat]
    cv_accur_arr_all_runs_std = [copy.deepcopy(accuracy_one_point_val) for _ in num_feat]
    test_accur_arr_all_runs_mn = [copy.deepcopy(accuracy_one_point_val) for _ in num_feat]
    test_accur_arr_all_runs_std = [copy.deepcopy(accuracy_one_point_val) for _ in num_feat]

    for j in range(len(num_feat)):
        for metric in cv_accur_arr_all_runs_mn[j].keys():
            cv_accur_arr_all_runs_mn[j][metric] = np.mean(cv_accur_arr_all_runs[j][metric])

        for metric in cv_accur_arr_all_runs_std[j].keys():
            cv_accur_arr_all_runs_std[j][metric] = np.std(cv_accur_arr_all_runs[j][metric])

        for metric in test_accur_arr_all_runs_mn[j].keys():
            test_accur_arr_all_runs_mn[j][metric] = np.mean(test_accur_arr_all_runs[j][metric])

        for metric in test_accur_arr_all_runs_std[j].keys():
            test_accur_arr_all_runs_std[j][metric] = np.std(test_accur_arr_all_runs[j][metric])
    return cv_accur_arr_all_runs_mn, cv_accur_arr_all_runs_std, test_accur_arr_all_runs_mn, test_accur_arr_all_runs_std, num_feat

def plot_accuracy_metric(baseline_test, baseline_cv, vary_test, vary_cv, rand_mean_test, rand_std_test, rand_mean_cv, rand_std_cv, num_feat_plot):
    rand_mean_test, rand_std_test = np.array(rand_mean_test), np.array(rand_std_test)
    rand_mean_cv, rand_std_cv = np.array(rand_mean_cv), np.array(rand_std_cv)
    plt.axhline(y=baseline_test, color='darkred', linestyle='--', linewidth=1, label='test baseline')
    plt.axhline(y=baseline_cv, color='darkblue', linestyle='--', linewidth=1, label='cv baseline')
    plt.plot(num_feat_plot, vary_test, c = "tab:red", alpha = 1, label = "test varying features", linewidth=1)
    plt.plot(num_feat_plot, vary_cv, c = "tab:blue", alpha = 1, label = "cv varying features", linewidth=1)
    plt.plot(num_feat_plot, rand_mean_test, label="test random, mean ± std", color='tab:orange', linewidth=1)
    plt.fill_between(num_feat_plot, rand_mean_test - rand_std_test, rand_mean_test + rand_std_test, alpha=0.3, color='tab:orange')
    plt.plot(num_feat_plot, rand_mean_cv, label="cv random, mean ± std", color='tab:green', linewidth=1)
    plt.fill_between(num_feat_plot, rand_mean_cv - rand_std_cv, rand_mean_cv + rand_std_cv, alpha=0.3, color='tab:green')
    plt.grid()





import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score
from collections import defaultdict
from functools import lru_cache
from joblib import Parallel, delayed
from tqdm import tqdm


def group_by_z(z):
    """Group indices by unique rows in z."""
    keys = [tuple(row) for row in z]
    groups = defaultdict(list)
    for idx, key in enumerate(keys):
        groups[key].append(idx)
    return groups


@lru_cache(maxsize=None)
def conditional_mutual_info_cached(x_bytes, y_bytes, z_bytes, contin):
    """Wrapper for caching CMI calculations using hashable inputs."""
    x = np.frombuffer(x_bytes, dtype=np.float64)
    y = np.frombuffer(y_bytes, dtype=np.float64)
    z = np.frombuffer(z_bytes, dtype=np.float64).reshape(-1, z_dim[0]) if z_bytes else np.array([])

    return conditional_mutual_info(x, y, z, contin)


def conditional_mutual_info(x, y, z, contin):
    """Estimate conditional mutual information I(x; y | z)"""
    x = np.asarray(x)
    y = np.asarray(y).ravel()

    if not contin:  # categorical
        x = x.astype(int)
        y = y.astype(int)
        if z.size == 0:
            return mutual_info_score(x, y)

        z = z.astype(int)
        groups = group_by_z(z)
        cmi = 0
        for indices in groups.values():
            if len(indices) <= 1:
                continue
                
            p_z = len(indices) / len(x)
            p_xy_z = mutual_info_score(x[indices], y[indices])
            cmi += p_z * p_xy_z
        return cmi

    else:  # continuous
        x = x.reshape(-1, 1) if x.ndim == 1 else x

        if z.size == 0:
            return mutual_info_regression(x, y)[0]

        groups = group_by_z(z)

        cmi = 0
        for indices in groups.values():
            if len(indices) <= 3:
                continue
                    
            p_z = len(indices) / len(x)
            x_vals = x[indices]
            y_vals = y[indices]
            p_xy_z = mutual_info_regression(x_vals, y_vals)[0]
            cmi += p_z * p_xy_z
        return cmi


def compute_cmi_parallel(f, X, y, MB, contin):
    z = X[list(MB)].values if MB else np.array([])

    # Convert to bytes for caching
    x_bytes = X[f].values.astype(np.float64).tobytes()
    y_bytes = y.astype(np.float64).tobytes()
    z_bytes = z.astype(np.float64).tobytes() if z.size else b''

    global z_dim
    z_dim = z.shape[1:] if z.size else (0,)

    cmi = conditional_mutual_info_cached(x_bytes, y_bytes, z_bytes, contin)
    return f, cmi

import warnings
import re
def iamb(X, y, contin=False, alpha=0.01, verbose=False, n_jobs=-1):

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=re.escape("Your system has an old version of glibc (< 2.28).")
    )

    MB = set()
    candidates = set(X.columns)
    added = True

    def maybe_tqdm(iterable, desc):
        return tqdm(iterable, desc=desc, leave=False) if verbose else iterable

    if verbose:
        print("=== FORWARD PHASE ===")
    while added:
        added = False
        candidate_list = list(candidates - MB)

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_cmi_parallel)(f, X, y, MB, contin)
            for f in maybe_tqdm(candidate_list, desc="Evaluating CMI")
        )

        mi_scores = {f: cmi for f, cmi in results}
        if mi_scores:
            best_feature = max(mi_scores, key=mi_scores.get)
            if mi_scores[best_feature] > alpha:
                MB.add(best_feature)
                added = True
                if verbose:
                    print(f"  Added: {best_feature}, CMI={mi_scores[best_feature]:.4f}")

    if verbose:
        print("=== BACKWARD PHASE ===")

    to_remove = []
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_cmi_parallel)(f, X, y, MB - {f}, contin)
        for f in maybe_tqdm(list(MB), desc="Checking removal")
    )

    for f, cmi in results:
        if cmi < alpha:
            to_remove.append((f, cmi))

    for f, cmi in to_remove:
        MB.remove(f)
        if verbose:
            print(f"  Removed: {f}, CMI={cmi:.4f}")

    return list(MB)



