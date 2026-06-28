"""
Utility functions for feature selection, model training, and evaluation.
"""

# Standard library 
import copy
import os
import pickle
import re
import subprocess
import warnings
from collections import Counter, defaultdict
from functools import lru_cache

# Third-party 
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import (
    SelectFromModel,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    mean_squared_error,
    mutual_info_score,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor

THREADS = 64

# Hardware helpers

def is_gpu_available() -> bool:
    try:
        subprocess.check_output(["nvidia-smi"])
        return True
    except Exception:
        return False

# Calibration

def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) for binary predictions.

    Groups predicted probabilities into equally spaced bins and computes
    the weighted mean absolute difference between mean predicted probability
    and empirical accuracy within each bin.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0 or 1).
    y_prob : array-like
        Predicted probabilities for the positive class.
    n_bins : int
        Number of equally spaced bins.

    Returns
    -------
    float
        ECE in [0, 1]; 0 = perfectly calibrated.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if np.any(mask):
            ece += np.abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.mean()
    return ece


def _ece_scorer_func(y_true, y_prob) -> float:
    return expected_calibration_error(y_true, y_prob)


# Core XGBoost training / evaluation

def xgboost_train_accur(X_train, y_train, X_test, y_test, device, groups=None, n_splits: int = 5):
    """
    Train an XGBoost classifier and return cross-validated and held-out metrics.

    Parameters
    ----------
    X_train, y_train : array-like
        Training features and labels (torch tensors or numpy arrays).
    X_test, y_test : array-like
        Test features and labels.
    device : str
        "cpu" or "cuda".
    groups : array-like, optional
        Group labels for StratifiedGroupKFold.
    n_splits : int
        Number of CV folds.

    Returns
    -------
    cv_accuracy_scores, test_accuracy_scores : dict
        Dicts keyed by metric name.
    """
    use_gpu = device != "cpu" and is_gpu_available()
    pipe = make_pipeline(
        XGBClassifier(
            n_jobs=None if use_gpu else THREADS,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
        )
    )
    ece_scorer = make_scorer(_ece_scorer_func, response_method="predict_proba", greater_is_better=False)
    mcc_scorer = make_scorer(matthews_corrcoef)

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "mcc": mcc_scorer,
        "f1": "f1",
        "ece": ece_scorer,
    }

    X_tr = X_train.cpu() if hasattr(X_train, "cpu") else X_train
    y_tr = y_train.cpu() if hasattr(y_train, "cpu") else y_train

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        if groups is not None:
            cv = StratifiedGroupKFold(n_splits=n_splits)
            cv_results = cross_validate(pipe, X_tr, y_tr, cv=cv, groups=groups,
                                        scoring=scoring, return_train_score=False)
        else:
            cv_results = cross_validate(pipe, X_tr, y_tr, cv=n_splits,
                                        scoring=scoring, return_train_score=False)

    cv_accuracy_scores = {
        "mcc": np.mean(cv_results["test_mcc"]),
        "balanced_accuracy": np.mean(cv_results["test_balanced_accuracy"]),
        "accuracy": np.mean(cv_results["test_accuracy"]),
        "precision": np.mean(cv_results["test_precision"]),
        "recall": np.mean(cv_results["test_recall"]),
        "f1": np.mean(cv_results["test_f1"]),
        "ece": -np.mean(cv_results["test_ece"]),
    }

    pipe.fit(X_tr, y_tr)

    X_te = X_test.cpu() if hasattr(X_test, "cpu") else X_test
    y_te = y_test.cpu() if hasattr(y_test, "cpu") else y_test

    y_pred = pipe.predict(X_te)
    y_prob = pipe.predict_proba(X_te)[:, 1] if len(np.unique(y_tr)) == 2 else None

    test_accuracy_scores = {
        "mcc": matthews_corrcoef(y_te, y_pred),
        "accuracy": accuracy_score(y_te, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "ece": expected_calibration_error(y_te, y_prob) if y_prob is not None else float("nan"),
    }
    return cv_accuracy_scores, test_accuracy_scores


# Feature-addition / removal curves

def xgboost_accur_select_features(
    X_train, X_test, y_train, y_test,
    sorted_indices, feat_step, device, split_id,
    add_rem_noise_rates,
    feat_removal: bool = False,
    train_test_feat_apply: bool = True,
    groups=None,
    fully_remove_feat_info: bool = False,
):
    """
    Sweep over feature subsets (adding or removing in SHAP order) and record
    CV and test metrics at each step.
    """
    cv_accur_arr, test_accur_arr = [], []

    cutoff = 30
    indices = (
        list(range(1, min(cutoff, len(sorted_indices))))
        + list(range(cutoff, len(sorted_indices), feat_step))
    )
    num_feat_plot = []

    for N in indices:
        select_feat = list(sorted_indices[:N]) if not feat_removal else list(sorted_indices[N:])
        num_feat_plot.append(N)

        if train_test_feat_apply:
            X_train_sel = X_train[:, select_feat]
            X_test_sel = X_test[:, select_feat]

            if add_rem_noise_rates is not None:
                add_rate, rem_rate = add_rem_noise_rates
                X_test_sel = flip_with_fractional_noise(  # noqa: F821 – defined externally
                    X_test_sel, add_rate, rem_rate, noise_std=0.3, hard_fn_flag=True
                )
        else:
            X_train_sel = X_train.clone() if hasattr(X_train, "clone") else X_train.copy()
            X_test_sel = X_test.clone() if hasattr(X_test, "clone") else X_test.copy()

            import torch
            all_features = torch.arange(X_test.shape[1])
            unselected = all_features[~torch.isin(all_features, torch.tensor(select_feat))]
            X_test_sel[:, unselected] = np.nan if fully_remove_feat_info else 0

            if add_rem_noise_rates is not None:
                add_rate, rem_rate = add_rem_noise_rates
                X_test_sel[:, select_feat] = flip_with_fractional_noise(  # noqa: F821
                    X_test_sel[:, select_feat], add_rate, rem_rate, noise_std=0.3, hard_fn_flag=True
                )

        cv_scores, test_scores = xgboost_train_accur(
            X_train_sel, y_train, X_test_sel, y_test, device, groups=groups
        )
        cv_accur_arr.append(cv_scores)
        test_accur_arr.append(test_scores)

    return cv_accur_arr, test_accur_arr, num_feat_plot


def shap_curves_file(
    train_test_feat_apply_flag, filename, feat_step, feat_removal,
    all_splits_dict, add_rem_noise_rates=None, fully_remove_feat_info=False,
):
    """Load cached accuracy curves from disk, or compute and save them."""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded existing results from {filename}")
        return results

    print("No existing file found. Computing from scratch...")
    results = defaultdict(dict)

    for split_id in all_shap_lists_dict.keys():  # noqa: F821 – module-level global
        if all_splits_dict[int(split_id)] == 0:
            continue

        split = all_splits_dict[int(split_id)]
        X_train = split["X_train"]
        y_train = split["y_train"]
        X_test  = split["X_test"]
        y_test  = split["y_test"]
        col_names = list(split["feature_names"])
        groups = split["taxa_group_names_train"]

        shap_list = all_shap_lists_dict[split_id]  # noqa: F821
        indices = [col_names.index(f) for f in shap_list if f in col_names]

        cv_arr, test_arr, num_feat = xgboost_accur_select_features(
            X_train.cpu(), X_test.cpu(), y_train.cpu(), y_test.cpu(),
            indices, feat_step, DEVICE,  # noqa: F821 – module-level global
            split_id, add_rem_noise_rates,
            feat_removal, train_test_feat_apply_flag,
            groups=groups,
            fully_remove_feat_info=fully_remove_feat_info,
        )

        results[split_id]["cv_accur"]   = cv_arr
        results[split_id]["test_accur"] = test_arr
        results[split_id]["num_feat"]   = num_feat
        print(f"  Split {split_id} done")

    with open(filename, "wb") as f:
        pickle.dump(results, f)
    return results


def random_feat_removal_curves(X_train, X_test, y_train, y_test, num_runs, feat_step, device, feat_removal, groups=None):
    """Run feature-sweep experiments with randomly shuffled feature orderings."""
    X_tr = X_train.cpu() if hasattr(X_train, "cpu") else X_train
    tot_num_feat = X_tr.shape[1]
    num_feat = list(range(1, tot_num_feat, feat_step))

    empty_point = {m: [] for m in ("mcc", "balanced_accuracy", "accuracy", "precision", "recall", "f1")}
    cv_runs  = [copy.deepcopy(empty_point) for _ in num_feat]
    test_runs = [copy.deepcopy(empty_point) for _ in num_feat]
    test_curves_per_run, cv_curves_per_run = [], []

    for i in range(num_runs):
        print(f"Random permutation run {i + 1}/{num_runs}")
        shuffled = np.random.permutation(tot_num_feat)
        cv_arr, test_arr, _ = xgboost_accur_select_features(
            X_train, X_test, y_train, y_test,
            shuffled, feat_step, device, split_id=None,
            add_rem_noise_rates=None, feat_removal=feat_removal, groups=groups,
        )
        test_curves_per_run.append(copy.deepcopy(test_arr))
        cv_curves_per_run.append(copy.deepcopy(cv_arr))

        for j, (cv_s, test_s) in enumerate(zip(cv_arr, test_arr)):
            for metric in empty_point:
                cv_runs[j][metric].append(cv_s[metric])
                test_runs[j][metric].append(test_s[metric])

    def _summarise(runs):
        mn  = [{m: np.mean(runs[j][m]) for m in empty_point} for j in range(len(num_feat))]
        std = [{m: np.std(runs[j][m])  for m in empty_point} for j in range(len(num_feat))]
        return mn, std

    cv_mn, cv_std     = _summarise(cv_runs)
    test_mn, test_std = _summarise(test_runs)
    return cv_mn, cv_std, test_mn, test_std, num_feat, test_curves_per_run, cv_curves_per_run


# Restricted feature-space evaluation (Markov Blanket experiments)

def find_accuracies_on_restricted_feat_space(
    all_splits_dict, all_markov_bound_dict_with_res, X_column_names, feature_condit, device
):
    """Evaluate classifier accuracy on feature subsets defined by Markov Blankets."""
    cv_splits   = defaultdict(list)
    test_splits = defaultdict(list)
    print("Processing splits...")

    for split_id in all_markov_bound_dict_with_res.keys():
        sid = str(split_id)
        split = all_splits_dict[int(sid)]
        X_train = split["X_train"]
        y_train = split["y_train"]
        X_test  = split["X_test"]
        y_test  = split["y_test"]
        groups  = split["taxa_group_names_train"]
        mbs     = all_markov_bound_dict_with_res[sid]["MB"]

        in_mb  = [i for i, v in enumerate(X_column_names) if v in mbs]
        out_mb = [i for i, v in enumerate(X_column_names) if v not in mbs]

        def _zeroed_test(mask_indices):
            X_mod = X_test.clone() if hasattr(X_test, "clone") else X_test.copy()
            X_mod[:, mask_indices] = np.nan
            return X_mod

        if feature_condit == "mb_train_test":
            cv_s, test_s = xgboost_train_accur(X_train[:, in_mb], y_train, X_test[:, in_mb], y_test, device, groups=groups)
        elif feature_condit == "mb_zero_test":
            cv_s, test_s = xgboost_train_accur(X_train, y_train, _zeroed_test(in_mb), y_test, device, groups=groups)
        elif feature_condit == "full":
            cv_s, test_s = xgboost_train_accur(X_train, y_train, X_test, y_test, device, groups=groups)
        elif feature_condit == "no_mb_train_test":
            cv_s, test_s = xgboost_train_accur(X_train[:, out_mb], y_train, X_test[:, out_mb], y_test, device, groups=groups)
        elif feature_condit == "no_mb_test":
            cv_s, test_s = xgboost_train_accur(X_train, y_train, _zeroed_test(out_mb), y_test, device, groups=groups)
        else:
            raise ValueError(f"Unknown feature_condit: {feature_condit!r}")

        for m in cv_s:
            cv_splits[m].append(cv_s[m])
            test_splits[m].append(test_s[m])

    def _agg(splits):
        return (
            {m: np.mean(splits[m]) for m in splits},
            {m: np.std(splits[m])  for m in splits},
        )

    cv_mn, cv_std     = _agg(cv_splits)
    test_mn, test_std = _agg(test_splits)
    print("Done!")
    return cv_mn, cv_std, test_mn, test_std


# Feature importance methods

def mutual_info_features(X_train, y_train, X_train_column_names, random_state, contin_flag=False):
    """Rank features by mutual information with the target."""
    mi = (mutual_info_regression if contin_flag else mutual_info_classif)(
        X_train, y_train, random_state=random_state
    )
    idx = np.argsort(mi)[::-1]
    return idx, [mi[i] for i in idx], [X_train_column_names[i] for i in idx]


def random_forest_features(X_train, y_train, X_train_column_names, random_state, contin_flag=False):
    """Rank features by Random Forest impurity importance."""
    Cls = RandomForestRegressor if contin_flag else RandomForestClassifier
    rf = Cls(n_estimators=100, random_state=random_state).fit(X_train, y_train)

    sel = SelectFromModel(rf, threshold="mean", prefit=True)
    n_sel = sel.transform(X_train).shape[1]
    print(f"Original: {X_train.shape[1]} features → selected: {n_sel}")

    idx = np.argsort(rf.feature_importances_)[::-1]
    return idx, [rf.feature_importances_[i] for i in idx], [X_train_column_names[i] for i in idx]


def shap_features(X_train, y_train, X_column_names, device, contin_flag=False):
    """
    Rank features by mean absolute SHAP value using XGBoost's native SHAP output.

    Works for both classification (contin_flag=False) and regression (contin_flag=True).

    Returns
    -------
    sorted_indices, sorted_importances, sorted_names, feature_shap_values
    """
    use_gpu = device != "cpu"
    kwargs = dict(tree_method="hist", device="cuda" if use_gpu else "cpu")

    model = (
        XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, n_jobs=-1, **kwargs)
        if contin_flag
        else XGBClassifier(n_jobs=None if use_gpu else -1, eval_metric="logloss",
                           use_label_encoder=False, **kwargs)
    )

    X_np = X_train.cpu().numpy() if hasattr(X_train, "cpu") else np.asarray(X_train)
    y_np = y_train.cpu().numpy() if hasattr(y_train, "cpu") else np.asarray(y_train)

    model.fit(X_np, y_np)

    shap_vals = model.get_booster().predict(xgb.DMatrix(X_np, label=y_np), pred_contribs=True)
    shap_vals = np.asarray(shap_vals)[:, :-1]  # drop bias column

    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1]
    return idx, mean_abs[idx], [X_column_names[i] for i in idx], shap_vals


def svc_features(X_train, y_train, X_train_column_names):
    """Rank features by L1-penalised LinearSVC coefficient magnitude."""
    MaxAbsScaler().fit_transform(X_train)  # scaling kept for parity; not passed to SVC
    svm = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=5000).fit(X_train, y_train)
    imp = np.abs(svm.coef_.ravel())
    idx = np.argsort(imp)[::-1]
    return idx, [imp[i] for i in idx], [X_train_column_names[i] for i in idx]


def shap_topN_frequency(shap_dict: dict, top_n: int) -> Counter:
    """
    Count how often each feature appears in the top-N SHAP features across splits.

    Parameters
    ----------
    shap_dict : dict
        ``{split_id: [feature_1, feature_2, ...]}`` ranked by SHAP importance.
    top_n : int
        Number of top features to consider per split.

    Returns
    -------
    Counter
        feature → frequency across splits.
    """
    counter: Counter = Counter()
    for features in shap_dict.values():
        counter.update(features[:top_n])
    return counter


# Accuracy-curve helpers

def accur_curves(accuracy_curves_all_splits_add_feat):
    """Unpack per-split accuracy-curve dicts into metric arrays."""
    bal_accur, recall, mcc, f1, num_feat = [], [], [], [], []
    for val in accuracy_curves_all_splits_add_feat.values():
        ba, re, mc, f = [], [], [], []
        for pt in val["test_accur"]:
            ba.append(pt["balanced_accuracy"])
            re.append(pt["recall"])
            mc.append(pt["mcc"])
            f.append(pt["f1"])
        bal_accur.append(ba)
        recall.append(re)
        mcc.append(mc)
        f1.append(f)
        num_feat.append(val["num_feat"])
    return bal_accur, recall, mcc, f1, num_feat


def accur_curves_regr(accuracy_curves_all_splits_add_feat):
    """Unpack regression accuracy curves (R² and RMSE)."""
    r2, rmse, num_feat = [], [], []
    for val in accuracy_curves_all_splits_add_feat.values():
        r2.append([pt["r2"]   for pt in val["test_accur"]])
        rmse.append([pt["rmse"] for pt in val["test_accur"]])
        num_feat.append(val["num_feat"])
    return r2, rmse, num_feat


def find_mean_std_curve(curves):
    """Compute element-wise mean and std across curves, truncated to the shortest."""
    min_len = min(len(c) for c in curves)
    arr = np.array([c[:min_len] for c in curves])
    return arr.mean(axis=0), arr.std(axis=0)


def find_decreas_curve_index(mean_curve, x_vals, thresh_percent=0.95, stability=5):
    """
    Return the x-value at which a curve first drops to ≤ thresh_percent of its
    starting value and stays there for `stability` consecutive steps.
    """
    y = np.asarray(mean_curve)
    threshold = thresh_percent * y[0]
    for i in range(len(y) - stability + 1):
        if np.all(y[i:i + stability] <= threshold):
            return x_vals[i]
    return None


def find_increas_curve_index(mean_curve, x_vals, thresh_percent=0.95):
    """Return the x-value at which a curve first reaches thresh_percent of its maximum."""
    y = np.asarray(mean_curve)
    idx = np.where(y >= thresh_percent * y[-1])[0][0]
    return x_vals[idx]


def find_mean_std_index_for_curves(curves, x_vals):
    """Return drop-off indices for each individual curve."""
    return [
        ind for curve in curves
        if (ind := find_decreas_curve_index(curve, x_vals)) is not None
    ]

# Conditional mutual information / IAMB

def _group_by_z(z):
    groups = defaultdict(list)
    for idx, key in enumerate(map(tuple, z)):
        groups[key].append(idx)
    return groups


def conditional_mutual_info(x, y, z, contin: bool):
    """Estimate I(x ; y | z) via discretised or continuous MI."""
    x = np.asarray(x)
    y = np.asarray(y).ravel()

    if not contin:
        x, y = x.astype(int), y.astype(int)
        if z.size == 0:
            return mutual_info_score(x, y)
        cmi = 0.0
        for idxs in _group_by_z(z.astype(int)).values():
            if len(idxs) > 1:
                cmi += (len(idxs) / len(x)) * mutual_info_score(x[idxs], y[idxs])
        return cmi

    x = x.reshape(-1, 1) if x.ndim == 1 else x
    if z.size == 0:
        return mutual_info_regression(x, y)[0]
    cmi = 0.0
    for idxs in _group_by_z(z).values():
        if len(idxs) > 3:
            cmi += (len(idxs) / len(x)) * mutual_info_regression(x[idxs], y[idxs])[0]
    return cmi


def _compute_cmi_worker(f, X, y, mb_set, contin):
    """Worker used by joblib for parallel CMI computation."""
    z = X[list(mb_set)].values if mb_set else np.array([]).reshape(0, 0)
    return f, conditional_mutual_info(X[f].values.astype(np.float64), y, z, contin)


def iamb(X, y, contin=False, alpha=0.01, verbose=False, n_jobs=-1):
    """
    Incremental Association Markov Blanket (IAMB) algorithm.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Target vector.
    contin : bool
        Use continuous MI estimator when True.
    alpha : float
        CMI threshold for add / remove decisions.
    verbose : bool
        Print progress.
    n_jobs : int
        Parallel workers (-1 = all CPUs).

    Returns
    -------
    list
        Selected feature names forming the Markov Blanket of y.
    """
    warnings.filterwarnings("ignore", category=FutureWarning,
                            message=re.escape("Your system has an old version of glibc (< 2.28)."))

    y = np.asarray(y)
    MB: set = set()
    candidates = set(X.columns)
    _wrap = lambda it, desc: tqdm(it, desc=desc, leave=False) if verbose else it  # noqa: E731

    # Forward phase
    if verbose:
        print("=== FORWARD PHASE ===")
    added = True
    while added:
        added = False
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_cmi_worker)(f, X, y, frozenset(MB), contin)
            for f in _wrap(list(candidates - MB), "Evaluating CMI")
        )
        if results:
            best, best_cmi = max(results, key=lambda r: r[1])
            if best_cmi > alpha:
                MB.add(best)
                added = True
                if verbose:
                    print(f"  Added: {best}, CMI={best_cmi:.4f}")

    # Backward phase
    if verbose:
        print("=== BACKWARD PHASE ===")
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_cmi_worker)(f, X, y, frozenset(MB - {f}), contin)
        for f in _wrap(list(MB), "Checking removal")
    )
    for f, cmi in results:
        if cmi < alpha:
            MB.discard(f)
            if verbose:
                print(f"  Removed: {f}, CMI={cmi:.4f}")

    return list(MB)


# Mixture-of-experts regression model

def label_ogt_range(y, high_thresh=45):
    """Label samples as 'low' or 'high' based on OGT threshold."""
    return np.where(np.asarray(y) < high_thresh, "low", "high")


def xgboost_mixture_of_experts_2_class_cv_full(
    X_train, y_train, range_ids, sample_weights, X_test, y_test,
    n_splits=5, taxonomy_labels=None, cv_flag=False,
):
    """
    Two-expert mixture model: a gating classifier routes samples to a
    low-OGT or high-OGT XGBoost regressor.

    Parameters
    ----------
    X_train, y_train : array-like
        Training data (torch tensors or numpy).
    range_ids : array-like
        Binary gate labels (0 = low, 1 = high).
    sample_weights : array-like
        Per-sample weights for the gating model.
    X_test, y_test : array-like
        Held-out evaluation data.
    n_splits : int
        CV folds (used when cv_flag=True).
    taxonomy_labels : array-like, optional
        Group labels for GroupKFold.
    cv_flag : bool
        Whether to run cross-validation on the training set.

    Returns
    -------
    dict
        Keys: cv_predictions, cv_true, cv_gate_probs, test_predictions,
              gating_model, gate_probs_test, cv_metrics, test_metrics.
    """
    temp_bound = 45

    def _to_np(arr):
        return arr.cpu().numpy() if hasattr(arr, "cpu") else np.asarray(arr)

    X_tr_np  = _to_np(X_train)
    y_tr_np  = _to_np(y_train).flatten()
    X_te_np  = _to_np(X_test)
    y_te_np  = _to_np(y_test).flatten()
    w_np     = _to_np(sample_weights)
    rid_np   = _to_np(range_ids)

    tree_method = "hist"
    xgb_device  = "cuda" if is_gpu_available() else "cpu"

    xgb_kwargs   = dict(tree_method=tree_method, device=xgb_device)
    gate_kwargs  = dict(**xgb_kwargs, objective="binary:logistic", eval_metric="logloss", n_jobs=-1)
    expert_kwargs = dict(**xgb_kwargs, n_estimators=100, max_depth=5, learning_rate=0.05)

    # ── Cross-validation ─────────────────────────────────────────────────────
    n = len(y_tr_np)
    final_cv_preds = np.zeros(n)
    true_cv_vals   = np.zeros(n)
    gate_probs_cv  = np.zeros((n, 2))
    gate_preds_cv  = np.zeros(n, dtype=int)
    gate_true_cv   = np.zeros(n, dtype=int)
    cv_metrics: dict = {}

    if cv_flag:
        fold_gen = (
            GroupKFold(n_splits=n_splits).split(X_tr_np, y_tr_np, groups=taxonomy_labels)
            if taxonomy_labels is not None
            else KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X_tr_np, y_tr_np)
        )

        for tr_idx, val_idx in fold_gen:
            X_tr, X_val = X_tr_np[tr_idx], X_tr_np[val_idx]
            y_tr, y_val = y_tr_np[tr_idx], y_tr_np[val_idx]
            w_tr, r_tr  = w_np[tr_idx], rid_np[tr_idx]

            gate = XGBClassifier(**gate_kwargs).fit(X_tr, r_tr, sample_weight=w_tr)
            gp_val = gate.predict_proba(X_val)
            gp_cv_pred = np.argmax(gp_val, axis=1)

            low_m, high_m = y_tr < temp_bound, y_tr >= temp_bound
            p_low  = XGBRegressor(**expert_kwargs).fit(X_tr[low_m],  y_tr[low_m]).predict(X_val)
            p_high = XGBRegressor(**expert_kwargs).fit(X_tr[high_m], y_tr[high_m]).predict(X_val)

            final_cv_preds[val_idx] = gp_val[:, 0] * p_low + gp_val[:, 1] * p_high
            true_cv_vals[val_idx]   = y_val
            gate_probs_cv[val_idx]  = gp_val
            gate_preds_cv[val_idx]  = gp_cv_pred
            gate_true_cv[val_idx]   = (y_val >= temp_bound).astype(int)

        cv_metrics = {
            "accuracy":          accuracy_score(gate_true_cv, gate_preds_cv),
            "balanced_accuracy": balanced_accuracy_score(gate_true_cv, gate_preds_cv),
            "precision":         precision_score(gate_true_cv, gate_preds_cv),
            "recall":            recall_score(gate_true_cv, gate_preds_cv),
            "f1":                f1_score(gate_true_cv, gate_preds_cv),
            "mcc":               matthews_corrcoef(gate_true_cv, gate_preds_cv),
            "ece":               expected_calibration_error(gate_true_cv, gate_probs_cv[:, 1]),
            "r2":                r2_score(true_cv_vals, final_cv_preds),
            "rmse":              np.sqrt(mean_squared_error(true_cv_vals, final_cv_preds)),
        }

    # ── Full-training final models ───────────────────────────────────────────
    gate_full = XGBClassifier(**gate_kwargs).fit(X_tr_np, rid_np, sample_weight=w_np)
    gp_test   = gate_full.predict_proba(X_te_np)
    gp_pred_te = np.argmax(gp_test, axis=1)
    gate_true_te = (y_te_np >= temp_bound).astype(int)

    low_m, high_m = y_tr_np < temp_bound, y_tr_np >= temp_bound
    p_low  = XGBRegressor(**expert_kwargs).fit(X_tr_np[low_m],  y_tr_np[low_m]).predict(X_te_np)
    p_high = XGBRegressor(**expert_kwargs).fit(X_tr_np[high_m], y_tr_np[high_m]).predict(X_te_np)
    final_test_pred = gp_test[:, 0] * p_low + gp_test[:, 1] * p_high

    test_metrics = {
        "accuracy":          accuracy_score(gate_true_te, gp_pred_te),
        "balanced_accuracy": balanced_accuracy_score(gate_true_te, gp_pred_te),
        "precision":         precision_score(gate_true_te, gp_pred_te),
        "recall":            recall_score(gate_true_te, gp_pred_te),
        "f1":                f1_score(gate_true_te, gp_pred_te),
        "mcc":               matthews_corrcoef(gate_true_te, gp_pred_te),
        "ece":               expected_calibration_error(gate_true_te, gp_test[:, 1]),
        "r2":                r2_score(y_te_np, final_test_pred),
        "rmse":              np.sqrt(mean_squared_error(y_te_np, final_test_pred)),
    }

    return {
        "cv_predictions":  final_cv_preds,
        "cv_true":         true_cv_vals,
        "cv_gate_probs":   gate_probs_cv,
        "test_predictions": final_test_pred,
        "gating_model":    gate_full,
        "gate_probs_test": gp_test,
        "cv_metrics":      cv_metrics,
        "test_metrics":    test_metrics,
    }


def find_accuracies_on_restricted_feat_space_contin(
    all_splits_dict,
    all_markov_bound_dict_with_res,
    X_column_names,
    feature_condit,
    device,
):
    cv_accur_dict_splits = defaultdict(list)
    test_accur_dict_splits = defaultdict(list)

    print("Processing splits...")

    for split_id in map(str, all_markov_bound_dict_with_res.keys()):

        X_val_train = all_splits_dict[int(split_id)]["X_train"]
        y_label_train = all_splits_dict[int(split_id)]["y_train"]
        X_val_test = all_splits_dict[int(split_id)]["X_test"]
        y_label_test = all_splits_dict[int(split_id)]["y_test"]

        range_labels = label_ogt_range(y_label_train)
        label_to_int = {"low": 0, "high": 1}
        range_ids = np.vectorize(label_to_int.get)(range_labels)

        classes = np.unique(range_ids)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=range_ids,
        )
        class_weights = dict(zip(classes, weights))
        sample_weights = np.array([class_weights[c] for c in range_ids])

        taxa_group_names_train = all_splits_dict[int(split_id)]["taxa_group_names_train"]
        mbs = all_markov_bound_dict_with_res[split_id]["MB"]

        if feature_condit == "mb_train_test":
            indices = [i for i, f in enumerate(X_column_names) if f in mbs]
            X_train, X_test = X_val_train[:, indices], X_val_test[:, indices]

        elif feature_condit == "mb_zero_test":
            indices = [i for i, f in enumerate(X_column_names) if f in mbs]
            X_train = X_val_train
            X_test = X_val_test.clone()
            X_test[:, indices] = 0

        elif feature_condit == "full":
            X_train, X_test = X_val_train, X_val_test

        elif feature_condit == "no_mb_train_test":
            indices = [i for i, f in enumerate(X_column_names) if f not in mbs]
            X_train, X_test = X_val_train[:, indices], X_val_test[:, indices]

        elif feature_condit == "no_mb_test":
            indices = [i for i, f in enumerate(X_column_names) if f not in mbs]
            X_train = X_val_train
            X_test = X_val_test.clone()
            X_test[:, indices] = 0

        else:
            raise ValueError(f"Unknown feature_condit: {feature_condit}")

        results = xgboost_mixture_of_experts_2_class_cv_full(
            X_train,
            y_label_train,
            range_ids,
            sample_weights,
            X_test,
            y_label_test,
            taxonomy_labels=taxa_group_names_train,
        )

        for metric, value in results["test_metrics"].items():
            if metric in results["cv_metrics"]:
                cv_accur_dict_splits[metric].append(results["cv_metrics"][metric])
            test_accur_dict_splits[metric].append(value)

    cv_mean, cv_std = defaultdict(float), defaultdict(float)
    test_mean, test_std = defaultdict(float), defaultdict(float)

    for metric in test_accur_dict_splits:

        cv_vals = cv_accur_dict_splits.get(metric, [])
        test_vals = test_accur_dict_splits.get(metric, [])

        cv_mean[metric] = np.mean(cv_vals) if len(cv_vals) else np.nan
        cv_std[metric] = np.std(cv_vals) if len(cv_vals) else np.nan

        test_mean[metric] = np.mean(test_vals) if len(test_vals) else np.nan
        test_std[metric] = np.std(test_vals) if len(test_vals) else np.nan

    print("Done!")
    return cv_mean, cv_std, test_mean, test_std


# COG annotation helper

def make_cog_descr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate a dataframe of top COG IDs with NCBI descriptions.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns 'MI', 'RandomForest', 'SHAP' containing COG IDs.

    Returns
    -------
    pd.DataFrame
        Same columns with descriptions appended (``COG_ID: Description``).
    """
    url = "https://ftp.ncbi.nlm.nih.gov/pub/COG/COG2024/data/cog-24.def.tab"
    lines = requests.get(url).content.decode("utf-8").splitlines()
    descr_df = pd.DataFrame(
        [l.split("\t") for l in lines if len(l.split("\t")) == 7],
        columns=["COG_ID", "Category", "Description", "Gene", "Function", "Gene_IDs", "PDB_ID"],
    )

    out = df.copy()
    for col in ("MI", "RandomForest", "SHAP"):
        merged = pd.merge(out[[col]], descr_df[["COG_ID", "Description"]],
                          left_on=col, right_on="COG_ID", how="left")
        out[col] = out[col] + ": " + merged["Description"].fillna("")

    return out[["MI", "RandomForest", "SHAP"]]