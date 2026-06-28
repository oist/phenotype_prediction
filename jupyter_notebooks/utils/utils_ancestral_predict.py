"""
Noise injection, model evaluation under noise, and plotting utilities.
"""

#  Standard library 
import contextlib
import os
import sys
import warnings

#  Third-party 
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from collections import defaultdict
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")


# Calibration

def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) for binary predictions.

    Bins predicted probabilities into equally spaced intervals and computes
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
        if mask.any():
            ece += np.abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.mean()
    return ece


# Noise injection

def _sample_noise_rate(mean: float, noise_type: str) -> float:
    """Draw a single noise-rate sample from the specified distribution."""
    if noise_type == "exp":
        return np.random.exponential(scale=mean)
    if noise_type == "gamma":
        scale = 0.5
        return np.random.gamma(mean / scale, scale)
    if noise_type == "unif":
        return np.random.uniform(0, 2 * mean)
    raise ValueError(f"Unknown noise_type: {noise_type!r}. Choose 'exp', 'gamma', or 'unif'.")


def apply_noise(genome: torch.Tensor, fp_rate: float, fn_rate: float) -> torch.Tensor:
    """
    Apply Poisson-distributed false-positive and false-negative noise to a
    single genome (count) vector.

    False negatives subtract Poisson(count * fn_rate) from each entry.
    False positives add Poisson(fp_rate) to every entry (including zeros).
    Result is clamped to ≥ 0.
    """
    genome_noisy = genome.float().clone()
    losses = torch.poisson(genome.float() * fn_rate)
    genome_noisy = torch.clamp(genome_noisy - losses.int(), min=0)

    fp_add = torch.poisson(torch.full_like(genome, fp_rate, dtype=torch.float))
    genome_noisy = genome_noisy + fp_add
    return genome_noisy


def flip_with_fractional_noise(
    X: torch.Tensor,
    fp_rate: float,
    fn_rate: float,
    noise_std: float = 0.3,
    hard_fn_flag: bool = False,
) -> torch.Tensor:
    """
    Apply fractional false-positive / false-negative noise to a feature matrix.

    For each row:
    - False negatives: a fraction `fn_rate` of positive entries are set to 0
      (hard_fn_flag=True) or decremented by 1 (hard_fn_flag=False).
    - False positives: a fraction `fp_rate` of all entries are incremented by 1.

    Result is clamped to ≥ 0.

    Parameters
    ----------
    X : torch.Tensor, shape (n_samples, n_features)
    fp_rate : float
    fn_rate : float
    noise_std : float
        Std of optional Gaussian perturbation (currently unused in hard mode).
    hard_fn_flag : bool
        If True, set affected entries to 0; otherwise decrement by 1.

    Returns
    -------
    torch.Tensor
        Noisy copy of X.
    """
    X_noisy = X.float().clone()
    for i in range(X_noisy.shape[0]):
        pos_idx = torch.nonzero(X[i] > 0).flatten()
        n_fn = int(round(fn_rate * len(pos_idx)))
        if n_fn > 0:
            fn_idx = pos_idx[torch.randperm(len(pos_idx))[:n_fn]]
            if hard_fn_flag:
                X_noisy[i, fn_idx] = 0
            else:
                X_noisy[i, fn_idx] -= 1

        all_idx = torch.nonzero(X[i] > -1).flatten()
        n_fp = int(round(fp_rate * len(all_idx)))
        if n_fp > 0:
            fp_idx = all_idx[torch.randperm(len(all_idx))[:n_fp]]
            X_noisy[i, fp_idx] += 1

    return torch.clamp(X_noisy, min=0.0)


def augment_data_with_noise(
    X: torch.Tensor,
    y: torch.Tensor,
    n_clones: int,
    mean_fp: float,
    mean_fn: float,
    noise_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Augment a dataset by adding `n_clones` noisy copies of each sample.

    Noise rates are drawn independently per clone from the specified
    distribution (``noise_type`` ∈ {'exp', 'gamma', 'unif'}).

    Parameters
    ----------
    X : torch.Tensor, shape (n_samples, n_features)
    y : torch.Tensor, shape (n_samples,) or (n_samples, 1)
    n_clones : int
        Number of noisy copies to generate per original sample.
    mean_fp, mean_fn : float
        Mean false-positive / false-negative rates passed to the sampler.
    noise_type : str
        Distribution from which to sample noise rates.

    Returns
    -------
    X_augmented, y_augmented : torch.Tensor
        Shuffled augmented dataset (original samples included).
    """
    if n_clones == 0:
        return X.clone(), y.clone()

    X_parts = [X]
    y_parts = [y]

    for i in range(X.shape[0]):
        genome = X[i]
        label  = y[i]
        for _ in range(n_clones):
            fp = _sample_noise_rate(mean_fp, noise_type)
            fn = _sample_noise_rate(mean_fn, noise_type)
            X_parts.append(apply_noise(genome, fp, fn).unsqueeze(0))
            y_parts.append(label.unsqueeze(0))

    X_aug = torch.cat(X_parts, dim=0)
    y_aug = torch.cat(y_parts, dim=0)

    perm = torch.randperm(len(X_aug))
    return X_aug[perm], y_aug[perm]

def process_res(res):
    return res.iloc[0] if not res.empty else ''

# Utility helpers

def label_ogt_range(y, high_thresh: float = 45.0) -> np.ndarray:
    """Label samples as 'low' or 'high' based on an OGT threshold."""
    return np.where(np.asarray(y).flatten() < high_thresh, "low", "high")


def _to_numpy(arr) -> np.ndarray:
    if hasattr(arr, "cpu"):
        return arr.cpu().numpy().flatten()
    return np.asarray(arr).flatten()


def _agg(arr: list) -> tuple[float, float]:
    """Return (mean, std) of a list."""
    return float(np.mean(arr)), float(np.std(arr))


@contextlib.contextmanager
def _suppress_stderr():
    """Redirect stderr to /dev/null (suppresses XGBoost C++ warnings)."""
    with open(os.devnull, "w") as fnull:
        old = sys.stderr
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stderr = old


# Noise-rate grid definitions

_FP_GRID = [0.0, 0.05, 0.1, 0.15, 0.2]
_FN_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

_FP_MEAN_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.5]
_FN_MEAN_GRID = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0]


def _filter_grid(grid: list, lo: float, hi: float) -> list:
    return [v for v in grid if lo <= v <= hi]


# Model evaluation under noise

def eval_trained_models_on_noisy_data(
    all_splits_dict: dict,
    trained_models: dict,
    hard_fn_flag: bool = False,
    min_fp: float = 0.0,
    min_fn: float = 0.0,
    max_fp: float = 0.2,
    max_fn: float = 1.0,
    n_jobs: int = -1,
    truncated_feature_set=None,
    test_or_val: str = "test",
) -> dict:
    """
    Evaluate trained classifiers on a grid of (FN rate, FP rate) noise levels.

    Parameters
    ----------
    all_splits_dict : dict
        Per-split data; each value must have 'X_test'/'X_val' and 'y_test'/'y_val'.
    trained_models : dict
        {split_id: fitted pipeline} mapping.
    hard_fn_flag : bool
        If True, false-negative entries are zeroed; otherwise decremented by 1.
    min_fp, min_fn : float
        Lower bounds of the noise grid (inclusive).
    max_fp, max_fn : float
        Upper bounds of the noise grid (inclusive).
    n_jobs : int
        Parallel workers passed to joblib.
    truncated_feature_set : array-like or None
        Optional feature-index subset applied before noise injection.
    test_or_val : str
        Which split partition to evaluate on ('test' or 'val').

    Returns
    -------
    dict
        ``{(fn_rate, fp_rate): {metric: (mean, std), ...}}``
        Metrics: mcc, accuracy, balanced_accuracy, precision, recall, f1, brier, ece.
    """
    fp_rates = _filter_grid(_FP_GRID, min_fp, max_fp)
    fn_rates = _filter_grid(_FN_GRID, min_fn, max_fn)
    x_key = "X_test" if test_or_val == "test" else "X_val"
    y_key = "y_test" if test_or_val == "test" else "y_val"

    test_data = {
        sid: (v[x_key].cpu(), v[y_key].cpu())
        for sid, v in all_splits_dict.items()
    }
    if truncated_feature_set is not None:
        test_data = {sid: (X[:, truncated_feature_set], y) for sid, (X, y) in test_data.items()}

    def _eval_one(rem_rate, add_rate):
        metrics = defaultdict(list)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for split_id, pipe in trained_models.items():
                X, y = test_data[int(split_id)]
                X_noisy = flip_with_fractional_noise(X, add_rate, rem_rate, hard_fn_flag=hard_fn_flag)
                with _suppress_stderr():
                    y_pred = pipe.predict(X_noisy)
                y_prob = pipe.predict_proba(X_noisy)[:, 1]

                metrics["mcc"].append(matthews_corrcoef(y, y_pred))
                metrics["accuracy"].append(accuracy_score(y, y_pred))
                metrics["balanced_accuracy"].append(balanced_accuracy_score(y, y_pred))
                metrics["precision"].append(precision_score(y, y_pred, zero_division=0))
                metrics["recall"].append(recall_score(y, y_pred, zero_division=0))
                metrics["f1"].append(f1_score(y, y_pred, zero_division=0))
                metrics["brier"].append(brier_score_loss(y, y_prob))
                metrics["ece"].append(expected_calibration_error(y, y_prob))

        return rem_rate, add_rate, {k: _agg(v) for k, v in metrics.items()}

    with _suppress_stderr():
        results = Parallel(n_jobs=n_jobs)(
            delayed(_eval_one)(rem, add)
            for rem in fn_rates
            for add in fp_rates
        )

    return {(rem, add): m for rem, add, m in results}


def eval_trained_models_on_noisy_data_classif_and_regress(
    trained_models: dict,
    all_splits_dict: dict,
    hard_fn_flag: bool = False,
    min_fp: float = 0.0,
    min_fn: float = 0.0,
    max_fp: float = 0.2,
    max_fn: float = 1.0,
    n_jobs: int = -1,
    truncated_feature_set=None,
    test_or_val: str = "val",
) -> dict:
    """
    Evaluate mixture-of-experts models (classifier + quantile regressors) under noise.

    Each entry in `trained_models` must be a 7-tuple:
        (classifier, reg_low, reg_high,
         reg_low_q10, reg_low_q90, reg_high_q10, reg_high_q90)

    Returns
    -------
    dict
        ``{(fn_rate, fp_rate): {metric: (mean, std), ...}}``
        Metrics include classification (mcc, accuracy, balanced_accuracy,
        precision, recall, f1, brier, ece), regression (rmse, r2), and
        interval calibration (coverage_low/high/final, mqce_low/high/final,
        interval_width).
    """
    fp_rates = _filter_grid(_FP_GRID, min_fp, max_fp)
    fn_rates = _filter_grid(_FN_GRID, min_fn, max_fn)
    noise_std = 0.3
    x_key = "X_test" if test_or_val == "test" else "X_val"
    y_key = "y_test" if test_or_val == "test" else "y_val"

    # Pre-cache tensors and OGT range labels per split
    split_cache = {}
    for split_id, models in trained_models.items():
        X = all_splits_dict[split_id][x_key]
        y = all_splits_dict[split_id][y_key]
        if truncated_feature_set is not None:
            X = X[:, truncated_feature_set]
        range_ids = np.vectorize({"low": 0, "high": 1}.get)(label_ogt_range(y))
        split_cache[split_id] = (X, y, range_ids, models)

    def _coverage(y_true, y_lo, y_hi) -> float:
        return float(np.mean((y_true >= y_lo) & (y_true <= y_hi)))

    def _mqce(y_true, y_q10, y_q90) -> float:
        return float(np.mean([
            abs(np.mean(y_true <= y_q10) - 0.1),
            abs(np.mean(y_true <= y_q90) - 0.9),
        ]))

    def _eval_one(rem_rate, add_rate):
        m = defaultdict(list)

        for _, (X, y, range_ids, models) in split_cache.items():
            clf, reg_lo, reg_hi, reg_lo_q10, reg_lo_q90, reg_hi_q10, reg_hi_q90 = models

            X_noisy = flip_with_fractional_noise(X, add_rate, rem_rate, noise_std, hard_fn_flag=True)
            y_np = _to_numpy(y)

            # Classification
            y_pred  = clf.predict(X_noisy)
            y_proba = clf.predict_proba(X_noisy)
            p_lo, p_hi = y_proba[:, 0], y_proba[:, 1]

            m["mcc"].append(matthews_corrcoef(range_ids, y_pred))
            m["accuracy"].append(accuracy_score(range_ids, y_pred))
            m["balanced_accuracy"].append(balanced_accuracy_score(range_ids, y_pred))
            m["precision"].append(precision_score(range_ids, y_pred, zero_division=0))
            m["recall"].append(recall_score(range_ids, y_pred, zero_division=0))
            m["f1"].append(f1_score(range_ids, y_pred, zero_division=0))
            m["brier"].append(brier_score_loss(range_ids, y_proba[:, 1]))
            m["ece"].append(expected_calibration_error(range_ids, y_proba[:, 1]))

            # Point-estimate regression
            final_pred = p_lo * reg_lo.predict(X_noisy) + p_hi * reg_hi.predict(X_noisy)
            m["rmse"].append(np.sqrt(mean_squared_error(y_np, final_pred)))
            m["r2"].append(r2_score(y_np, final_pred))

            # Quantile calibration — low group
            lo_mask = range_ids == 0
            if lo_mask.sum() > 0:
                y_lo = y_np[lo_mask]
                q10 = reg_lo_q10.predict(X_noisy[lo_mask])
                q90 = reg_lo_q90.predict(X_noisy[lo_mask])
                m["coverage_low"].append(_coverage(y_lo, q10, q90))
                m["mqce_low"].append(_mqce(y_lo, q10, q90))

            # Quantile calibration — high group
            hi_mask = range_ids == 1
            if hi_mask.sum() > 0:
                y_hi = y_np[hi_mask]
                q10 = reg_hi_q10.predict(X_noisy[hi_mask])
                q90 = reg_hi_q90.predict(X_noisy[hi_mask])
                m["coverage_high"].append(_coverage(y_hi, q10, q90))
                m["mqce_high"].append(_mqce(y_hi, q10, q90))

            # End-to-end combined interval
            f_lo = p_lo * reg_lo_q10.predict(X_noisy) + p_hi * reg_hi_q10.predict(X_noisy)
            f_hi = p_lo * reg_lo_q90.predict(X_noisy) + p_hi * reg_hi_q90.predict(X_noisy)
            m["coverage_final"].append(_coverage(y_np, f_lo, f_hi))
            m["mqce_final"].append(_mqce(y_np, f_lo, f_hi))
            m["interval_width"].append(float(np.mean(f_hi - f_lo)))

        return rem_rate, add_rate, {k: _agg(v) for k, v in m.items()}

    results = Parallel(n_jobs=n_jobs)(
        delayed(_eval_one)(rem, add)
        for rem in fn_rates
        for add in fp_rates
    )
    return {(rem, add): metrics for rem, add, metrics in results}


def read_and_evaluate_models_for_x_and_sigma(
    trained_models_dir: str,
    x_noisy_samples: int,
    noise_type: str,
    metric: str,
    all_splits_dict: dict,
    output_dir: str,
    clean_test_flag: bool = True,
    add_rate: float = None,
    rem_rate: float = None,
    noise_std: float = 0.3,
    hard_fn_flag: bool = None,
) -> dict:
    """
    Load serialised models trained at each (FP mean, FN mean) noise level and
    evaluate them on the validation split.

    Returns
    -------
    dict
        ``{(fp_mean, fn_mean): (mean, std)}`` for the chosen `metric`.
    """
    rate_pairs = [(fp, fn) for fp in _FP_MEAN_GRID for fn in _FN_MEAN_GRID]
    noise_increase_accuracy = {}

    for fp_mean, fn_mean in tqdm(rate_pairs, desc="Processing noise rates"):
        filename = (
            f"trained_models_fp_{fp_mean}_fn_{fn_mean}"
            f"_noise_type_{noise_type}_x_{x_noisy_samples}.pkl"
        )
        filepath = os.path.join(output_dir, trained_models_dir, filename)

        metric_keys = ["mcc", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "brier"]
        metrics_accum = defaultdict(list)

        if os.path.exists(filepath):
            loaded_models = joblib.load(filepath)
            for split_id, pipe in loaded_models.items():
                X = all_splits_dict[split_id]["X_val"]
                y = all_splits_dict[split_id]["y_val"]

                if clean_test_flag:
                    X_eval = X
                else:
                    X_eval = flip_with_fractional_noise(
                        X.cpu(), add_rate, rem_rate, noise_std, hard_fn_flag=hard_fn_flag
                    )

                y_pred = pipe.predict(X_eval)
                y_prob = pipe.predict_proba(X_eval)[:, 1]
                y_cpu  = y.cpu()

                metrics_accum["mcc"].append(matthews_corrcoef(y_cpu, y_pred))
                metrics_accum["accuracy"].append(accuracy_score(y_cpu, y_pred))
                metrics_accum["balanced_accuracy"].append(balanced_accuracy_score(y_cpu, y_pred))
                metrics_accum["precision"].append(precision_score(y_cpu, y_pred, zero_division=0))
                metrics_accum["recall"].append(recall_score(y_cpu, y_pred, zero_division=0))
                metrics_accum["f1"].append(f1_score(y_cpu, y_pred, zero_division=0))
                metrics_accum["brier"].append(brier_score_loss(y_cpu, y_prob))

            scores = {k: [np.mean(v), np.std(v)] for k, v in metrics_accum.items()}
        else:
            scores = {k: [None, None] for k in metric_keys}

        noise_increase_accuracy[(fp_mean, fn_mean)] = scores

    return {k: v[metric] for k, v in noise_increase_accuracy.items()}


def read_and_evaluate_models_for_x_and_sigma_regress(
    trained_models_dir: str,
    x_noisy_samples: int,
    noise_type: str,
    metric: str,
    all_splits_dict: dict,
    output_directory: str,
    clean_test_flag: bool = True,
    add_rate: float = None,
    rem_rate: float = None,
    noise_std: float = 0.3,
    hard_fn_flag: bool = None,
) -> dict:
    """
    Load serialised mixture-of-experts models and evaluate them on the
    validation split. Each stored model is a (classifier, reg_low, reg_high)
    triple.

    Returns
    -------
    dict
        ``{(fp_mean, fn_mean): (mean, std)}`` for the chosen `metric`.
    """
    rate_pairs = [(fp, fn) for fp in _FP_MEAN_GRID for fn in _FN_MEAN_GRID]
    noise_increase_accuracy = {}

    for fp_mean, fn_mean in tqdm(rate_pairs, desc="Processing noise rates"):
        filename = (
            f"trained_models_fp_{fp_mean}_fn_{fn_mean}"
            f"_noise_type_{noise_type}_x_{x_noisy_samples}.pkl"
        )
        filepath = os.path.join(output_directory, trained_models_dir, filename)

        metric_keys = ["mcc", "accuracy", "balanced_accuracy", "precision",
                       "recall", "f1", "brier", "rmse", "r2"]
        metrics_accum = defaultdict(list)

        if os.path.exists(filepath):
            loaded_models = joblib.load(filepath)
            for split_id, model_tuple in loaded_models.items():
                classifier, reg_low, reg_high = model_tuple
                X = all_splits_dict[split_id]["X_val"]
                y = all_splits_dict[split_id]["y_val"]

                range_ids = np.vectorize({"low": 0, "high": 1}.get)(label_ogt_range(y))

                X_eval = X if clean_test_flag else flip_with_fractional_noise(
                    X.cpu(), add_rate, rem_rate, noise_std, hard_fn_flag=hard_fn_flag
                )

                clf_pred  = classifier.predict(X_eval)
                clf_probs = classifier.predict_proba(X_eval)
                pred_low  = reg_low.predict(X_eval)
                pred_high = reg_high.predict(X_eval)
                final_pred = clf_probs[:, 0] * pred_low + clf_probs[:, 1] * pred_high

                y_np = _to_numpy(y)
                metrics_accum["mcc"].append(matthews_corrcoef(range_ids, clf_pred))
                metrics_accum["accuracy"].append(accuracy_score(range_ids, clf_pred))
                metrics_accum["balanced_accuracy"].append(balanced_accuracy_score(range_ids, clf_pred))
                metrics_accum["precision"].append(precision_score(range_ids, clf_pred, zero_division=0))
                metrics_accum["recall"].append(recall_score(range_ids, clf_pred, zero_division=0))
                metrics_accum["f1"].append(f1_score(range_ids, clf_pred, zero_division=0))
                metrics_accum["brier"].append(brier_score_loss(range_ids, clf_probs[:, 1]))
                metrics_accum["rmse"].append(np.sqrt(mean_squared_error(y_np, final_pred)))
                metrics_accum["r2"].append(r2_score(y_np, final_pred))

            scores = {k: [np.mean(v), np.std(v)] for k, v in metrics_accum.items()}
        else:
            scores = {k: [None, None] for k in metric_keys}

        noise_increase_accuracy[(fp_mean, fn_mean)] = scores

    return {k: v[metric] for k, v in noise_increase_accuracy.items()}


# Surface-integral analysis

def fp_fn_surface_integral(
    metric: str,
    min_fp: float, max_fp: float,
    min_fn: float, max_fn: float,
    cog_remov_add_accuracies: dict,
) -> float:
    """
    Compute the 2-D trapezoidal integral of a metric surface over a
    (FN rate, FP rate) grid.

    Parameters
    ----------
    metric : str
        Key into each ``cog_remov_add_accuracies`` entry dict.
    min_fp, max_fp, min_fn, max_fn : float
        Integration bounds (inclusive).
    cog_remov_add_accuracies : dict
        ``{(fn_rate, fp_rate): {metric: (mean, std), ...}}``.

    Returns
    -------
    float
        2-D integral approximated by repeated trapezoidal rule.
    """
    fn_space = sorted({fn for fn, fp in cog_remov_add_accuracies if min_fn <= fn <= max_fn})
    fp_space = sorted({fp for fn, fp in cog_remov_add_accuracies if min_fp <= fp <= max_fp})

    Z = np.array([
        [cog_remov_add_accuracies[(fn, fp)][metric][0] for fn in fn_space]
        for fp in fp_space
    ])
    return float(np.trapezoid(np.trapezoid(Z, x=fn_space, axis=1), x=fp_space))


def fp_curve_areas_one_model(
    metric: str,
    max_fp: float, max_fn: float,
    cog_remov_add_accuracies: dict,
) -> list[float]:
    """
    For each FP rate ≤ max_fp, compute the area under the metric-vs-FN-rate
    curve (trapezoidal rule).
    """
    fn_space = sorted({fn for fn, fp in cog_remov_add_accuracies if fn <= max_fn})
    fp_curves: dict[float, list] = defaultdict(list)
    for (fn, fp) in cog_remov_add_accuracies:
        if fp <= max_fp and fn <= max_fn:
            fp_curves[fp].append(cog_remov_add_accuracies[(fn, fp)][metric][0])

    return [float(np.trapezoid(fp_curves[fp], fn_space)) for fp in fp_curves]


# Plotting

def plot_one_accur_measure(ax, accuracy_measure: str, cog_remov_add_accuracies: dict,
                           alpha: float = 1.0, fontsize: int = 13) -> None:
    """
    Plot metric mean ± std vs. FN rate, one line per FP rate.

    Parameters
    ----------
    ax : matplotlib Axes
    accuracy_measure : str
        Metric key to plot.
    cog_remov_add_accuracies : dict
        ``{(fn_rate, fp_rate): {metric: (mean, std), ...}}``.
    """
    one_measure = {k: v[accuracy_measure] for k, v in cog_remov_add_accuracies.items()}

    fn_rates = sorted({k[0] for k in one_measure})
    fp_rates = sorted({k[1] for k in one_measure})

    colors = [plt.cm.tab10(i / max(len(fn_rates) - 1, 1)) for i in range(len(fn_rates))]

    for i, fp in enumerate(fp_rates):
        pts = [(fn, *one_measure[(fn, fp)]) for fn in fn_rates if (fn, fp) in one_measure]
        if not pts:
            continue
        fns, means, stds = zip(*pts)
        ax.errorbar(fns, means, yerr=stds, label=fr"$r_{{FP}}={fp}$",
                    marker="o", capsize=3, linestyle="-", color=colors[i], alpha=alpha)

    ax.set_xlabel(r"$r_{FN}$", fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)


def plot_model_groups(noise_accuracy: dict, ax, vmin: float = 0.0, vmax: float = 1.0,
                      cmap: str = "coolwarm", value: str = "Mean") -> None:
    """Heatmap of mean ± std metric values over a (FP, FN) noise grid."""
    rows = [[fp, fn, mean, std] for (fp, fn), (mean, std) in noise_accuracy.items()]
    df = pd.DataFrame(rows, columns=["FP", "FN", "Mean", "Std"])
    pivot = df.pivot(index="FP", columns="FN", values=value).astype(float)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap,
                vmin=vmin, vmax=vmax, mask=pivot.isna(), ax=ax, cbar=False)


def plot_model_groups_surf_int(noise_accuracy: dict, ax, vmin: float = 0.0, vmax: float = 1.0,
                               cmap: str = "coolwarm", value: str = "Mean",
                               tot_int_value: float = 0.04) -> None:
    """
    Heatmap of metric values normalised by `tot_int_value` (surface-integral
    comparison variant of :func:`plot_model_groups`).
    """
    rows = [[fp, fn, mean] for (fp, fn), mean in noise_accuracy.items()]
    df = pd.DataFrame(rows, columns=["FP", "FN", "Mean"])
    pivot = df.pivot(index="FP", columns="FN", values=value).astype(float) / tot_int_value
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap,
                vmin=vmin, vmax=vmax, mask=pivot.isna(), ax=ax, cbar=False)