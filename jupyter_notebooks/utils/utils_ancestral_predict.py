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

def flip_with_fractional_noise_bernoulli(
    X: torch.Tensor,
    fp_rate: float,
    fn_rate: float,
    noise_std: float = 0.3,
    hard_fn_flag: bool = False,
) -> torch.Tensor:
    """
    Apply fractional false-positive / false-negative noise to a feature matrix,
    using independent per-entry Bernoulli draws instead of a fixed count per row.

    For each entry (row, feature):
    - False negatives: if the entry is positive, it is flipped (set to 0 if
      hard_fn_flag=True, else decremented by 1) independently with
      probability `fn_rate`.
    - False positives: EVERY entry (regardless of its current value, whether
      it's an original negative, an original positive that survived FN, or
      a positive just zeroed by FN) is independently incremented by 1 with
      probability `fp_rate`. This matches the original rounding-based
      function's behavior, where `all_idx = torch.nonzero(X[i] > -1)`
      selects all indices unconditionally.

    This gives the same expected per-entry flip probability as
    `flip_with_fractional_noise`, but avoids the rounding/discretization
    artifacts that appear when the number of features (or positives) per row
    is small (e.g. round(fn_rate * len(pos_idx)) collapsing to 0 or jumping
    in large steps). At large n_features/n_positives the two functions are
    statistically equivalent; at small n_features they should now match
    closely at every fn_rate/fp_rate combination, including the fn_rate=1
    boundary.

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

    # False negatives: independent Bernoulli(fn_rate) per positive entry
    pos_mask = X > 0
    fn_draw = torch.rand_like(X_noisy) < fn_rate
    fn_mask = pos_mask & fn_draw
    if hard_fn_flag:
        X_noisy[fn_mask] = 0
    else:
        X_noisy[fn_mask] -= 1

    # False positives: independent Bernoulli(fp_rate) applied to ALL entries,
    fp_draw = torch.rand_like(X_noisy) < fp_rate
    X_noisy[fp_draw] += 1

    return torch.clamp(X_noisy, min=0.0)

def eval_trained_models_on_noisy_data_bernoulli(
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
                X_noisy = flip_with_fractional_noise_bernoulli(X, add_rate, rem_rate, hard_fn_flag=hard_fn_flag)
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
    return_raw: bool = False,
) -> dict:
    """
    Evaluate trained classifiers on a grid of (FN rate, FP rate) noise levels.

    Parameters
    ----------
    ...
    return_raw : bool
        If False, return (mean, std) for each metric across splits.
        If True, return the full list of metric values across splits.

    Returns
    -------
    dict
        If return_raw=False:
            {(fn_rate, fp_rate): {metric: (mean, std), ...}}

        If return_raw=True:
            {(fn_rate, fp_rate): {metric: [value_per_split, ...], ...}}
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
        test_data = {
            sid: (X[:, truncated_feature_set], y)
            for sid, (X, y) in test_data.items()
        }

    def _eval_one(rem_rate, add_rate):
        metrics = defaultdict(list)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            for split_id, pipe in trained_models.items():
                X, y = test_data[int(split_id)]

                X_noisy = flip_with_fractional_noise(
                    X,
                    add_rate,
                    rem_rate,
                    hard_fn_flag=hard_fn_flag
                )

                with _suppress_stderr():
                    y_pred = pipe.predict(X_noisy)

                y_prob = pipe.predict_proba(X_noisy)[:, 1]

                metrics["mcc"].append(
                    matthews_corrcoef(y, y_pred)
                )
                metrics["accuracy"].append(
                    accuracy_score(y, y_pred)
                )
                metrics["balanced_accuracy"].append(
                    balanced_accuracy_score(y, y_pred)
                )
                metrics["precision"].append(
                    precision_score(y, y_pred, zero_division=0)
                )
                metrics["recall"].append(
                    recall_score(y, y_pred, zero_division=0)
                )
                metrics["f1"].append(
                    f1_score(y, y_pred, zero_division=0)
                )
                metrics["brier"].append(
                    brier_score_loss(y, y_prob)
                )
                metrics["ece"].append(
                    expected_calibration_error(y, y_prob)
                )

        if return_raw:
            return rem_rate, add_rate, dict(metrics)
        else:
            return rem_rate, add_rate, {
                k: _agg(v) for k, v in metrics.items()
            }

    with _suppress_stderr():
        results = Parallel(n_jobs=n_jobs)(
            delayed(_eval_one)(rem, add)
            for rem in fn_rates
            for add in fp_rates
        )

    return {
        (rem, add): metrics
        for rem, add, metrics in results
    }


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
    return_raw: bool = False,
) -> dict:
    """
    Evaluate mixture-of-experts models (classifier + quantile regressors) under noise.

    Each entry in `trained_models` must be a 7-tuple:
        (classifier, reg_low, reg_high,
         reg_low_q10, reg_low_q90, reg_high_q10, reg_high_q90)

    Returns
    -------
    dict
        ``{(fn_rate, fp_rate): {metric: (mean, f), ...}}``
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

        if return_raw:
            return rem_rate, add_rate, dict(m)
        else:
            return rem_rate, add_rate, {
                k: _agg(v) for k, v in m.items()
            }

        
       # return rem_rate, add_rate, {k: _agg(v) for k, v in m.items()}

    results = Parallel(n_jobs=n_jobs)(
        delayed(_eval_one)(rem, add)
        for rem in fn_rates
        for add in fp_rates
    )
    return {(rem, add): metrics for rem, add, metrics in results}

def eval_trained_models_on_noisy_data_classif_and_regress_simplif(
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
    return_raw: bool = False,
) -> dict:
    """
    Evaluate mixture-of-experts models (classifier + quantile regressors) under noise.

    Each entry in `trained_models` must be a 7-tuple:
        (classifier, reg_low, reg_high,
         reg_low_q10, reg_low_q90, reg_high_q10, reg_high_q90)

    Returns
    -------
    dict
        ``{(fn_rate, fp_rate): {metric: (mean, f), ...}}``
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

    def _eval_one(rem_rate, add_rate):
        m = defaultdict(list)

        for _, (X, y, range_ids, models) in split_cache.items():
            clf, reg_lo, reg_hi = models  #, reg_lo_q10, reg_lo_q90, reg_hi_q10, reg_hi_q90

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


        if return_raw:
            return rem_rate, add_rate, dict(m)
        else:
            return rem_rate, add_rate, {
                k: _agg(v) for k, v in m.items()
            }

        
       # return rem_rate, add_rate, {k: _agg(v) for k, v in m.items()}

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


def wagner_reconstruction(tree, leaf_states, state_labels=None, gain_cost=1.0, loss_cost=1.0):
    """
    Wagner parsimony reconstruction for a binary character.

    Wagner parsimony assumes:
        0 -> 0 = 0
        1 -> 1 = 0
        0 -> 1 = 1
        1 -> 0 = 1

    Parameters
    ----------
    tree : Bio.Phylo tree
        Tree whose terminal node names are keys in leaf_states.

    leaf_states : dict
        Mapping:
            leaf_name -> 0 or 1

    state_labels : dict, optional
        Mapping from numerical states to labels.

        Example:
            {0: "Diderm", 1: "Monoderm"}

        If None, numerical labels "State_0" and "State_1"
        are used.

    Returns
    -------
    node_annotations : dict
        node -> state label or "Ambiguous"

    node_confidence : dict
        node -> relative parsimony support for the selected state.
        This is not a calibrated statistical probability.

    node_probabilities : dict
        node -> np.array([support_state_0, support_state_1])

        These are score-derived relative supports, not statistical
        probabilities.

    scores : dict
        node -> np.array([
            score_if_state_0,
            score_if_state_1
        ])

    parsimony_score : float
        Minimum number of state changes required by the reconstruction.
    """

    states = [0, 1]

    # Generalized transition costs (Wagner is the gain_cost == loss_cost case)
    cost = np.array([
        [0.0, gain_cost],
        [loss_cost, 0.0]
    ])

    if state_labels is None:
        state_labels = {
            0: "State_0",
            1: "State_1"
        }

    scores = {}

    # ---------------------------------------------------------
    # 1. Bottom-up Wagner score calculation
    # ---------------------------------------------------------

    def calculate_scores(node):

        if node.is_terminal():

            if node.name not in leaf_states:
                raise ValueError(
                    f"Leaf '{node.name}' is not present in leaf_states."
                )

            observed = leaf_states[node.name]

            if observed not in states:
                raise ValueError(
                    f"Invalid state {observed} for leaf "
                    f"'{node.name}'. Expected 0 or 1."
                )

            # Observed state is fixed
            scores[node] = np.array([
                0.0 if observed == 0 else np.inf,
                0.0 if observed == 1 else np.inf
            ])

            return scores[node]

        child_scores = [
            calculate_scores(child)
            for child in node.clades
        ]

        node_score = np.zeros(2)

        for parent_state in states:

            total = 0.0

            for child_score in child_scores:

                total += min(
                    child_score[child_state]
                    + cost[parent_state, child_state]
                    for child_state in states
                )

            node_score[parent_state] = total

        scores[node] = node_score

        return node_score

    root_scores = calculate_scores(tree.root)

    parsimony_score = float(np.min(root_scores))

    # ---------------------------------------------------------
    # 2. Score-derived relative support
    # ---------------------------------------------------------

    node_probabilities = {}

    for node, node_score in scores.items():

        min_score = np.min(node_score)

        weights = np.exp(-(node_score - min_score))

        node_probabilities[node] = (
            weights / np.sum(weights)
        )

    # ---------------------------------------------------------
    # 3. Find optimal states
    # ---------------------------------------------------------

    optimal_states = {}

    def find_optimal_states(node, allowed_parent_states=None):

        if allowed_parent_states is None:

            candidate_scores = scores[node]

        else:

            candidate_scores = np.array([
                scores[node][state]
                + min(
                    cost[parent_state, state]
                    for parent_state in allowed_parent_states
                )
                for state in states
            ])

        min_score = np.min(candidate_scores)

        best_states = [
            state
            for state in states
            if np.isclose(
                candidate_scores[state],
                min_score
            )
        ]

        optimal_states[node] = best_states

        for child in node.clades:
            find_optimal_states(
                child,
                allowed_parent_states=best_states
            )

    find_optimal_states(tree.root)

    # ---------------------------------------------------------
    # 4. Convert numerical states to labels
    # ---------------------------------------------------------

    node_annotations = {}
    node_confidence = {}

    for node, best_states in optimal_states.items():

        if node.is_terminal():

            observed = leaf_states[node.name]

            node_annotations[node] = state_labels[observed]
            node_confidence[node] = 1.0

        elif len(best_states) == 1:

            state = best_states[0]

            node_annotations[node] = state_labels[state]

            node_confidence[node] = (
                node_probabilities[node][state]
            )

        else:

            node_annotations[node] = "Ambiguous"

            node_confidence[node] = (
                max(node_probabilities[node])
            )

    return (
        node_annotations,
        node_confidence,
        node_probabilities,
        scores,
        parsimony_score
    )    

def dollo_reconstruction(tree, leaf_states, state_labels=None,
                          loss_cost=1.0, gain_cost=None):
    """
    Dollo parsimony reconstruction for a binary character, implemented
    as the g -> infinity limit of Wagner parsimony (see
    wagner_reconstruction), run through the identical Sankoff DP.

    Dollo assumes the character originates at most once on the tree
    (a single "gain"), but can be lost independently any number of
    times afterward at low/no cost.

    Parameters
    ----------
    tree, leaf_states, state_labels : see wagner_reconstruction

    loss_cost : float, default 1.0
        Cost of a 1 -> 0 transition ("loss").

    gain_cost : float, optional
        Cost of a 0 -> 1 transition ("gain"). If None, automatically
        set to a value large enough that the optimizer will never
        prefer a second independent gain over any number of losses
        (a safe margin above the maximum possible number of loss
        events on this tree).

    Returns
    -------
    Same 5-tuple as wagner_reconstruction:
    (node_annotations, node_confidence, node_probabilities,
     scores, parsimony_score)

    Notes
    -----
    This uses a large-but-finite gain_cost rather than literal
    infinity, to keep the DP numerically well-behaved. For very deep
    or large trees, verify that increasing gain_cost further does not
    change the reconstruction (i.e. that you're past the plateau where
    the answer has stabilized) -- see the gain/loss cost-ratio scan
    discussed earlier.
    """

    if gain_cost is None:
        n_leaves = tree.count_terminals()
        # More than enough: no reconstruction ever needs more losses
        # than there are leaves, so this guarantees a 2nd gain is
        # never cheaper than any achievable number of losses.
        gain_cost = (n_leaves + 1) * loss_cost * 10

    return wagner_reconstruction(
        tree,
        leaf_states,
        state_labels=state_labels,
        gain_cost=gain_cost,
        loss_cost=loss_cost
    )


import numpy as np


def wagner_reconstruction_continuous11(tree, leaf_states):
    """
    Continuous Wagner parsimony reconstruction.

    The objective is to minimize the total absolute change:

        sum |parent_state - child_state|

    across all branches.

    Internal nodes can take arbitrary continuous values; they
    are NOT restricted to values observed at the leaves.

    Parameters
    ----------
    tree : Bio.Phylo tree
        Tree whose terminal node names are keys in leaf_states.

    leaf_states : dict
        Mapping:
            leaf_name -> continuous numerical value

        Example:
            {
                "leaf1": 20.5,
                "leaf2": 22.1,
                "leaf3": 31.7,
                "leaf4": 29.4
            }

    Returns
    -------
    node_annotations : dict
        node -> reconstructed continuous value

    scores : dict
        node -> minimum parsimony score for the subtree
        conditional on the reconstructed state.

    parsimony_score : float
        Minimum total absolute amount of evolutionary change.

    """

    # ---------------------------------------------------------
    # Check input
    # ---------------------------------------------------------

    if len(leaf_states) == 0:
        raise ValueError("leaf_states is empty.")

    leaf_states = {
        name: float(value)
        for name, value in leaf_states.items()
    }

    # ---------------------------------------------------------
    # 1. Bottom-up reconstruction
    # ---------------------------------------------------------

    # For continuous Wagner parsimony, each node can be represented
    # by an interval of values that are equally optimal.
    #
    # This is the classic Fitch-style interval representation
    # for continuous Wagner parsimony.

    intervals = {}
    scores = {}

    def calculate_interval(node):

        # -----------------------------------------------------
        # Leaf
        # -----------------------------------------------------

        if node.is_terminal():

            if node.name not in leaf_states:
                raise ValueError(
                    f"Leaf '{node.name}' is not present "
                    f"in leaf_states."
                )

            value = leaf_states[node.name]

            intervals[node] = (value, value)
            scores[node] = 0.0

            return intervals[node], scores[node]

        # -----------------------------------------------------
        # Internal node
        # -----------------------------------------------------

        child_results = [
            calculate_interval(child)
            for child in node.clades
        ]

        child_intervals = [
            result[0]
            for result in child_results
        ]

        child_scores = [
            result[1]
            for result in child_results
        ]

        # Find the interval containing values that minimize
        # the sum of distances to the child intervals.

        lower = max(
            interval[0]
            for interval in child_intervals
        )

        upper = min(
            interval[1]
            for interval in child_intervals
        )

        if lower <= upper:
            # Child intervals overlap.
            # Any value in the overlap is optimal.
            node_interval = (lower, upper)

            additional_cost = 0.0

        else:
            # Intervals do not overlap.
            #
            # The minimum cost is the gap between the
            # intervals. Multiple children are handled by
            # summing their contributions.

            sorted_intervals = sorted(child_intervals)

            # Use the median-based representation.
            endpoints = []

            for interval in child_intervals:
                endpoints.extend(interval)

            median = np.median(endpoints)

            node_interval = (median, median)

            additional_cost = sum(
                max(
                    interval[0] - median,
                    0,
                    median - interval[1]
                )
                for interval in child_intervals
            )

        scores[node] = (
            sum(child_scores)
            + additional_cost
        )

        intervals[node] = node_interval

        return intervals[node], scores[node]

    root_interval, parsimony_score = calculate_interval(tree.root)

    # ---------------------------------------------------------
    # 2. Top-down reconstruction
    # ---------------------------------------------------------

    node_annotations = {}

    def reconstruct(node, parent_value=None):

        interval = intervals[node]

        if node.is_terminal():

            value = leaf_states[node.name]

        else:

            lower, upper = interval

            if parent_value is None:

                # At root, choose a value from the optimal interval.
                value = (lower + upper) / 2.0

            else:

                # Choose the value in the optimal interval
                # closest to the parent's reconstructed value.

                value = np.clip(
                    parent_value,
                    lower,
                    upper
                )

        node_annotations[node] = float(value)

        for child in node.clades:
            reconstruct(
                child,
                parent_value=value
            )

    reconstruct(tree.root)

    return (
        node_annotations,
        scores,
        float(parsimony_score),
        intervals
    )   



import numpy as np


def wagner_reconstruction_continuous(tree, leaf_states):
    """
    Continuous Wagner parsimony reconstruction.

    The objective is to minimize the total absolute change:

        sum |parent_state - child_state|

    across all branches.

    Internal nodes can take arbitrary continuous values; they
    are NOT restricted to values observed at the leaves.

    Parameters
    ----------
    tree : Bio.Phylo tree
        Tree whose terminal node names are keys in leaf_states.

    leaf_states : dict
        Mapping:
            leaf_name -> continuous numerical value

        Example:
            {
                "leaf1": 20.5,
                "leaf2": 22.1,
                "leaf3": 31.7,
                "leaf4": 29.4
            }

    Returns
    -------
    node_annotations : dict
        node -> reconstructed continuous value

    scores : dict
        node -> minimum parsimony score for the subtree
        conditional on the reconstructed state.

    parsimony_score : float
        Minimum total absolute amount of evolutionary change.

    intervals : dict
        node -> (lower, upper) tuple giving the full range of
        equally-optimal ancestral values at that node (not just
        a single point).
    """

    # ---------------------------------------------------------
    # Check input
    # ---------------------------------------------------------

    if len(leaf_states) == 0:
        raise ValueError("leaf_states is empty.")

    leaf_states = {
        name: float(value)
        for name, value in leaf_states.items()
    }

    # ---------------------------------------------------------
    # 1. Bottom-up reconstruction
    # ---------------------------------------------------------

    # For continuous Wagner parsimony, each node can be represented
    # by an interval of values that are equally optimal.
    #
    # The optimal interval at an internal node is found via the
    # classic "median of intervals" result: sort all 2n endpoints
    # from the n children's intervals; the optimal region is bounded
    # by the n-th and (n+1)-th smallest endpoints. This subsumes both
    # the "overlapping" and "non-overlapping" cases in one formula.

    intervals = {}
    scores = {}

    def calculate_interval(node):

        # -----------------------------------------------------
        # Leaf
        # -----------------------------------------------------

        if node.is_terminal():

            if node.name not in leaf_states:
                raise ValueError(
                    f"Leaf '{node.name}' is not present "
                    f"in leaf_states."
                )

            value = leaf_states[node.name]

            intervals[node] = (value, value)
            scores[node] = 0.0

            return intervals[node], scores[node]

        # -----------------------------------------------------
        # Internal node
        # -----------------------------------------------------

        child_results = [
            calculate_interval(child)
            for child in node.clades
        ]

        child_intervals = [
            result[0]
            for result in child_results
        ]

        child_scores = [
            result[1]
            for result in child_results
        ]

        n = len(child_intervals)

        endpoints = sorted(
            v
            for lo, hi in child_intervals
            for v in (lo, hi)
        )

        # The optimal interval is bounded by the n-th and (n+1)-th
        # smallest endpoints (0-indexed: endpoints[n-1], endpoints[n]).
        # This is exact: any value in this range achieves the same
        # minimum total distance to all child intervals.
        lower_opt = endpoints[n - 1]
        upper_opt = endpoints[n]

        node_interval = (lower_opt, upper_opt)

        additional_cost = sum(
            max(
                lo - lower_opt,
                0.0,
                lower_opt - hi
            )
            for lo, hi in child_intervals
        )

        scores[node] = (
            sum(child_scores)
            + additional_cost
        )

        intervals[node] = node_interval

        return intervals[node], scores[node]

    root_interval, parsimony_score = calculate_interval(tree.root)

    # ---------------------------------------------------------
    # 2. Top-down reconstruction
    # ---------------------------------------------------------

    node_annotations = {}

    def reconstruct(node, parent_value=None):

        interval = intervals[node]

        if node.is_terminal():

            value = leaf_states[node.name]

        else:

            lower, upper = interval

            if parent_value is None:

                # At root, choose a value from the optimal interval.
                value = (lower + upper) / 2.0

            else:

                # Choose the value in the optimal interval
                # closest to the parent's reconstructed value.

                value = np.clip(
                    parent_value,
                    lower,
                    upper
                )

        node_annotations[node] = float(value)

        for child in node.clades:
            reconstruct(
                child,
                parent_value=value
            )

    reconstruct(tree.root)

    return (
        node_annotations,
        scores,
        float(parsimony_score),
        intervals
    ) 

def norm(name):
    """Remove quotes and GTDB 'RS_' / 'GB_' prefixes."""
    name = str(name).strip("'\"")
    return name[3:] if name[:3] in ("RS_", "GB_") else name

def acc_key(acc):
    """Accession without prefix or version: 'RS_GCF_000005845.2' -> 'GCF_000005845'."""
    return norm(acc).split(".")[0]

def read_X(filename):
    df = pd.read_csv(filename, sep="\t")
    return torch.tensor(df.drop(columns=["accession"]).to_numpy(dtype=np.float32))

def read_y(filename, label_map):
    ann = pd.read_csv(filename, sep="\t")["annotation"]
    if ann.dtype == object:  # text labels -> 0/1
        ann = ann.map(label_map)
        if ann.isna().any():
            raise ValueError(f"Labels not in label_map in {filename}")
    return torch.tensor(ann.astype(int).to_numpy(), dtype=torch.long)

# def read_y_contin(filename):
#     ann = pd.read_csv(filename, sep="\t")["annotation"]
#     return torch.tensor(ann.to_numpy(), dtype=torch.long) 

def read_y_contin(filename):
    ann = pd.read_csv(filename, sep="\t")["annotation"]

    # Convert nonnumeric values to NaN
    ann = pd.to_numeric(ann, errors="coerce")

    # If the file contains no valid annotations, return an empty tensor
    ann = ann.dropna()

    if ann.empty:
        return torch.empty(0, dtype=torch.float32)

    return torch.tensor(
        ann.to_numpy(dtype=np.float32),
        dtype=torch.float32
    )
    

def read_taxa(filename, tax_level):
    if filename is None:
        return None
    return pd.read_csv(filename, sep="\t")[tax_level].tolist()

def norm(name):
    """Remove GTDB 'RS_' / 'GB_' prefixes so accessions match the tree tip names."""
    name = str(name).strip("'\"")
    return name[3:] if name[:3] in ("RS_", "GB_") else name

def read_acc(filename):
    return pd.read_csv(filename, sep="\t")["accession"].map(norm).tolist() 

def place(acc, ranks, tax_by_acc, taxon_to_tips, acc_key, tree, leaf_by_name):
    """Returns (clade, rank) for a genome, or (None, None) if even its domain isn't in the tree."""
    _clade_cache = {}
    tax = tax_by_acc.get(acc_key(acc))
    if tax is None:
        return None, None
    for rank, taxon in reversed(list(zip(ranks, tax))):     # species -> genus -> ... -> domain
        tips = taxon_to_tips.get((rank, taxon))
        if tips:
            key = (rank, taxon)
            if key not in _clade_cache:
                _clade_cache[key] = tree.common_ancestor([leaf_by_name[t] for t in tips])
            return _clade_cache[key], rank
    return None, None


def _rank_ok(rank, coarsest_rank, rank_inx):
    """True if the placement rank is at least as fine as coarsest_rank."""
    if coarsest_rank is None:
        return True
    return rank is not None and rank_inx[rank] >= rank_inx[coarsest_rank]

TIME_UNIT_FACTOR = 1
def nearest_train_times(train_ids, test_ids, rank_inx, ranks, tax_by_acc, taxon_to_tips, acc_key, tree, leaf_by_name, age, parent, coarsest_rank=None,
                        filter_train=True, return_excluded=False):
    """Returns (distances in Ma, number of test genomes not placed).
    Distance = 2 x age of the lowest node joining the test genome's position
    with the position of any training genome.

    coarsest_rank : e.g. "family" -> only genomes placed at family level or
        finer (family, genus, species) are used. None = no limit.
    filter_train : also apply the limit to training genomes.
    return_excluded : if True, also return the number of test genomes dropped
        by the rank limit (placed, but too coarsely).
    """
    train_nodes = set()
    TIME_UNIT_FACTOR = 1
    for acc in train_ids:
        n, rank = place(acc, ranks, tax_by_acc, taxon_to_tips, acc_key, tree, leaf_by_name)
        if n is None:
            continue
        if filter_train and not _rank_ok(rank, coarsest_rank, rank_inx):
            continue
        train_nodes.add(id(n))

    has_train = {}
    for clade in tree.find_clades(order="postorder"):
        has_train[id(clade)] = (id(clade) in train_nodes) or \
                               any(has_train[id(c)] for c in clade.clades)

    times, missing, excluded = [], 0, 0
    for acc in test_ids:
        node, rank = place(acc, ranks, tax_by_acc, taxon_to_tips, acc_key, tree, leaf_by_name)
        if node is None:
            missing += 1
            continue
        if not _rank_ok(rank, coarsest_rank, rank_inx):
            excluded += 1
            continue
        while node is not None and not has_train[id(node)]:
            node = parent.get(id(node))
        if node is not None:
            times.append(2 * age[id(node)] * TIME_UNIT_FACTOR)

    if return_excluded:
        return np.array(times), missing, excluded
    return np.array(times), missing


# ---- 6. Placement check: at which rank genomes land on the tree ----
# def placement_report(split_dict):
#     all_acc = list(split_dict["acc_train"]) + list(split_dict["acc_test"])
#     ranks = pd.Series([place(a, ranks)[1] for a in all_acc]).fillna("not placed")
#     print(ranks.value_counts().reindex(RANKS[::-1] + ["not placed"]).dropna().astype(int))

def read_acc_for_split(y_filename, taxa_filename, tax_level, label_map, all_acc, phylum_of):
    """Accessions for one subset of a split.
    Text annot files -> read directly. Phylum (torch annot files) -> rebuild from the taxa file."""
    if tax_level != "phylum":
        return read_acc(y_filename)

    with open(taxa_filename) as f:
        lines = [l.strip() for l in f if l.strip()]

    # Case 1: taxa file has 'accession<TAB>phylum' rows -> take the accessions
    if "\t" in lines[0]:
        rows = [l.split("\t") for l in lines]
        if rows[0][0] == "accession":
            rows = rows[1:]
        return [norm(r[0]) for r in rows]

    # Case 2: taxa file has only phylum names -> all genomes belonging to those phyla
    strip = lambda s: s.split("__")[-1]            # 'p__Firmicutes' -> 'Firmicutes'
    groups = {strip(g) for g in lines} - {"phylum"}
    return [a for a in all_acc if strip(phylum_of.get(a, "")) in groups]    




import os
from collections import defaultdict

import torch
import numpy as np


def read_splits(tax_level, data_dir, num_splits_to_read, label_map, all_acc, phylum_of):
    all_splits_dict = defaultdict(int)

    def file_exists_and_nonempty(filename):
        return (
            filename is not None
            and os.path.isfile(filename)
            and os.path.getsize(filename) > 0
        )

    def empty_like_data():
        return torch.empty(0)

    def empty_like_labels():
        return torch.empty(0)

    for split_id in range(num_splits_to_read):
        # print(f"split_id = {split_id}")

        # -------------------------------------------------------
        # Read train data
        # -------------------------------------------------------
        data_filename_train = (
            f"../{data_dir}/input_data_train_val_test/{tax_level}/"
            f"train_data_{tax_level}_split_{split_id}"
        )
        y_filename_train = (
            f"../{data_dir}/input_data_train_val_test/{tax_level}/"
            f"train_annot_{tax_level}_split_{split_id}"
        )
        taxa_names_filename_train = (
            f"../{data_dir}/input_data_train_val_test/{tax_level}/"
            f"train_taxa_{tax_level}_split_{split_id}"
            if tax_level != "random" else None
        )

        if tax_level == "phylum":
            X_val_train = torch.load(data_filename_train)
            y_label_train = torch.load(y_filename_train)
            with open(taxa_names_filename_train, "r") as f:
                taxa_group_names_train = [line.strip() for line in f]
        else:
            X_val_train = read_X(data_filename_train)
            y_label_train = (
                read_y_contin(y_filename_train)
                if label_map is None
                else read_y(y_filename_train, label_map)
            )
            taxa_group_names_train = read_taxa(
                taxa_names_filename_train, tax_level
            )

        # -------------------------------------------------------
        # Read validation data, if all required files are valid
        # -------------------------------------------------------
        data_filename_val = (
            f"../{data_dir}/input_data_train_val_test/{tax_level}/"
            f"val_data_{tax_level}_split_{split_id}"
        )
        y_filename_val = (
            f"../{data_dir}/input_data_train_val_test/{tax_level}/"
            f"val_annot_{tax_level}_split_{split_id}"
        )
        taxa_names_filename_val = (
            f"../{data_dir}/input_data_train_val_test/{tax_level}/"
            f"val_taxa_{tax_level}_split_{split_id}"
            if tax_level != "random" else None
        )

        val_files_valid = (
            file_exists_and_nonempty(data_filename_val)
            and file_exists_and_nonempty(y_filename_val)
            and (
                tax_level == "random"
                or file_exists_and_nonempty(taxa_names_filename_val)
            )
        )

        if val_files_valid:
            try:
                if tax_level == "phylum":
                    X_val_val = torch.load(data_filename_val)
                    y_label_val = torch.load(y_filename_val)
                    with open(taxa_names_filename_val, "r") as f:
                        taxa_group_names_val = [
                            line.strip() for line in f
                        ]
                else:
                    X_val_val = read_X(data_filename_val)
                    y_label_val = (
                        read_y_contin(y_filename_val)
                        if label_map is None
                        else read_y(y_filename_val, label_map)
                    )
                    taxa_group_names_val = read_taxa(
                        taxa_names_filename_val, tax_level
                    )

                # Treat successfully loaded but empty data as missing.
                if (
                    X_val_val is None
                    or y_label_val is None
                    or len(X_val_val) == 0
                    or len(y_label_val) == 0
                ):
                    val_files_valid = False

            except (EOFError, OSError, ValueError, RuntimeError) as e:
                print(
                    f"  Validation data unavailable for split {split_id}: {e}"
                )
                val_files_valid = False

        if not val_files_valid:
            print(
                f"  Missing or empty validation data for split {split_id}; "
                "using empty validation fields."
            )
            X_val_val = empty_like_data()
            y_label_val = empty_like_labels()
            taxa_group_names_val = []

        # -------------------------------------------------------
        # Read test data
        # -------------------------------------------------------
        data_filename_test = (
            f"../{data_dir}/input_data_train_val_test/{tax_level}/"
            f"test_data_{tax_level}_split_{split_id}"
        )
        y_filename_test = (
            f"../{data_dir}/input_data_train_val_test/{tax_level}/"
            f"test_annot_{tax_level}_split_{split_id}"
        )
        taxa_names_filename_test = (
            f"../{data_dir}/input_data_train_val_test/{tax_level}/"
            f"test_taxa_{tax_level}_split_{split_id}"
            if tax_level != "random" else None
        )

        if tax_level == "phylum":
            X_val_test = torch.load(data_filename_test)
            y_label_test = torch.load(y_filename_test)
            with open(taxa_names_filename_test, "r") as f:
                taxa_group_names_test = [line.strip() for line in f]
        else:
            X_val_test = read_X(data_filename_test)
            y_label_test = (
                read_y_contin(y_filename_test)
                if label_map is None
                else read_y(y_filename_test, label_map)
            )
            taxa_group_names_test = read_taxa(
                taxa_names_filename_test, tax_level
            )

        if tax_level == "random":
            taxa_group_names_train = None
            taxa_group_names_test = None
            taxa_group_names_val = None if val_files_valid else []

        # -------------------------------------------------------
        # Store split if train and test are valid
        # -------------------------------------------------------
        if (
            X_val_train is not None
            and y_label_train is not None
            and X_val_test is not None
            and y_label_test is not None
        ):
            curr_split_dict = defaultdict(str)

            curr_split_dict["X_train"] = X_val_train
            curr_split_dict["y_train"] = y_label_train
            curr_split_dict["acc_train"] = read_acc_for_split(
                y_filename_train,
                taxa_names_filename_train,
                tax_level,
                label_map,
                all_acc,
                phylum_of,
            )
            curr_split_dict["taxa_group_names_train"] = taxa_group_names_train

            curr_split_dict["X_test"] = X_val_test
            curr_split_dict["y_test"] = y_label_test
            curr_split_dict["acc_test"] = read_acc_for_split(
                y_filename_test,
                taxa_names_filename_test,
                tax_level,
                label_map,
                all_acc,
                phylum_of,
            )
            curr_split_dict["taxa_group_names_test"] = taxa_group_names_test

            curr_split_dict["X_val"] = X_val_val
            curr_split_dict["y_val"] = y_label_val
            curr_split_dict["taxa_group_names_val"] = taxa_group_names_val

            if val_files_valid:
                curr_split_dict["acc_val"] = read_acc_for_split(
                    y_filename_val,
                    taxa_names_filename_val,
                    tax_level,
                    label_map,
                    all_acc,
                    phylum_of,
                )
            else:
                curr_split_dict["acc_val"] = []

            all_splits_dict[split_id] = curr_split_dict

    return all_splits_dict
def rolling_mean_std_curve(x, y, window, step=None, min_points=1):
    """
    Rolling (sliding-window) mean and std of y as a function of x.

    Returns
    -------
    mean_x, mean_y, std_y : np.ndarray
        Mean x, mean y, and std of y within each window.
        std_y is 0 for windows with a single point (ddof=0).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if step is None:
        step = window / 2

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    centers = np.arange(x.min(), x.max() + step, step)

    lo = np.searchsorted(x, centers - window / 2, side="left")
    hi = np.searchsorted(x, centers + window / 2, side="left")

    cs_x  = np.concatenate([[0], np.cumsum(x)])
    cs_y  = np.concatenate([[0], np.cumsum(y)])
    cs_y2 = np.concatenate([[0], np.cumsum(y**2)])

    counts = hi - lo
    valid = counts >= min_points

    n = counts[valid]
    mean_x = (cs_x[hi] - cs_x[lo])[valid] / n
    mean_y = (cs_y[hi] - cs_y[lo])[valid] / n

    # Var = E[y^2] - (E[y])^2, computed from cumulative sums in O(1) per window
    mean_y2 = (cs_y2[hi] - cs_y2[lo])[valid] / n
    var_y = np.maximum(mean_y2 - mean_y**2, 0)  # guard tiny negative from float error
    std_y = np.sqrt(var_y)

    return mean_x, mean_y, std_y   

# def nearest_train_times(train_ids, test_ids, ranks):
#     """Returns (distances in Ma, number of test genomes not placed).
#     Distance = 2 x age of the lowest node joining the test genome's position
#     with the position of any training genome."""
#     train_nodes = {id(n) for n, _ in map(place, train_ids) if n is not None}

#     has_train = {}
#     for clade in tree.find_clades(order="postorder"):
#         has_train[id(clade)] = (id(clade) in train_nodes) or \
#                                any(has_train[id(c)] for c in clade.clades)

#     times, missing = [], 0
#     for acc in test_ids:
#         node, _ = place(acc, ranks)
#         if node is None:
#             missing += 1
#             continue
#         while node is not None and not has_train[id(node)]:
#             node = parent.get(id(node))
#         if node is not None:
#             times.append(2 * age[id(node)] * TIME_UNIT_FACTOR)
#     return np.array(times), missing


# ---- 6. Placement check: at which rank genomes land on the tree ----
def placement_report(split_dict, ranks, tax_by_acc, taxon_to_tips, acc_key, tree, leaf_by_name):
    all_acc = list(split_dict["acc_train"]) + list(split_dict["acc_test"])
    ranks = pd.Series([place(a, ranks, tax_by_acc, taxon_to_tips, acc_key, tree, leaf_by_name)[1] for a in all_acc]).fillna("not placed")
    print(ranks.value_counts().reindex(RANKS[::-1] + ["not placed"]).dropna().astype(int))    