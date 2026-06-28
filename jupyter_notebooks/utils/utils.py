"""
Data loading, preprocessing, training helpers, and plotting utilities.
"""

#  Standard library 
import argparse
import logging
import os
import random
from collections import defaultdict

# Third-party 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler
from xgboost import XGBClassifier, XGBRegressor


# Argument parsing helpers

def str2bool(v) -> bool:
    """Convert common string representations of booleans to bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# Data loading

def _warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def _load_tsv(path: str, label: str) -> pd.DataFrame | None:
    """Read a TSV, returning None and printing a warning if empty."""
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        _warn(f"{label} file '{path}' is empty. Skipping.")
        return None
    return df


def _require_accession(df: pd.DataFrame, path: str) -> bool:
    if "accession" not in df.columns:
        _warn(f"'accession' column missing in '{path}'. Skipping.")
        return False
    return True


def _read_taxa(taxa_filename: str | None) -> list | None:
    """Load the last column of a taxa TSV as a list, or return None."""
    if taxa_filename is None:
        return None
    try:
        df = pd.read_csv(taxa_filename, sep="\t")
        if df.empty:
            _warn(f"Taxa file '{taxa_filename}' is empty. Setting taxa_label = None.")
            return None
        return df.iloc[:, -1].tolist()
    except Exception as e:
        _warn(f"Could not read taxa file '{taxa_filename}': {e}")
        return None


def _to_tensor(arr, device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(arr, dtype=dtype).to(device)


#  OGT 

def read_ogt_data(X_filename, y_filename, taxa_filename, device, return_genome_accession=False):
    """Load OGT feature matrix, labels, and optional taxa/accession data."""
    df_x = _load_tsv(X_filename, "X")
    if df_x is None or not _require_accession(df_x, X_filename):
        return None, None, None, None, None

    df_y = _load_tsv(y_filename, "Y")
    if df_y is None or not _require_accession(df_y, y_filename):
        return None, None, None, None, None

    col_names = df_x.columns[1:]  # drop 'accession'
    genome_accession = df_x["accession"] if return_genome_accession else None

    X = df_x.drop(columns=["accession"]).apply(pd.to_numeric, errors="coerce").fillna(0).values
    X_tensor = _to_tensor(X, device)

    y_df = df_y.drop(columns=["accession"])
    if y_df.empty:
        _warn(f"Y file '{y_filename}' has no columns after dropping 'accession'. Skipping.")
        return None, None, None, None, None
    y_tensor = _to_tensor(y_df.values, device)

    taxa = _read_taxa(taxa_filename)
    return X_tensor, y_tensor, col_names, taxa, genome_accession


# ── Diderm ───────────────────────────────────────────────────────────────────

def read_diderm_data(X_filename, y_filename, taxa_filename, device, return_genome_accession=False):
    """Load Diderm/Monoderm classification data."""
    df_x = _load_tsv(X_filename, "X")
    if df_x is None or not _require_accession(df_x, X_filename):
        return None, None, None, None, None
    df_x = df_x.drop_duplicates(subset="accession", keep="first")

    df_y = _load_tsv(y_filename, "Y")
    if df_y is None or not _require_accession(df_y, y_filename):
        return None, None, None, None, None
    df_y = df_y.drop_duplicates(subset="accession", keep="first")

    col_names = df_x.columns[1:]

    merged = pd.merge(df_x, df_y, on="accession", how="inner")
    if merged.empty:
        _warn("Merge between X and Y produced no rows. Skipping.")
        return None, None, None, None, None

    genome_accession = merged["accession"] if return_genome_accession else None

    # Optionally merge taxa
    taxa_label = None
    if taxa_filename is not None:
        taxa_df = pd.read_csv(taxa_filename, sep="\t")
        if taxa_df.empty:
            _warn(f"Taxa file '{taxa_filename}' is empty.")
        else:
            taxa_df = taxa_df.drop_duplicates(subset="accession", keep="first")
            merged = pd.merge(merged, taxa_df, on="accession", how="inner")
            if merged.empty:
                _warn("Merge with taxa produced no rows. Skipping.")
                return None, None, None, None, None
            taxa_label = merged.iloc[:, -1].tolist()
            merged = merged.drop(columns=[merged.columns[-1]])

    X = merged.drop(columns=["annotation", "accession"]).apply(pd.to_numeric, errors="coerce").fillna(0).values
    X_tensor = _to_tensor(X, device)

    y = merged["annotation"].map({"Diderm": 0, "Monoderm": 1})
    y_tensor = _to_tensor(y.values, device)

    return X_tensor, y_tensor, col_names, taxa_label, genome_accession


#  Aerob 

def read_aerob_data(X_filename, y_filename, taxa_filename, device, return_genome_accession=False):
    """Load aerobe/anaerobe classification data."""
    df_x = _load_tsv(X_filename, "X")
    df_y = _load_tsv(y_filename, "Y")
    if df_x is None or df_y is None:
        return None, None, None, None, None
    if not _require_accession(df_x, X_filename) or not _require_accession(df_y, y_filename):
        return None, None, None, None, None

    col_names = df_x.columns[1:]

    merged = pd.merge(df_x, df_y, on="accession", how="inner")
    if merged.empty:
        _warn(f"Merge between {X_filename} and {y_filename} produced no overlapping accessions.")
        return None, None, None, None, None

    if "annotation" not in merged.columns:
        _warn(f"No 'annotation' column found after merge in {y_filename}.")
        return None, None, None, None, None

    genome_accession = merged["accession"] if return_genome_accession else None

    try:
        X = merged.drop(columns=["annotation", "accession"]).apply(pd.to_numeric, errors="coerce").fillna(0).values
    except Exception as e:
        print(f"[ERROR] Failed to extract numeric features from {X_filename}: {e}")
        return None, None, None, None, None

    X_tensor = _to_tensor(X, device)

    y = merged["annotation"].map({"anaerobe": 0, "aerobe": 1})
    if y.isnull().any():
        _warn(f"Some labels could not be mapped to 0/1 in {y_filename}.")
        y = y.fillna(-1)
    y_tensor = _to_tensor(y.values, device)

    taxa = _read_taxa(taxa_filename)
    return X_tensor, y_tensor, col_names, taxa, genome_accession


# Sporulation 

def read_sporulat_data(X_filename, y_filename, taxa_filename, device, return_genome_accession=False):
    """Load sporulation (yes/no) classification data."""
    df_x = _load_tsv(X_filename, "X")
    if df_x is None or not _require_accession(df_x, X_filename):
        return None, None, None, None, None

    df_y = _load_tsv(y_filename, "Y")
    if df_y is None or not _require_accession(df_y, y_filename):
        return None, None, None, None, None

    merged = pd.merge(df_x, df_y, on="accession", how="inner")
    if merged.empty:
        _warn(f"Merge between {X_filename} and {y_filename} produced no overlapping rows. Skipping.")
        return None, None, None, None, None

    if "annotation" not in merged.columns:
        _warn("'annotation' column missing in merged data. Skipping.")
        return None, None, None, None, None

    col_names = df_x.columns[1:]
    genome_accession = merged["accession"] if return_genome_accession else None

    X = merged.drop(columns=["annotation", "accession"]).apply(pd.to_numeric, errors="coerce").fillna(0).values
    X_tensor = _to_tensor(X, device)

    y = merged["annotation"].map({"no": 0, "yes": 1})
    if y.isnull().any():
        _warn(f"Some labels in '{y_filename}' could not be mapped to 0/1.")
        y = y.fillna(-1)
    y_tensor = _to_tensor(y.values, device)

    taxa = _read_taxa(taxa_filename)
    return X_tensor, y_tensor, col_names, taxa, genome_accession


#  Aerob legacy pipeline (Polars-based) 

_GTDB_PATHS = [
    ("data_aerob/bac120_metadata_r202.tsv", "data_aerob/ar122_metadata_r202.tsv"),
    ("../data_aerob/bac120_metadata_r202.tsv", "../data_aerob/ar122_metadata_r202.tsv"),
]
_COG_BLACKLIST = ["COG0411", "COG0459", "COG0564", "COG1344", "COG4177"]
_TARGET_COL = "oxytolerance"
_LABEL_MAP = pl.when(pl.col(_TARGET_COL) == "anaerobe").then(0)\
               .when(pl.col(_TARGET_COL) == "aerobe").then(1)\
               .when(pl.col(_TARGET_COL) == "anaerobic_with_respiration_genes").then(2)\
               .otherwise(None).alias(_TARGET_COL)


def _load_gtdb() -> pl.DataFrame:
    for bac_path, arc_path in _GTDB_PATHS:
        try:
            gtdb = pl.concat([
                pl.read_csv(bac_path, separator="\t"),
                pl.read_csv(arc_path, separator="\t"),
            ])
            return gtdb
        except FileNotFoundError:
            continue
    raise FileNotFoundError("Could not locate GTDB metadata files.")


def read_xy_data(data_filename: str, y_filename: str, remove_noise: bool = True):
    """Load aerob feature data joined with GTDB taxonomy and OGT labels."""
    gtdb = _load_gtdb()
    gtdb = gtdb.filter(pl.col("gtdb_representative") == "t")
    logging.info("Read %d GTDB reps", len(gtdb))

    for rank, idx in (("phylum", 1), ("class", 2), ("order", 3), ("family", 4), ("genus", 5)):
        gtdb = gtdb.with_columns(
            pl.col("gtdb_taxonomy").str.split(";").list.get(idx).alias(rank)
        )

    y = pl.read_csv(y_filename, separator="\t").unique()
    logging.info("Read y: %s", y.shape)
    logging.info("Class counts: %s", y.group_by(_TARGET_COL).agg(pl.len()))

    d = pl.read_csv(data_filename, separator="\t")
    d = d.join(gtdb.select(["accession", "phylum", "class", "order", "family", "genus"]),
               on="accession", how="left")
    d = d.join(y, on="accession", how="inner")

    if remove_noise:
        d = d.filter(pl.col("false_negative_rate") == 0)
        d = d.filter(pl.col("false_positive_rate") == 0)

    print(f"Class counts in data: {d.group_by(_TARGET_COL).agg(pl.len())}")

    exclude = ["accession", _TARGET_COL, "phylum", "class", "order", "family", "genus",
               "false_negative_rate", "false_positive_rate"]
    X = d.select(pl.exclude(exclude)).to_pandas().drop(columns=_COG_BLACKLIST, errors="ignore")
    y_pd = d.select(_LABEL_MAP).to_pandas()

    return d, X, y_pd


def process_aerob_dataset(X_filename: str, y_filename: str, device):
    """Full aerob data pipeline: load, convert to tensor."""
    d3, X_df, y_df = read_xy_data(X_filename, y_filename, remove_noise=True)
    col_names = X_df.columns

    X_tensor = _to_tensor(X_df.values, device)
    y_tensor = _to_tensor(y_df.values, device).squeeze(1)

    return X_tensor, col_names, y_tensor, d3.to_pandas()


def table_row_subsampling(d3: pl.DataFrame):
    """Subsample to balance aerobe/anaerobe classes."""
    X = d3.select(pl.exclude([_TARGET_COL])).to_pandas()
    y = d3.select(_LABEL_MAP).to_pandas()

    idx_aero  = y[y[_TARGET_COL] == 1].index.tolist()
    idx_anaero = y[y[_TARGET_COL] == 0].index.tolist()
    n_aero, n_anaero = len(idx_aero), len(idx_anaero)

    if n_aero > n_anaero:
        print(f"Sub-sampling {n_aero} aerobes to {n_anaero} anaerobes")
        final_idx = random.sample(idx_aero, n_anaero) + idx_anaero
    else:
        print(f"Sub-sampling {n_anaero} anaerobes to {n_aero} aerobes")
        final_idx = idx_aero + random.sample(idx_anaero, n_aero)

    X_sub = X.iloc[final_idx].reset_index(drop=True)
    y_sub = y.iloc[final_idx].reset_index(drop=True)
    n_aero_sub = y_sub[_TARGET_COL].sum()
    print(f"Sub-sampled: {len(y_sub)} total, {n_aero_sub} aerobes, {len(y_sub) - n_aero_sub} anaerobes")
    return X_sub, y_sub


# Evaluation metrics

def evaluate_metrics(y_true, y_pred) -> dict:
    """Standard classification metrics (macro-averaged)."""
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc":       matthews_corrcoef(y_true, y_pred),
    }


def evaluate_metrics_contin_and_bin(
    true_test_labels, predict_test_labels_contin, predict_test_labels_binary, threshold
) -> dict:
    """Combined regression + binary classification metrics."""
    true_bin = (np.asarray(true_test_labels) >= threshold).astype(int)
    return {
        "balanced_accuracy": balanced_accuracy_score(true_bin, predict_test_labels_binary),
        "precision": precision_score(true_bin, predict_test_labels_binary, average="macro", zero_division=0),
        "recall":    recall_score(true_bin, predict_test_labels_binary, average="macro", zero_division=0),
        "f1":        f1_score(true_bin, predict_test_labels_binary, average="macro", zero_division=0),
        "mcc":       matthews_corrcoef(true_bin, predict_test_labels_binary),
        "rmse":      np.sqrt(mean_squared_error(true_test_labels, predict_test_labels_contin)),
        "r2":        r2_score(true_test_labels, predict_test_labels_contin),
    }


# Phylogeny-based prediction

def _climb_to_sibling_leaves(genome_name, nodes, genome_accesion_train, genome_accesion_test):
    """
    Walk up the phylogenetic tree from `genome_name` until a set of sibling
    leaves is found that is fully contained in the training set.

    Returns the sibling leaf set, or an empty set if the genome is not in `nodes`.
    """
    if genome_name not in nodes:
        return set()

    leaves_to_exclude = {genome_name}
    current_node = nodes[genome_name]
    sibling_leaves: set = set()

    while not sibling_leaves or not sibling_leaves.issubset(genome_accesion_train):
        parent = current_node.up
        sibling_leaves = {
            leaf
            for sib in parent.children
            for leaf in sib.get_leaf_names()
            if leaf not in leaves_to_exclude
            and (leaf in genome_accesion_train or leaf in genome_accesion_test)
        }
        leaves_to_exclude.update(sibling_leaves)
        current_node = parent

    return sibling_leaves


def one_split_phylogeny_prediction(
    genome_accesion_train, genome_accesion_test,
    train_labels_dict, test_labels_dict,
    nodes_bact, nodes_arch,
):
    """Nearest-neighbour phylogenetic prediction (binary / rounded mean)."""
    true_labels, pred_labels = [], []

    for genome in genome_accesion_test:
        nodes = nodes_bact if genome in nodes_bact else (nodes_arch if genome in nodes_arch else None)
        if nodes is None:
            continue

        siblings = _climb_to_sibling_leaves(genome, nodes, genome_accesion_train, genome_accesion_test)
        if siblings:
            true_labels.append(test_labels_dict[genome])
            pred_labels.append(round(np.mean([train_labels_dict[s] for s in siblings])))

    return true_labels, pred_labels


def one_split_phylogeny_prediction_contin(
    genome_accesion_train, genome_accesion_test,
    train_labels_dict, test_labels_dict,
    nodes_bact, nodes_arch,
    ogt_threshold: float = 45.0,
):
    """Nearest-neighbour phylogenetic prediction (continuous + binary)."""
    true_labels, pred_contin, pred_binary = [], [], []

    for genome in genome_accesion_test:
        nodes = nodes_bact if genome in nodes_bact else (nodes_arch if genome in nodes_arch else None)
        if nodes is None:
            continue

        siblings = _climb_to_sibling_leaves(genome, nodes, genome_accesion_train, genome_accesion_test)
        if siblings:
            sib_vals = [train_labels_dict[s] for s in siblings]
            true_labels.append(test_labels_dict[genome])
            pred_contin.append(round(np.mean(sib_vals)))
            pred_binary.append(round(np.mean([int(v > ogt_threshold) for v in sib_vals])))

    return true_labels, pred_contin, pred_binary


# XGBoost training helpers

def train_xgboost(X_train, y_train, X_test, y_test, weights=None, model=None, taxonomy_labels=None):
    """
    Train an XGBoost regressor with optional group-aware CV and sample weights.

    Returns
    -------
    y_true_cv, y_pred_cv, y_pred_test, fitted_model
    """
    if model is None:
        model = XGBRegressor(
            reg_alpha=1.0, reg_lambda=1.0, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            n_estimators=300, learning_rate=0.05,
        )

    X_tr = X_train.cpu() if hasattr(X_train, "cpu") else X_train
    y_tr = y_train.cpu() if hasattr(y_train, "cpu") else y_train
    X_te = X_test.cpu() if hasattr(X_test, "cpu") else X_test

    weights_arr = np.asarray(weights) if weights is not None else None

    n_splits = 5
    fold_gen = (
        GroupKFold(n_splits=n_splits).split(X_tr, y_tr, groups=taxonomy_labels)
        if taxonomy_labels is not None
        else KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X_tr, y_tr)
    )
    if taxonomy_labels is not None:
        print("Using taxonomy-aware CV folds")

    y_true_list, y_pred_list = [], []
    for tr_idx, val_idx in fold_gen:
        X_fold_tr, X_fold_val = X_tr[tr_idx], X_tr[val_idx]
        y_fold_tr, y_fold_val = y_tr[tr_idx], y_tr[val_idx]
        w_fold = weights_arr[tr_idx] if weights_arr is not None else None
        model.fit(X_fold_tr, y_fold_tr, sample_weight=w_fold)
        y_true_list.append(y_fold_val)
        y_pred_list.append(model.predict(X_fold_val))

    y_true_cv = np.concatenate(y_true_list)
    y_pred_cv = np.concatenate(y_pred_list)

    y_tr_np = y_tr.numpy() if hasattr(y_tr, "numpy") else np.asarray(y_tr)
    model.fit(X_tr, y_tr_np, sample_weight=weights_arr)
    y_pred_test = model.predict(X_te)

    return y_true_cv, y_pred_cv, y_pred_test, model


def train_xgboost_classification(X_train, y_train, X_test, y_test, num_classes: int = 50):
    """
    Train a multi-class XGBoost classifier with stratified 5-fold CV.

    Returns
    -------
    y_true_cv, y_pred_cv, y_pred_test
    """
    def _make_model():
        return XGBClassifier(
            n_jobs=-1, tree_method="hist",
            objective="multi:softmax", num_class=num_classes, eval_metric="mlogloss",
        )

    X_tr = X_train.cpu() if hasattr(X_train, "cpu") else X_train
    y_tr = y_train.cpu() if hasattr(y_train, "cpu") else y_train
    X_te = X_test.cpu() if hasattr(X_test, "cpu") else X_test

    y_true_list, y_pred_list = [], []
    for tr_idx, val_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_tr, y_tr):
        m = _make_model()
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        y_true_list.append(y_tr[val_idx])
        y_pred_list.append(m.predict(X_tr[val_idx]))

    model = _make_model()
    model.fit(X_tr, y_tr)
    y_pred_test = model.predict(X_te)

    return np.concatenate(y_true_list), np.concatenate(y_pred_list), y_pred_test


def xgboost_accuracy_contin(X_train, X_test, y_train, y_test, sorted_indices, feat_step, feat_removal=False):
    """Sweep over feature subsets and record regression metrics (RMSE, R²)."""
    rmse_test, r2_test, rmse_cv, r2_cv, num_feat_plot = [], [], [], [], []

    for N in range(1, len(sorted_indices), feat_step):
        sel = list(sorted_indices[:N] if not feat_removal else sorted_indices[N:])
        num_feat_plot.append(N)

        y_true_cv, y_pred_cv, y_pred_t, _ = train_xgboost(
            X_train[:, sel], y_train, X_test[:, sel], y_test
        )
        rmse_test.append(np.sqrt(mean_squared_error(y_test, y_pred_t)))
        r2_test.append(r2_score(y_test, y_pred_t))
        rmse_cv.append(np.sqrt(mean_squared_error(y_true_cv, y_pred_cv)))
        r2_cv.append(r2_score(y_true_cv, y_pred_cv))

    return rmse_test, r2_test, rmse_cv, r2_cv, num_feat_plot


def random_feat_removal_curves_ogt(X_train, X_test, y_train, y_test, num_runs, feat_step, feat_removal):
    """Average regression curves over random feature orderings."""
    X_tr = X_train.cpu() if hasattr(X_train, "cpu") else X_train
    X_te = X_test.cpu() if hasattr(X_test, "cpu") else X_test
    tot = X_tr.shape[1]

    all_rmse_test, all_r2_test, all_rmse_cv, all_r2_cv = [], [], [], []
    for _ in range(num_runs):
        perm = np.random.permutation(tot)
        rt, r2t, rc, r2c, num_feat_plot = xgboost_accuracy_contin(
            X_tr, X_te, y_train, y_test, perm, feat_step, feat_removal
        )
        all_rmse_test.append(rt);  all_r2_test.append(r2t)
        all_rmse_cv.append(rc);    all_r2_cv.append(r2c)

    def _stats(arrs):
        a = np.array(arrs)
        return a.mean(axis=0), a.std(axis=0)

    rmse_test_mn, rmse_test_std = _stats(all_rmse_test)
    r2_test_mn,   r2_test_std   = _stats(all_r2_test)
    rmse_cv_mn,   rmse_cv_std   = _stats(all_rmse_cv)
    r2_cv_mn,     r2_cv_std     = _stats(all_r2_cv)

    return (rmse_test_mn, rmse_test_std, r2_test_mn, r2_test_std,
            rmse_cv_mn, rmse_cv_std, r2_cv_mn, r2_cv_std)


# Plotting helpers

def generate_colors_from_colormap(colormap_name: str, N: int):
    """Return a ListedColormap and list of N colours sampled from a named colormap."""
    cmap = plt.cm.get_cmap(colormap_name, N)
    colors = [cmap(i) for i in range(N)]
    return ListedColormap(colors), colors

def tsne_plot(X_train, perplexity, learning_rate, random_seed,
              y_train=None, colors=None, colorbar=False, alpha=0.6, s=10):
    """Scale data, run t-SNE, and scatter-plot the result."""
    X_scaled = MaxAbsScaler().fit_transform(X_train)
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate,
                max_iter=3000, init="pca", random_state=random_seed)
    X_tsne = tsne.fit_transform(X_scaled)
    print(f"t-SNE output shape: {X_tsne.shape}")

    listed_cmap = (
        colors if colors is not None
        else ListedColormap(cm.nipy_spectral(np.linspace(0, 1, len(np.unique(y_train)))))
    )
    if y_train is not None:
        sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train,
                         alpha=alpha, s=s, cmap=listed_cmap, zorder=2)
        if colorbar:
            plt.colorbar(sc)
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=alpha, s=s, zorder=2)

    plt.xlabel("tSNE1"); plt.ylabel("tSNE2"); plt.title("tSNE")


def plot_accuracy_metric(metric, test_accuracy_scores, cv_accuracy_scores,
                         test_accur_arr, test_accur_arr_rem,
                         cv_accur_arr, cv_accur_arr_rem,
                         num_feat, tot_num_feat):
    """Plot add/remove feature-sweep curves against a full-feature baseline."""
    plt.axhline(y=test_accuracy_scores[metric], color="darkred",  linestyle="--", linewidth=1.5, label="baseline test")
    plt.axhline(y=cv_accuracy_scores[metric],   color="darkblue", linestyle="--", linewidth=1.5, label="baseline CV")
    plt.plot(num_feat, [s[metric] for s in test_accur_arr], c="tab:red",  label="test | add")
    plt.plot(num_feat, [s[metric] for s in cv_accur_arr],   c="tab:blue", label="cv | add")
    rem_x = [tot_num_feat - n for n in num_feat]
    plt.plot(rem_x, [s[metric] for s in test_accur_arr_rem], c="tab:red",  alpha=0.5, label="test | remove")
    plt.plot(rem_x, [s[metric] for s in cv_accur_arr_rem],   c="tab:blue", alpha=0.5, label="cv | remove")
    plt.xlabel("number of features added/removed")
    plt.ylabel(metric)


# Noise / matching-probability analysis helpers

_FALSE_POSIT_UNIQ = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def group_matching_probab(file_path: str) -> pd.DataFrame:
    """Compute per-(FN-rate, FP-rate) prediction accuracy and error-type rates."""
    df = pd.read_csv(file_path, delimiter="\t")
    df["prediction_correct"] = df["prediction"] == df["y_actual"]
    df["fp_prediction"] = (df["prediction"] == 1) & (df["y_actual"] == 0)
    df["fn_prediction"] = (df["prediction"] == 0) & (df["y_actual"] == 1)

    grouped = (
        df.groupby(["false_negative_rate", "false_positive_rate"])
        .agg(prediction_correct="mean", fp_prediction="mean", fn_prediction="mean")
        .reset_index()
        .rename(columns={
            "prediction_correct": "matching_probability",
            "fp_prediction": "mean_fp_prediction",
            "fn_prediction": "mean_fn_prediction",
        })
    )
    return grouped


def generate_tables(grouped: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a grouped DataFrame into one sub-table per unique FP rate."""
    cols = ["false_negative_rate", "matching_probability", "mean_fp_prediction", "mean_fn_prediction"]
    return [
        grouped[grouped["false_positive_rate"] == fp][cols]
        for fp in grouped["false_positive_rate"].unique()
    ]


def find_aver_accuracy(table_dict: dict) -> float:
    return float(np.mean([table_dict[k]["matching_probability"].mean() for k in table_dict]))


def find_average_table(csv_files: list[str], result_directory: str) -> dict:
    """Average matching-probability tables across CV folds."""
    tables_by_fp: dict[float, list] = defaultdict(list)
    for csv_file in csv_files:
        grouped = group_matching_probab(os.path.join(result_directory, csv_file))
        for i, table in enumerate(generate_tables(grouped)):
            tables_by_fp[_FALSE_POSIT_UNIQ[i]].append(table)

    averaged = {}
    for fp, tables in tables_by_fp.items():
        combined = pd.concat(tables)
        averaged[fp] = (
            combined.groupby("false_negative_rate", as_index=False)
            .agg({
                "matching_probability": "mean",
                "mean_fp_prediction":   "mean",
                "mean_fn_prediction":   "mean",
            })
        )
    return averaged


def find_accuracies(num_ind_points: int, result_directory: str) -> dict:
    """Print and return accuracy summaries for a given number of inducing points."""
    cv_files   = [f for f in os.listdir(result_directory)
                  if f.endswith(".csv") and "cross_valid" in f and f"indPoints_{num_ind_points}" in f]
    test_files = [f for f in os.listdir(result_directory)
                  if f.endswith(".csv") and "holdout_test" in f and f"indPoints_{num_ind_points}" in f]

    grouped = group_matching_probab(os.path.join(result_directory, test_files[0]))
    print(f"\nHold-out results for {num_ind_points} inducing points:")
    print(f"  Accuracy:          {grouped['matching_probability'].mean():.3f}")
    print(f"  False-positive:    {grouped['mean_fp_prediction'].mean():.3f}")
    print(f"  False-negative:    {grouped['mean_fn_prediction'].mean():.3f}")

    tables_avg = find_average_table(cv_files, result_directory)
    print(f"  CV accuracy:       {find_aver_accuracy(tables_avg):.3f}")
    return tables_avg


def plot_results(column_name: str, num_ind_points: int, fp_to_plot: list, tables_average_folds: dict):
    """Plot a matching-probability metric against gene-removal rate."""
    plt.figure(figsize=(6, 4))
    for i, (fp_val, table) in enumerate(tables_average_folds.items()):
        if _FALSE_POSIT_UNIQ[i] in fp_to_plot:
            plt.scatter(table["false_negative_rate"], table[column_name])
            plt.plot(table["false_negative_rate"], table[column_name],
                     label=f"extra genes rate = {fp_val}")
    plt.xlabel("gene removal rate")
    plt.ylim([0.85, 1])
    plt.ylabel("accuracy")
    plt.legend()
    plt.title(f"{column_name} — SetTransformer with {num_ind_points} inducing points")
    plt.grid(True, zorder=1)
    plt.show()


# Bin-level error statistics

def calculate_aver_std(y_test: np.ndarray, diff: np.ndarray, num_bins: int):
    """
    Compute mean and std of `diff` within equal-width bins of `y_test`.

    Returns bin centres, mean_diff, std_diff (NaN where a bin is empty).
    """
    bins = np.linspace(y_test.min(), y_test.max(), num=num_bins)
    centres = 0.5 * (bins[1:] + bins[:-1])
    bin_idx = np.digitize(y_test, bins) - 1

    mean_diff = np.full(len(centres), np.nan)
    std_diff  = np.full(len(centres), np.nan)
    for i in range(len(centres)):
        mask = bin_idx == i
        if mask.any():
            mean_diff[i] = diff[mask].mean()
            std_diff[i]  = diff[mask].std()

    return centres, mean_diff, std_diff