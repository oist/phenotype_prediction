import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
try:
    from scipy.ndimage import uniform_filter1d
except ImportError:
    def uniform_filter1d(a, size):
        return np.convolve(a, np.ones(size) / size, mode="same")

TREE_DIR = "trees"

TREE_FILE_AEROB = f"{TREE_DIR}/aerob_C65_annotated_on_noisy_train_fp_0.1_fn_0.5_noise_type_unif_x_100_ancRootTable.tree"
TREE_FILE_CELLENVEL = f"{TREE_DIR}/diderm_C65_annotated_on_noisy_train_fp_0.1_fn_0.5_noise_type_exp_x_100_ancRootTable.tree"
TREE_FILE_SPORUL = f"{TREE_DIR}/sporul_C65_annotated_on_noisy_train_fp_0.05_fn_1.0_noise_type_exp_x_100_ancRootTable.tree"
TREE_FILE_OGT = f"{TREE_DIR}/ogt_C65_annotated_on_noisy_train_fp_0.1_fn_1.0_noise_type_exp_x_100_ancRootTable.tree"
TREE_FILE_GC = f"{TREE_DIR}/gc_C65_annotated_on_noisy_train_fp_0.5_fn_1.0_noise_type_exp_x_100_ancRootTable.tree"

WINDOW_FRACTION = 10   # window = len(ages) // WINDOW_FRACTION
N_BINS = 400            # number of age bins for OGT / GC plots
N_BINS_GAUSS = 15       # number of age bins for Gaussian evolution plot



TREES = {
    "Oxygen use":  (TREE_FILE_AEROB,    "#d62728"),
    "Cell envelope": (TREE_FILE_CELLENVEL,"#2ca02c"),
    "Sporulation": (TREE_FILE_SPORUL,   "#05343E"),
    "OGT":    (TREE_FILE_OGT,      "#1f77b4"),
    "GC content": (TREE_FILE_GC,     "#ff7f0e"),
}


#  minimal Newick parser 

class Node:
    __slots__ = ["children", "branch_length", "max_proba", "proba_and_uncert",
                 "phenotype_pred", "final_value", "combined_var", "interval_width", "depth"]
    def __init__(self):
        self.children       = []
        self.branch_length  = 0.0
        self.max_proba      = None
        self.proba_and_uncert = None
        self.phenotype_pred = None
        self.final_value    = None
        self.combined_var   = None
        self.interval_width = None
        self.depth          = 0.0


def _skip_annotation(s, pos):
    """Consume a [&...] block; return (text_inside, pos_after)."""
    if pos >= len(s) or s[pos] != "[":
        return None, pos
    depth, start = 0, pos
    while pos < len(s):
        if   s[pos] == "[": depth += 1
        elif s[pos] == "]":
            depth -= 1
            if depth == 0:
                return s[start + 1 : pos], pos + 1
        pos += 1
    return None, pos


def _parse_node(s, pos):
    node = Node()

    while pos < len(s) and s[pos] in " \t\n\r":
        pos += 1

    if pos < len(s) and s[pos] == "(":
        pos += 1                            # opening '('
        while True:
            while pos < len(s) and s[pos] in " \t\n\r":
                pos += 1
            child, pos = _parse_node(s, pos)
            node.children.append(child)
            while pos < len(s) and s[pos] in " \t\n\r":
                pos += 1
            if pos < len(s) and s[pos] == ",":
                pos += 1
            elif pos < len(s) and s[pos] == ")":
                pos += 1
                break
        # optional internal-node label (bootstrap / height number)
        m = re.match(r"[^\s:,)\[;]+", s[pos:])
        if m:
            pos += len(m.group())
    else:
        # leaf: read name up to ':', ',', ')', '[', or end
        m = re.match(r"[^\s:,)\[;]+", s[pos:])
        if m:
            pos += len(m.group())

    # branch length after ':'
    if pos < len(s) and s[pos] == ":":
        pos += 1
        m = re.match(r"[0-9]+\.?[0-9]*(?:[eE][+\-]?[0-9]+)?", s[pos:])
        if m:
            node.branch_length = float(m.group())
            pos += len(m.group())

    # annotation block [&...]
    annot, pos = _skip_annotation(s, pos)
    if annot:
        m2 = re.search(r"max_proba\w*=([\d.]+)", annot)
        if m2:
            node.max_proba = float(m2.group(1))
        m3 = re.search(r"proba_and_uncert\s*=\s*([\d.]+)", annot)
        if m3:
            node.proba_and_uncert = float(m3.group(1))
        # second key=value after !color is the phenotype prediction
        m4 = re.search(r"!color=[^,]+,\s*(\w+)\s*=\s*([^,\]]+)", annot)
        if m4:
            node.phenotype_pred = f"{m4.group(1)}={m4.group(2).strip()}"
        # *_final field (ogt_final, gc_final, ...)
        m5 = re.search(r"\w+_final\s*=\s*([\d.]+)", annot)
        if m5:
            node.final_value = float(m5.group(1))
        m6 = re.search(r"combined_var\s*=\s*([\d.]+)", annot)
        if m6:
            node.combined_var = float(m6.group(1))
        m7 = re.search(r"interval_width\s*=\s*([\d.]+)", annot)
        if m7:
            node.interval_width = float(m7.group(1))

    return node, pos


def parse_tree(nexus_text):
    m = re.search(r"Tree\s+\w+\s*=\s*(.+?)(?:;|\Z)", nexus_text,
                  re.DOTALL | re.IGNORECASE)
    newick = m.group(1).strip()
    root, _ = _parse_node(newick, 0)
    return root


# ── compute node depths (distance from root) ──────────────────────────────

def _assign_depths(node, d=0.0):
    node.depth = d
    for child in node.children:
        _assign_depths(child, d + child.branch_length)


def _max_leaf_depth(node):
    if not node.children:
        return node.depth
    return max(_max_leaf_depth(c) for c in node.children)


def collect_internal_nodes(node, results):
    if node.children and node.max_proba is not None:
        results.append(node)
    for child in node.children:
        collect_internal_nodes(child, results)





def _collect_branches(node, parent, total_depth, results):
    """Collect (parent_age, child_age, combined_var) for every branch."""
    age = total_depth - node.depth
    if parent is not None and node.combined_var is not None:
        parent_age = total_depth - parent.depth
        results.append((parent_age, age, node.combined_var))
    for child in node.children:
        _collect_branches(child, node, total_depth, results)


def _collect_branches_proba(node, parent, total_depth, results):
    """Collect (parent_age, child_age, max_proba) for every annotated branch."""
    if parent is not None and node.max_proba is not None:
        age = total_depth - node.depth
        parent_age = total_depth - parent.depth
        results.append((parent_age, age, node.max_proba))
    for child in node.children:
        _collect_branches_proba(child, node, total_depth, results)


def _collect_branches_interval(node, parent, total_depth, results):
    """Collect (parent_age, child_age, interval_width) for every annotated branch."""
    if parent is not None and node.interval_width is not None:
        age = total_depth - node.depth
        parent_age = total_depth - parent.depth
        results.append((parent_age, age, node.interval_width))
    for child in node.children:
        _collect_branches_interval(child, node, total_depth, results)


def ages_and_probas(tree_file):
    with open(tree_file) as f:
        content = f.read()
    root = parse_tree(content)
    _assign_depths(root)
    total_depth = _max_leaf_depth(root)
    internal = []
    collect_internal_nodes(root, internal)
    ages         = np.array([total_depth - n.depth  for n in internal])
    proba        = np.array([n.max_proba             for n in internal])
    combined_var = np.array([n.combined_var if n.combined_var is not None else np.nan
                             for n in internal])
    branches = []
    _collect_branches(root, None, total_depth, branches)
    order = np.argsort(ages)
    return ages[order], proba[order], combined_var[order], branches, root


TREES_COMBINED = {
    "GC content": (f"{TREE_DIR}/gc_C65_annotated_on_noisy_train_fp_0.5_fn_1.0_noise_type_exp_x_100_ancRootTable.tree",
                   "#ff7f0e"),
    "OGT":        (f"{TREE_DIR}/ogt_C65_annotated_on_noisy_train_fp_0.1_fn_1.0_noise_type_exp_x_100_ancRootTable.tree",
                   "#1f77b4"),
}


#  figure 1: max class probability 

fig1, ax1 = plt.subplots(figsize=(10, 4))

for label, (tree_file, color) in TREES.items():
    with open(tree_file) as f:
        content = f.read()
    root = parse_tree(content)
    _assign_depths(root)
    total_depth = _max_leaf_depth(root)

    proba_branches = []
    _collect_branches_proba(root, None, total_depth, proba_branches)

    all_ages = [a for pa, ca, _ in proba_branches for a in (pa, ca)]
    age_min, age_max = min(all_ages), max(all_ages)
    edges = np.linspace(age_min, age_max, N_BINS + 1)
    centers, means = [], []
    for i in range(N_BINS):
        lo, hi = edges[i], edges[i + 1]
        vals = [mp for pa, ca, mp in proba_branches if ca < hi and pa > lo]
        if not vals:
            continue
        centers.append((lo + hi) / 2)
        means.append(np.mean(vals))

    ax1.plot(centers, means, color=color, linewidth=2, label=label, alpha=0.8)
    print(f"{label}: {len(proba_branches)} branches, {N_BINS} bins")
    final_str = f"  final={root.final_value}" if root.final_value is not None else ""
    print(f"  root  phenotype={root.phenotype_pred}  "
          f"max_proba={root.max_proba}  proba_and_uncert={root.proba_and_uncert}"
          f"{final_str}")

ax1.set_xlabel("Node age (Mya)", fontsize=14)
ax1.set_ylim(0.5, 1.05)
ax1.set_ylabel("Max class probability", fontsize=14)
ax1.invert_xaxis()
ax1.legend(frameon=False)
ax1.tick_params(labelsize=18)
ax1.xaxis.set_major_locator(plt.MultipleLocator(500))
fig1.tight_layout()

for ext in ("pdf", "svg"):
    fig1.savefig(f"all_trees_age_vs_maxproba.{ext}", bbox_inches="tight")
print("Saved all_trees_age_vs_maxproba.pdf/svg")


def branch_binned_mean_std(branches, n_bins):
    """Mean/std of sqrt(combined_var) per bin, counting each branch in every bin it spans."""
    all_ages = [a for p, c, _ in branches for a in (p, c)]
    age_min, age_max = min(all_ages), max(all_ages)
    edges = np.linspace(age_min, age_max, n_bins + 1)
    centers, means, stds = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # branch spans [child_age, parent_age]; overlaps bin if child_age < hi and parent_age > lo
        vals = [np.sqrt(cv) for parent_age, child_age, cv in branches
                if child_age < hi and parent_age > lo]
        if not vals:
            continue
        centers.append((lo + hi) / 2)
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return np.array(centers), np.array(means), np.array(stds)


def plot_combined_var(label, tree_file, color, out, ylabel):
    _, _, _, branches, root = ages_and_probas(tree_file)
    centers, means, stds = branch_binned_mean_std(branches, N_BINS)
    root_age  = max(p for p, _, _ in branches)
    root_uncert = np.sqrt(root.combined_var) if root.combined_var is not None else None

    fig, ax = plt.subplots(figsize=(12, 5))
    if root_uncert is not None:
        all_x = np.concatenate([centers, [root_age]])
        all_y = np.concatenate([means, [root_uncert]])
        ax.plot(all_x, all_y, color=color, linewidth=2, label=label, alpha=0.8)
        ax.scatter([root_age], [root_uncert], color=color, s=60, zorder=5,
                   marker="*", label=f"root ({root_uncert:.2f})")
    else:
        ax.plot(centers, means, color=color, linewidth=2, label=label, alpha=0.8)
    ax.set_xlabel("Node age (Mya)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(label, fontsize=14)
    ax.invert_xaxis()
    ax.tick_params(labelsize=18)
    ax.xaxis.set_major_locator(plt.MultipleLocator(500))
    ax.legend(frameon=False, fontsize=12)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(out.replace(".pdf", f".{ext}"), bbox_inches="tight")
    print(f"Saved {out.replace('.pdf', '.pdf/.svg')}")


def plot_interval_width(label, tree_file, color, out, ylabel):
    with open(tree_file) as f:
        content = f.read()
    root = parse_tree(content)
    _assign_depths(root)
    total_depth = _max_leaf_depth(root)

    branches = []
    _collect_branches_interval(root, None, total_depth, branches)

    all_ages = [a for pa, ca, _ in branches for a in (pa, ca)]
    age_min, age_max = min(all_ages), max(all_ages)
    edges = np.linspace(age_min, age_max, N_BINS + 1)
    centers, means = [], []
    for i in range(N_BINS):
        lo, hi = edges[i], edges[i + 1]
        vals = [v for pa, ca, v in branches if ca < hi and pa > lo]
        if not vals:
            continue
        centers.append((lo + hi) / 2)
        means.append(np.mean(vals))
    centers = np.array(centers)
    means   = np.array(means)

    root_age = total_depth
    root_iw  = root.interval_width

    fig, ax = plt.subplots(figsize=(12, 5))
    if root_iw is not None:
        all_x = np.concatenate([centers, [root_age]])
        all_y = np.concatenate([means,   [root_iw]])
        ax.plot(all_x, all_y, color=color, linewidth=2, label=label, alpha=0.8)
        ax.scatter([root_age], [root_iw], color=color, s=60, zorder=5,
                   marker="*", label=f"root ({root_iw:.2f})")
    else:
        ax.plot(centers, means, color=color, linewidth=2, label=label, alpha=0.8)
    ax.set_xlabel("Node age (Mya)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(label, fontsize=14)
    ax.invert_xaxis()
    ax.tick_params(labelsize=18)
    ax.xaxis.set_major_locator(plt.MultipleLocator(500))
    ax.legend(frameon=False, fontsize=12)
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(out.replace(".pdf", f".{ext}"), bbox_inches="tight")
    print(f"Saved {out.replace('.pdf', '.pdf/.svg')}")


# ── figures 2 & 3: combined_var per phenotype ─────────────────────────────

TREES_COMBINED_YLABELS = {
    "GC content": "Prediction uncertainty (percent)",
    "OGT":        "Prediction uncertainty (°C)",
}

for label, (tree_file, color) in TREES_COMBINED.items():
    ages, _, combined_var, branches, root = ages_and_probas(tree_file)
    root_age = max(p for p, _, _ in branches)
    uncert = np.sqrt(root.combined_var) if root.combined_var is not None else None
    print(f"{label}  root: √combined_var={uncert:.4f}  "
          f"final={root.final_value}  proba_and_uncert={root.proba_and_uncert}")
    # 4 internal nodes with ages closest to (but below) root age
    cv_by_age = [(a, cv) for a, cv in zip(ages, combined_var) if not np.isnan(cv)]
    cv_by_age.sort(key=lambda x: x[0], reverse=True)
    print(f"  4 nearest nodes to root:")
    for age, cv in cv_by_age[:4]:
        print(f"    age={age:.1f}  √combined_var={np.sqrt(cv):.4f}")
    out = f"{label.lower().replace(' ', '_')}_age_vs_combined_var.pdf"  # base name
    plot_combined_var(label, tree_file, color, out, TREES_COMBINED_YLABELS[label])

TREES_INTERVAL_YLABELS = {
    "GC content": "Prediction interval width (percent)",
    "OGT":        "Prediction interval width (°C)",
}

for label, (tree_file, color) in TREES_COMBINED.items():
    out = f"{label.lower().replace(' ', '_')}_age_vs_interval_width.pdf"
    plot_interval_width(label, tree_file, color, out, TREES_INTERVAL_YLABELS[label])

# ── Gaussian evolution plot ───────────────────────────────────────────────

def _collect_leaves(node, results):
    """Collect (combined_var, final_value) for all leaf nodes that have both."""
    if not node.children:
        if node.combined_var is not None and node.final_value is not None:
            results.append((node.combined_var, node.final_value))
    for child in node.children:
        _collect_leaves(child, results)


def _collect_branches_full(node, parent, total_depth, results):
    """Collect (parent_age, child_age, combined_var, final_value) for every branch
    except the root (which is handled separately in plot_gaussian_evolution)."""
    age = total_depth - node.depth
    if parent is not None and node.combined_var is not None and node.final_value is not None:
        parent_age = total_depth - parent.depth
        results.append((parent_age, age, node.combined_var, node.final_value))
    for child in node.children:
        _collect_branches_full(child, node, total_depth, results)


def _gauss_pdf(y, mu, var):
    sigma = np.sqrt(max(var, 1e-9))
    return np.exp(-0.5 * ((y - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def plot_gaussian_evolution(label, tree_file, color, out, ylabel, n_bins=N_BINS_GAUSS):
    with open(tree_file) as f:
        content = f.read()
    root_node = parse_tree(content)
    _assign_depths(root_node)
    total_depth = _max_leaf_depth(root_node)

    branches = []
    _collect_branches_full(root_node, None, total_depth, branches)
    if not branches:
        return

    root_age  = total_depth   # root.depth == 0
    edges     = np.linspace(0, root_age, n_bins + 1)
    bin_width = edges[1] - edges[0]

    # y-axis range: cover root + all branches
    fv_arr = np.array([fv for _, _, _, fv in branches] +
                      ([root_node.final_value] if root_node.final_value else []))
    cv_arr = np.array([cv for _, _, cv, _ in branches] +
                      ([root_node.combined_var] if root_node.combined_var else []))
    y_min  = (fv_arr - 3 * np.sqrt(cv_arr)).min()
    y_max  = (fv_arr + 3 * np.sqrt(cv_arr)).max()
    y_grid = np.linspace(y_min, y_max, 400)

    # regular bins: violin at right (older) edge, covering [0, root_age - bin_width]
    bin_xs, pdfs, bin_means = [], [], []
    for i in range(n_bins - 1):
        lo, hi = edges[i], edges[i + 1]
        items = [(fv, cv) for pa, ca, cv, fv in branches if ca < hi and pa > lo]
        if not items:
            continue
        pdf = sum(_gauss_pdf(y_grid, mu, var) for mu, var in items) / len(items)
        bin_xs.append(hi)
        pdfs.append(pdf)
        bin_means.append(np.mean([fv for fv, _ in items]))

    # root: pure single Gaussian at root_age
    root_pdf  = _gauss_pdf(y_grid, root_node.final_value, root_node.combined_var)
    root_mean = root_node.final_value

    # leaf distribution — one bin-width past age 0
    leaves = []
    _collect_leaves(root_node, leaves)
    leaf_pdf  = None
    leaf_mean = None
    if leaves:
        leaf_pdf  = sum(_gauss_pdf(y_grid, fv, cv) for cv, fv in leaves) / len(leaves)
        leaf_mean = np.mean([fv for _, fv in leaves])

    global_max = max([p.max() for p in pdfs] + [root_pdf.max()] +
                     ([leaf_pdf.max()] if leaf_pdf is not None else []))
    scale = bin_width * 0.75 / global_max

    def _mean_line(ax, x_c, pdf, mean_val):
        idx = np.argmin(np.abs(y_grid - mean_val))
        ax.plot([x_c, x_c + scale * pdf[idx]], [mean_val, mean_val],
                color="red", lw=1.5, zorder=5)

    fig, ax = plt.subplots(figsize=(12, 5))

    # root violin at root_age
    ax.fill_betweenx(y_grid, root_age, root_age + scale * root_pdf, alpha=0.7, color=color)
    ax.plot(root_age + scale * root_pdf, y_grid, color="black", lw=1.0, alpha=0.8)
    ax.axvline(root_age, color="black", lw=0.5, ls="--", alpha=0.4)
    _mean_line(ax, root_age, root_pdf, root_mean)

    for x_c, pdf, mean_val in zip(bin_xs, pdfs, bin_means):
        ax.fill_betweenx(y_grid, x_c, x_c + scale * pdf, alpha=0.5, color=color)
        ax.plot(x_c + scale * pdf, y_grid, color="black", lw=0.5, alpha=0.5)
        ax.axvline(x_c, color="black", lw=0.5, ls="--", alpha=0.4)
        _mean_line(ax, x_c, pdf, mean_val)

    if leaf_pdf is not None:
        leaf_x = 0.0
        ax.fill_betweenx(y_grid, leaf_x, leaf_x + scale * leaf_pdf, alpha=0.7, color=color)
        ax.plot(leaf_x + scale * leaf_pdf, y_grid, color="black", lw=1.0, alpha=0.8)
        ax.axvline(leaf_x, color="black", lw=0.5, ls="--", alpha=0.4)
        _mean_line(ax, leaf_x, leaf_pdf, leaf_mean)
        ax.axhline(leaf_mean, color="grey", lw=1.0, ls="--", alpha=0.7, zorder=1)

    ax.set_xlabel("Node age (Mya)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(label, fontsize=14)
    ax.invert_xaxis()
    ax.tick_params(labelsize=18)
    ax.xaxis.set_major_locator(plt.MultipleLocator(500))
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(out.replace(".pdf", f".{ext}"), bbox_inches="tight")
    print(f"Saved {out.replace('.pdf', '.pdf/.svg')}")


TREES_GAUSS_YLABELS = {
    "GC content": "GC content (%)",
    "OGT":        "Optimal growth temperature (°C)",
}

for label, (tree_file, color) in TREES_COMBINED.items():
    out = f"{label.lower().replace(' ', '_')}_gauss_evolution.pdf"
    plot_gaussian_evolution(label, tree_file, color, out, TREES_GAUSS_YLABELS[label])


# ── boxplot + scatter evolution plot ─────────────────────────────────────────

def plot_boxplot_evolution(label, tree_file, color, out, ylabel, n_bins=N_BINS_GAUSS):
    with open(tree_file) as f:
        content = f.read()
    root_node = parse_tree(content)
    _assign_depths(root_node)
    total_depth = _max_leaf_depth(root_node)

    branches = []
    _collect_branches_full(root_node, None, total_depth, branches)
    if not branches:
        return

    root_age  = total_depth
    edges     = np.linspace(0, root_age, n_bins + 1)
    bin_width = edges[1] - edges[0]
    box_width = bin_width * 0.4

    bin_xs, bin_vals = [], []
    for i in range(n_bins - 1):
        lo, hi = edges[i], edges[i + 1]
        vals = [fv for pa, ca, _, fv in branches if ca < hi and pa > lo]
        if not vals:
            continue
        bin_xs.append(hi)
        bin_vals.append(vals)

    leaves = []
    _collect_leaves(root_node, leaves)
    leaf_vals = [fv for _, fv in leaves] if leaves else None

    fig, ax = plt.subplots(figsize=(12, 5))

    face_rgba = (*mcolors.to_rgb(color), 0.3)
    bp_style = dict(
        vert=True, manage_ticks=False, patch_artist=True,
        boxprops=dict(facecolor=face_rgba, edgecolor="black", linewidth=0.8),
        medianprops=dict(color="red", lw=1.5, zorder=6),
        whiskerprops=dict(color="black", alpha=0.8),
        capprops=dict(color="black", alpha=0.8),
        flierprops=dict(marker=""),
    )

    for x_c, vals in zip(bin_xs, bin_vals):
        ax.boxplot(vals, positions=[x_c], widths=box_width, **bp_style)
        jitter = np.random.uniform(-box_width * 0.3, box_width * 0.3, len(vals))
        ax.scatter(np.full(len(vals), x_c) + jitter, vals,
                   color=color, s=4, alpha=0.3, zorder=3)

    if root_node.final_value is not None:
        ax.plot([root_age - box_width / 2, root_age + box_width / 2],
                [root_node.final_value, root_node.final_value],
                color="red", lw=1.5, zorder=6)

    if leaf_vals:
        ax.boxplot(leaf_vals, positions=[0.0], widths=box_width, **bp_style)
        jitter = np.random.uniform(-box_width * 0.3, box_width * 0.3, len(leaf_vals))
        ax.scatter(np.full(len(leaf_vals), 0.0) + jitter, leaf_vals,
                   color=color, s=4, alpha=0.3, zorder=3)
        ax.axhline(np.median(leaf_vals), color="grey", lw=1.0, ls="--", alpha=0.7, zorder=1)

    ax.set_xlabel("Node age (Mya)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(label, fontsize=14)
    ax.invert_xaxis()
    ax.tick_params(labelsize=18)
    ax.xaxis.set_major_locator(plt.MultipleLocator(500))
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(out.replace(".pdf", f".{ext}"), bbox_inches="tight")
    print(f"Saved {out.replace('.pdf', '.pdf/.svg')}")


for label, (tree_file, color) in TREES_COMBINED.items():
    out = f"{label.lower().replace(' ', '_')}_boxplot_evolution.pdf"
    plot_boxplot_evolution(label, tree_file, color, out, TREES_GAUSS_YLABELS[label])

plt.show()
