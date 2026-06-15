"""Export supplementary cleaning figures for beta-lactamase in LaTeX style.

This script recreates the figure set produced around alignment/tree cleaning in
`utils/alignmentAndTreeSetUp.py`, but saves publication-ready PDF figures.

Key behavior requested by workflow:
- Uses beta-lactamase defaults from `final_betaLac/`.
- Stores numeric intermediates in `temp/`.
- Stores figures in `paper_figures_v3/`.
- PCA fit is done on the reweighted alignment, and all other alignments are
  projected in that same PCA basis.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

if "cgi" not in sys.modules:
    # Python 3.13 removed cgi; ete3 still imports it.
    import types

    cgi_mod = types.ModuleType("cgi")
    cgi_mod.escape = html.escape
    sys.modules["cgi"] = cgi_mod

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from ete3 import Tree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Ensure repository root is first on sys.path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.alignmentAndTreeSetUp import cleanAlignmentAndTree
from utils.toolsForTreesAndMSAs import read_fasta1, read_fasta2


# -----------------------------
# Styling
# -----------------------------

def configure_matplotlib(use_tex: bool = True) -> None:
    mpl.rcParams.update(
        {
                "figure.figsize": (9.0, 5.8),
            "figure.dpi": 150,
            "savefig.dpi": 300,
                "font.size": 13,
                "axes.labelsize": 16,
                "axes.titlesize": 16,
                "legend.fontsize": 12,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "font.family": "serif",
            "text.usetex": use_tex,
        }
    )


# -----------------------------
# Paths / config
# -----------------------------

@dataclass
class BetaLacPaths:
    root: Path
    full_alignment: Path
    reweighted_alignment: Path
    cleaned_alignment: Path
    collapsed_alignment: Path
    clean_tree: Path
    clean_tree_midpoint: Path
    collapsed_tree: Path


def default_paths(project_root: Path) -> BetaLacPaths:
    base = project_root / "final_betaLac"
    inter = base / "intermediate_stuff"
    return BetaLacPaths(
        root=project_root,
        full_alignment=base / "PF13354_noinsert_max19gaps_nodupl_noclose_BetaLact.faa",
        reweighted_alignment=base / "PF13354_reweighted.fa",
        cleaned_alignment=inter / "betaLac_cleaned.fasta",
        collapsed_alignment=base / "betaLac_collapsed.fasta",
        clean_tree=inter / "betaLactree_fromcleaned.nwk",
        clean_tree_midpoint=inter / "betaLactree_fromcleaned_midpointrooted.nwk",
        collapsed_tree=inter / "betaLactree_collapsed_midpointrooted.nwk",
    )


# -----------------------------
# Core computations
# -----------------------------

def _hamming_with_gaps(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a != b))


def _hamming_no_gaps(a: np.ndarray, b: np.ndarray) -> float:
    mask = (a != 0) & (b != 0)
    if not np.any(mask):
        return 0.0
    return float(np.mean(a[mask] != b[mask]))


def sibling_hamming_vs_tree_distance(tree_path: Path, msa_path: Path, nogaps: bool) -> np.ndarray:
    tree = Tree(str(tree_path))
    sequences, name_to_index, _ = read_fasta2(str(msa_path))

    out: List[Tuple[float, float]] = []
    for leaf in tree.iter_leaves():
        if leaf.name not in name_to_index:
            continue
        for sibling in leaf.get_sisters():
            if not sibling.is_leaf() or sibling.name not in name_to_index:
                continue
            d_tree = tree.get_distance(leaf, sibling)
            if d_tree <= 0:
                continue
            s1 = np.asarray(sequences[name_to_index[leaf.name]], dtype=int)
            s2 = np.asarray(sequences[name_to_index[sibling.name]], dtype=int)
            d_ham = _hamming_no_gaps(s1, s2) if nogaps else _hamming_with_gaps(s1, s2)
            out.append((d_tree, d_ham))
    return np.asarray(out, dtype=float)


def child_parent_branch_lengths(tree_path: Path) -> np.ndarray:
    tree = Tree(str(tree_path))
    vals = []
    for leaf in tree.iter_leaves():
        if leaf.up is not None:
            vals.append(float(leaf.get_distance(leaf.up)))
    return np.asarray(vals, dtype=float)


def node_depths(tree_path: Path) -> np.ndarray:
    tree = Tree(str(tree_path))
    root = tree.get_tree_root()
    vals = [float(root.get_distance(node, topology_only=False)) for node in tree.traverse()]
    return np.asarray(vals, dtype=float)


def branch_lengths(tree_path: Path) -> np.ndarray:
    tree = Tree(str(tree_path))
    vals = [float(node.dist) for node in tree.traverse("levelorder")]
    return np.asarray(vals, dtype=float)


def pairwise_hamming_histogram(
    msa: np.ndarray,
    seq_len: int,
    pair_sample_limit: int = 1_500_000,
    seed: int = 1,
) -> np.ndarray:
    n = msa.shape[0]
    total_pairs = n * (n - 1) // 2
    hist = np.zeros(seq_len + 1, dtype=np.float64)

    rng = np.random.default_rng(seed)
    if total_pairs <= pair_sample_limit:
        for i in range(n):
            d = np.sum(msa[i + 1 :] != msa[i], axis=1)
            binc = np.bincount(d, minlength=seq_len + 1)
            hist[: len(binc)] += binc
        return hist

    # Subsample random pairs for large MSAs.
    idx_i = rng.integers(0, n, size=pair_sample_limit)
    idx_j = rng.integers(0, n, size=pair_sample_limit)
    neq = idx_i != idx_j
    idx_i = idx_i[neq]
    idx_j = idx_j[neq]
    d = np.sum(msa[idx_i] != msa[idx_j], axis=1)
    binc = np.bincount(d, minlength=seq_len + 1)
    hist[: len(binc)] = binc
    return hist


def one_hot_21(msa: np.ndarray) -> np.ndarray:
    n, l = msa.shape
    eye = np.eye(21, dtype=np.float32)
    return eye[msa].reshape(n, l * 21)


def sample_rows(arr: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if arr.shape[0] <= limit:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=limit, replace=False)
    return arr[idx]


def fit_reference_pca(
    reference_msa: np.ndarray,
    max_ref_sequences: int = 8000,
    seed: int = 1,
) -> Tuple[StandardScaler, PCA, np.ndarray]:
    ref = sample_rows(reference_msa, max_ref_sequences, seed)
    ref_oh = one_hot_21(ref)
    scaler = StandardScaler(with_mean=True, with_std=True)
    ref_scaled = scaler.fit_transform(ref_oh)
    pca = PCA(n_components=2, random_state=seed)
    ref_proj = pca.fit_transform(ref_scaled)
    return scaler, pca, ref_proj


def project_msa(msa: np.ndarray, scaler: StandardScaler, pca: PCA, max_sequences: int, seed: int) -> np.ndarray:
    msa_s = sample_rows(msa, max_sequences, seed)
    oh = one_hot_21(msa_s)
    return pca.transform(scaler.transform(oh))


# -----------------------------
# Plot helpers
# -----------------------------

def save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def add_large_panel_legend(
    ax: plt.Axes,
    fontsize: int,
    ncol: int = 1,
    loc: str = "best",
) -> None:
    """Add a large legend with smart placement and a readable foreground box."""
    leg = ax.legend(
        loc=loc,
        frameon=True,
        fontsize=fontsize,
        ncol=ncol,
        borderpad=0.55,
        handletextpad=0.6,
        labelspacing=0.45,
    )
    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.92)
    frame.set_edgecolor("#444444")
    frame.set_linewidth(0.8)
    leg.set_zorder(200)


def plot_hist_logy(values: np.ndarray, title: str, xlabel: str, out_path: Path, bins: int = 100) -> None:
    fig, ax = plt.subplots()
    ax.hist(values, bins=bins, edgecolor="black", alpha=0.8)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    save_fig(fig, out_path)


def plot_scatter(xy: np.ndarray, title: str, xlabel: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots()
    if xy.size > 0:
        ax.scatter(xy[:, 0], xy[:, 1], s=12, alpha=0.55)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_fig(fig, out_path)


def plot_two_hist_overlay(a: np.ndarray, b: np.ndarray, labels: Tuple[str, str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots()
    bins = np.logspace(-10, 1, 50)
    ax.hist(a, bins=bins, alpha=0.55, label=labels[0])
    ax.hist(b, bins=bins, alpha=0.55, label=labels[1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Branch length")
    ax.set_ylabel("Count")
    ax.set_title(title)
    add_large_panel_legend(ax, fontsize=16)
    save_fig(fig, out_path)


def plot_depth_overlay(a: np.ndarray, b: np.ndarray, labels: Tuple[str, str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.hist(a, bins=100, alpha=0.55, edgecolor="black", label=labels[0])
    ax.hist(b, bins=100, alpha=0.55, edgecolor="black", label=labels[1])
    ax.set_xlabel("Distance to root")
    ax.set_ylabel("Count")
    ax.set_title(title)
    add_large_panel_legend(ax, fontsize=16)
    save_fig(fig, out_path)


def plot_pairwise_histograms(
    hist_a: np.ndarray,
    hist_b: np.ndarray,
    labels: Tuple[str, str],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(hist_a))
    width = 0.42
    ax.bar(x - width / 2, hist_a, width=width, alpha=0.8, label=labels[0])
    ax.bar(x + width / 2, hist_b, width=width, alpha=0.8, label=labels[1])
    ax.set_yscale("log")
    ax.set_xlabel("Pairwise Hamming distance")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    add_large_panel_legend(ax, fontsize=16)
    save_fig(fig, out_path)


def plot_reweighted_pca(
    ref_proj: np.ndarray,
    projected: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    colors = {
        "Full": "#1f77b4",
        "Cleaned": "#ff7f0e",
        "Collapsed": "#2ca02c",
    }
    for label, points in projected.items():
        ax.scatter(points[:, 0], points[:, 1], s=10, alpha=0.45, c=colors.get(label, None), label=label)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    add_large_panel_legend(ax, fontsize=16)
    save_fig(fig, out_path)


def plot_full_panel(
    lengths_cp: np.ndarray,
    dist_gap: np.ndarray,
    depth_clean: np.ndarray,
    depth_mid: np.ndarray,
    br_clean: np.ndarray,
    br_collapsed: np.ndarray,
    hist_full: np.ndarray,
    hist_coll: np.ndarray,
    ref_proj: np.ndarray,
    proj_full: np.ndarray,
    proj_clean: np.ndarray,
    proj_coll: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(17.5, 16.0))
    axs = axes.ravel()
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.07, top=0.97, wspace=0.22, hspace=0.34)

    label_fs = 27
    tick_fs = 21
    legend_fs = 22
    panel_letter_fs = 30

    # A
    ax = axs[0]
    ax.hist(lengths_cp, bins=100, alpha=0.80, edgecolor="black")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t_{\mathrm{leaf,parent}}$", fontsize=label_fs)
    ax.set_ylabel("Count", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: "" if np.isclose(y, 1e4) else mpl.ticker.LogFormatterMathtext()(y))
    )

    # B
    ax = axs[1]
    if dist_gap.size > 0:
        ax.scatter(dist_gap[:, 0], dist_gap[:, 1], s=20, alpha=0.55)
    ax.set_xlabel(r"$t$", fontsize=label_fs)
    ax.set_ylabel(r"$d_{\mathrm{H}}$", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)

    # C (was D)
    ax = axs[2]
    bins = np.logspace(-10, 1, 50)
    ax.hist(br_clean, bins=bins, alpha=0.55, label="FastTree tree")
    ax.hist(br_collapsed, bins=bins, alpha=0.55, label="Final tree")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Branch length", fontsize=label_fs)
    ax.set_ylabel("Count", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    add_large_panel_legend(ax, fontsize=legend_fs, loc="upper center")

    # D (was E)
    ax = axs[3]
    x = np.arange(len(hist_full))
    width = 0.42
    ax.bar(x - width / 2, hist_full, width=width, alpha=0.8, label="Raw")
    ax.bar(x + width / 2, hist_coll, width=width, alpha=0.8, label="Final")
    ax.set_yscale("log")
    ax.set_xlabel("Pairwise Hamming distance", fontsize=label_fs)
    ax.set_ylabel("Frequency", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    add_large_panel_legend(ax, fontsize=legend_fs, loc="upper left")

    # E (was C, moved to penultimate)
    ax = axs[4]
    ax.hist(depth_clean, bins=100, alpha=0.55, edgecolor="black", label="Tree")
    ax.hist(depth_mid, bins=100, alpha=0.55, edgecolor="black", label="Midpoint-rooted tree")
    ax.set_xlabel("Distance to root on the tree", fontsize=label_fs)
    ax.set_ylabel("Count", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    add_large_panel_legend(ax, fontsize=legend_fs, loc="upper right")

    # F
    ax = axs[5]
    ax.scatter(proj_full[:, 0], proj_full[:, 1], s=16, alpha=0.45, c="#1f77b4", label="Raw")
    ax.scatter(proj_coll[:, 0], proj_coll[:, 1], s=16, alpha=0.45, c="#E69F00", label="Final")
    ax.set_xlabel("PC1", fontsize=label_fs)
    ax.set_ylabel("PC2", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    add_large_panel_legend(ax, fontsize=legend_fs, loc="upper right")

    letters = [r"$\mathbf{A}$", r"$\mathbf{B}$", r"$\mathbf{C}$", r"$\mathbf{D}$", r"$\mathbf{E}$", r"$\mathbf{F}$"]
    for idx, letter in enumerate(letters):
        axs[idx].text(
            -0.10,
            1.05,
            letter,
            transform=axs[idx].transAxes,
            fontsize=panel_letter_fs,
            va="top",
            ha="left",
        )

    save_fig(fig, out_path)


def build_full_panel_legend_text() -> str:
    return (
        "A  t_leaf,parent on the clean tree\n"
        "   tree: final_betaLac/intermediate_stuff/betaLactree_fromcleaned.nwk\n"
        "   data: final_betaLac/intermediate_stuff/betaLac_cleaned.fasta\n\n"
        "B  d_H versus t for sibling leaves\n"
        "   tree: same clean tree\n"
        "   data: same cleaned alignment\n\n"
        "C  Branch-length distributions\n"
        "   tree: clean tree versus collapsed midpoint-rooted tree\n"
        "   data: tree-only analysis\n\n"
        "D  Raw versus collapsed pairwise d_H\n"
        "   data: final_betaLac/PF13354_noinsert_max19gaps_nodupl_noclose_BetaLact.faa\n"
        "        and final_betaLac/betaLac_collapsed.fasta\n\n"
        "E  Distance to root on the tree\n"
        "   tree: clean tree and midpoint-rooted clean tree\n"
        "   data: cleaned alignment\n\n"
        "F  PCA in the reweighted basis\n"
        "   basis: final_betaLac/PF13354_reweighted.fa\n"
        "   projected: raw, cleaned, collapsed alignments"
    )


# -----------------------------
# Driver
# -----------------------------

def run_export(
    project_root: Path,
    out_fig_dir: Path,
    temp_dir: Path,
    rerun_cleaning: bool,
    use_tex: bool,
    max_pca_ref: int,
    max_pca_proj: int,
    pair_sample_limit: int,
) -> None:
    configure_matplotlib(use_tex=use_tex)
    paths = default_paths(project_root)

    temp_dir.mkdir(parents=True, exist_ok=True)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    if rerun_cleaning:
        rerun_base = temp_dir / "rerun_cleaning"
        rerun_align = rerun_base / "alignments"
        rerun_tree = rerun_base / "trees"
        rerun_align.mkdir(parents=True, exist_ok=True)
        rerun_tree.mkdir(parents=True, exist_ok=True)

        cleanAlignmentAndTree(
            full_alignment=str(paths.full_alignment),
            tree_folder=str(rerun_tree) + "/",
            alignment_folder=str(rerun_align) + "/",
            family_name="betaLac",
            prune=False,
            save_folder=str(out_fig_dir) + "/",
        )

        paths.cleaned_alignment = rerun_align / "betaLac_cleaned.fasta"
        paths.collapsed_alignment = rerun_align / "betaLac_collapsed.fasta"
        paths.clean_tree = rerun_tree / "betaLactree_fromcleaned.nwk"
        paths.clean_tree_midpoint = rerun_tree / "betaLactree_fromcleaned_midpointrooted.nwk"
        paths.collapsed_tree = rerun_tree / "betaLactree_collapsed_midpointrooted.nwk"

    required = [
        paths.full_alignment,
        paths.reweighted_alignment,
        paths.cleaned_alignment,
        paths.collapsed_alignment,
        paths.clean_tree,
        paths.clean_tree_midpoint,
        paths.collapsed_tree,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    # Load MSAs.
    full_msa = np.asarray(read_fasta1(str(paths.full_alignment)), dtype=np.int16)
    cleaned_msa = np.asarray(read_fasta1(str(paths.cleaned_alignment)), dtype=np.int16)
    collapsed_msa = np.asarray(read_fasta1(str(paths.collapsed_alignment)), dtype=np.int16)
    reweighted_msa = np.asarray(read_fasta1(str(paths.reweighted_alignment)), dtype=np.int16)

    # 1) Clean tree: child-parent lengths.
    lengths_cp = child_parent_branch_lengths(paths.clean_tree)
    plot_hist_logy(
        lengths_cp,
        "Clean tree: leaf-to-parent branch lengths",
        "Branch length",
        out_fig_dir / "SuppClean_01_child_parent_lengths.pdf",
        bins=300,
    )

    # 2) Clean tree: sibling hamming vs tree distance (with/without gaps).
    dist_gap = sibling_hamming_vs_tree_distance(paths.clean_tree, paths.cleaned_alignment, nogaps=False)
    plot_scatter(
        dist_gap,
        "Clean tree: sibling Hamming vs tree distance (with gaps)",
        "Tree distance",
        "Hamming distance",
        out_fig_dir / "SuppClean_02_hamming_vs_tree_with_gaps.pdf",
    )

    dist_nogap = sibling_hamming_vs_tree_distance(paths.clean_tree, paths.cleaned_alignment, nogaps=True)
    plot_scatter(
        dist_nogap,
        "Clean tree: sibling Hamming vs tree distance (no gaps)",
        "Tree distance",
        "Hamming distance (no gaps)",
        out_fig_dir / "SuppClean_03_hamming_vs_tree_no_gaps.pdf",
    )

    # 3) Root-depth overlay (clean vs midpoint-rerooted clean).
    depth_clean = node_depths(paths.clean_tree)
    depth_mid = node_depths(paths.clean_tree_midpoint)
    plot_depth_overlay(
        depth_clean,
        depth_mid,
        ("Clean", "Midpoint-rerooted clean"),
        "Leaf/node depth distribution",
        out_fig_dir / "SuppClean_04_depth_clean_vs_midpoint.pdf",
    )

    # 4) Branch-length overlay (clean vs collapsed).
    br_clean = branch_lengths(paths.clean_tree)
    br_collapsed = branch_lengths(paths.collapsed_tree)
    plot_two_hist_overlay(
        br_clean,
        br_collapsed,
        ("Clean tree", "Collapsed tree"),
        "Branch-length distributions",
        out_fig_dir / "SuppClean_05_branch_lengths_clean_vs_collapsed.pdf",
    )

    # 5) Pairwise Hamming histograms.
    seq_len = int(full_msa.shape[1])
    hist_full = pairwise_hamming_histogram(full_msa, seq_len, pair_sample_limit=pair_sample_limit, seed=1)
    hist_coll = pairwise_hamming_histogram(collapsed_msa, seq_len, pair_sample_limit=pair_sample_limit, seed=3)

    plot_pairwise_histograms(
        hist_full,
        hist_coll,
        ("Raw", "Collapsed"),
        "Pairwise Hamming: raw vs collapsed",
        out_fig_dir / "SuppClean_06_pairwise_raw_vs_collapsed.pdf",
    )

    # 6) PCA in reweighted basis.
    scaler, pca, ref_proj = fit_reference_pca(reweighted_msa, max_ref_sequences=max_pca_ref, seed=7)
    proj_full = project_msa(full_msa, scaler, pca, max_sequences=max_pca_proj, seed=8)
    proj_clean = project_msa(cleaned_msa, scaler, pca, max_sequences=max_pca_proj, seed=9)
    proj_coll = project_msa(collapsed_msa, scaler, pca, max_sequences=max_pca_proj, seed=10)

    plot_reweighted_pca(
        ref_proj,
        projected={"Full": proj_full, "Cleaned": proj_clean, "Collapsed": proj_coll},
        out_path=out_fig_dir / "SuppClean_08_pca_reweighted_basis.pdf",
    )

    # Full panel (A-G): excludes no-gap Hamming plot by request.
    plot_full_panel(
        lengths_cp=lengths_cp,
        dist_gap=dist_gap,
        depth_clean=depth_clean,
        depth_mid=depth_mid,
        br_clean=br_clean,
        br_collapsed=br_collapsed,
        hist_full=hist_full,
        hist_coll=hist_coll,
        ref_proj=ref_proj,
        proj_full=proj_full,
        proj_clean=proj_clean,
        proj_coll=proj_coll,
        out_path=out_fig_dir / "SuppClean_full_panel.pdf",
    )

    legend_text = build_full_panel_legend_text()
    legend_path = out_fig_dir / "SuppClean_full_panel_legend.txt"
    with open(legend_path, "w", encoding="utf-8") as f:
        f.write(legend_text)
        f.write("\n")

    summary = {
        "paths": {
            "full_alignment": str(paths.full_alignment),
            "reweighted_alignment": str(paths.reweighted_alignment),
            "cleaned_alignment": str(paths.cleaned_alignment),
            "collapsed_alignment": str(paths.collapsed_alignment),
            "clean_tree": str(paths.clean_tree),
            "clean_tree_midpoint": str(paths.clean_tree_midpoint),
            "collapsed_tree": str(paths.collapsed_tree),
        },
        "counts": {
            "full_n": int(full_msa.shape[0]),
            "cleaned_n": int(cleaned_msa.shape[0]),
            "collapsed_n": int(collapsed_msa.shape[0]),
            "reweighted_n": int(reweighted_msa.shape[0]),
            "seq_len": seq_len,
        },
        "pca": {
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        },
        "outputs_dir": str(out_fig_dir),
        "full_panel": str(out_fig_dir / "SuppClean_full_panel.pdf"),
        "full_panel_legend": str(legend_path),
    }
    with open(temp_dir / "export_supp_figures_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Supplementary cleaning figures exported.")
    print(f"Figures: {out_fig_dir}")
    print(f"Summary: {temp_dir / 'export_supp_figures_summary.json'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export beta-lactamase cleaning supplementary figures.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root directory (default: repository root).",
    )
    parser.add_argument(
        "--out-fig-dir",
        type=Path,
        default=Path("paper_figures_v3") / "supp_cleaning",
        help="Directory where PDF figures are written.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("temp") / "supp_cleaning",
        help="Directory for intermediate numeric outputs and summary.",
    )
    parser.add_argument(
        "--rerun-cleaning",
        action="store_true",
        help="Rerun cleanAlignmentAndTree and use outputs from temp.",
    )
    parser.add_argument(
        "--use-tex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LaTeX text rendering via matplotlib text.usetex=True (default: enabled).",
    )
    parser.add_argument(
        "--max-pca-ref",
        type=int,
        default=8000,
        help="Max number of sequences for PCA fit on reweighted alignment.",
    )
    parser.add_argument(
        "--max-pca-proj",
        type=int,
        default=6000,
        help="Max number of sequences projected per non-reference alignment.",
    )
    parser.add_argument(
        "--pair-sample-limit",
        type=int,
        default=1500000,
        help="Max number of random sequence pairs for pairwise Hamming histogram when exact computation is too large.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_export(
        project_root=args.project_root.resolve(),
        out_fig_dir=(args.project_root / args.out_fig_dir).resolve()
        if not args.out_fig_dir.is_absolute()
        else args.out_fig_dir,
        temp_dir=(args.project_root / args.temp_dir).resolve() if not args.temp_dir.is_absolute() else args.temp_dir,
        rerun_cleaning=args.rerun_cleaning,
        use_tex=args.use_tex,
        max_pca_ref=args.max_pca_ref,
        max_pca_proj=args.max_pca_proj,
        pair_sample_limit=args.pair_sample_limit,
    )


if __name__ == "__main__":
    main()
