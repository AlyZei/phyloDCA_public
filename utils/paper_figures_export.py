"""
Paper Figures Export Module
===========================
This module contains all functions necessary to generate Figures 2-5 for the paper
and export them to a single PDF with LaTeX captions.

Usage:
    from paper_figures_export import export_all_figures_to_pdf
    export_all_figures_to_pdf(config)
"""

import os
import re
import ast
import json
import pickle
import hashlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import torch
from matplotlib import cm, rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, gaussian_kde

# Import from local utils
from utils.toolsForTreesAndMSAs import read_fasta1, read_fasta2, calculate_hamming_distance
from utils.PottsEnergies import energy_of_msa, energy, read_potts_parameters_proteins
from utils.ci_and_cd_entropy import context_dependent_entropy_msa_torch
from utils.utils import createFolder, get_all_file_paths


# ============================================================
# --- HELPER FUNCTIONS ---
# ============================================================

# Hamming distance function is imported from toolsForTreesAndMSAs as calculate_hamming_distance


def yang_score(sequence: np.array, posterior_prob: np.array) -> float:
    """Average probability of amino acids in a sequence under posterior."""
    return np.sum(posterior_prob[np.arange(len(sequence)), sequence]) / len(sequence)


def _format_gt_legend_label(label):
    """Convert WT-like labels to LaTeX GT labels (e.g., WT1 or WT 1 -> $\bm{s}^\text{GT1}$)."""
    if not isinstance(label, str):
        return label
    match = re.match(r"^WT\s*(\d+)$", label)
    if match:
        idx = match.group(1)
        return rf'$\bm{{s}}^\text{{GT{idx}}}$'
    return label


def _file_signature(path: str):
    """Return a stable file signature for cache invalidation."""
    try:
        st = os.stat(path)
        return {
            "path": os.path.abspath(path),
            "mtime_ns": int(st.st_mtime_ns),
            "size": int(st.st_size),
        }
    except OSError:
        return {"path": os.path.abspath(path), "missing": True}


def _build_cache_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def get_pairwise_hamming_dist(msa_tensor, dist_matrix=False, batch_size=500):
    """
    Compute pairwise Hamming distance histogram using PyTorch (memory-efficient).
    
    Parameters:
        msa_tensor: torch.Tensor of shape (N, L) with integer sequences
        dist_matrix: if True, return full distance matrix instead of histogram
        batch_size: number of rows to process at once to limit memory usage
    
    Returns:
        histogram of pairwise distances or full distance matrix
    """
    if not isinstance(msa_tensor, torch.Tensor):
        msa_tensor = torch.tensor(msa_tensor, dtype=torch.long)
    
    N, L = msa_tensor.shape
    
    if dist_matrix:
        # For small MSAs, compute full matrix
        if N <= batch_size:
            matches = (msa_tensor.unsqueeze(0) == msa_tensor.unsqueeze(1)).sum(dim=2)
            return L - matches
        else:
            # Compute in batches
            distances = torch.zeros((N, N), dtype=torch.float32)
            for i in range(0, N, batch_size):
                end_i = min(i + batch_size, N)
                batch_i = msa_tensor[i:end_i]
                for j in range(0, N, batch_size):
                    end_j = min(j + batch_size, N)
                    batch_j = msa_tensor[j:end_j]
                    matches = (batch_i.unsqueeze(1) == batch_j.unsqueeze(0)).sum(dim=2)
                    distances[i:end_i, j:end_j] = L - matches
            return distances
    
    # Memory-efficient histogram computation using batches
    hist = torch.zeros(L + 1, dtype=torch.float32)
    
    # Process in batches to balance memory and speed
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        batch_i = msa_tensor[i:end_i]
        
        # Compare with all rows after this batch (for upper triangle)
        for j in range(i, N, batch_size):
            end_j = min(j + batch_size, N)
            batch_j = msa_tensor[j:end_j]
            
            # Compute pairwise distances for this batch pair
            # Shape: (batch_i_size, batch_j_size)
            distances_batch = (batch_i.unsqueeze(1) != batch_j.unsqueeze(0)).sum(dim=2)
            
            # For diagonal blocks, only count upper triangle
            if i == j:
                # Create mask for upper triangle within this block
                n_i = distances_batch.shape[0]
                triu_idx = torch.triu_indices(n_i, n_i, offset=1)
                upper_dists = distances_batch[triu_idx[0], triu_idx[1]]
                # Use bincount for efficiency
                counts = torch.bincount(upper_dists.long(), minlength=L+1)
                hist[:len(counts)] += counts.float()
            else:
                # For off-diagonal blocks, count all pairs
                all_dists = distances_batch.flatten()
                counts = torch.bincount(all_dists.long(), minlength=L+1)
                hist[:len(counts)] += counts.float()
    
    return hist


def load_ancestral_posterior(folder, seq, mu, prefix="DBD"):
    """Load ancestral posterior probability matrix."""
    path = f"{folder}/{prefix}_{seq}_mu{mu}_ancestral_probability"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.loadtxt(path)


def load_Felsenstein_samples(msa_folder, seq, mu, M):
    """Load Felsenstein-sampled MSA."""
    return np.loadtxt(f'{msa_folder}/{seq}_mu={mu}_depth=None_M={M}', dtype=int)


def load_reshuffled_MSA(msa_folder, seq, mu, M, T):
    """Load reshuffled/epistatic MSA (at temperature T)."""
    return np.loadtxt(f'{msa_folder}/{seq}_mu={mu}_depth=None_shuffled_M={M}_T={T}', dtype=int)


def compute_intra_pairwise_hamming(msa):
    """Compute intra-MSA pairwise Hamming distances."""
    D = 1 - (msa[:, None, :] == msa[None, :, :]).mean(axis=-1)
    avg_pairwise = D.mean(axis=1)
    min_pairwise = D.min(axis=1)
    return avg_pairwise, min_pairwise


def normalize(v):
    """Normalize array to [0, 1] range."""
    v = np.asarray(v, dtype=float)
    if np.allclose(v.max(), v.min()):
        return np.zeros_like(v)
    return (v - v.min()) / (v.max() - v.min())


def compute_all_scores(dca_seqs, posterior=None):
    """Compute all scoring metrics for sequences."""
    avg_pairwise, min_pairwise = compute_intra_pairwise_hamming(dca_seqs)
    yang = None
    if posterior is not None:
        yang = np.array([yang_score(seq, posterior) for seq in dca_seqs])
    return {"avg_pairwise": avg_pairwise, "min_pairwise": min_pairwise, "yang": yang}


def select_topN(dca_seqs, energies, posterior=None, scoring="pairwise", percentage=None, topN=10):
    """
    Select top N sequences based on scoring method.
    
    Parameters:
        dca_seqs: MSA sequences
        energies: Potts energies for each sequence
        posterior: ancestral probability matrix (needed for yang/combined)
        scoring: 'yang', 'pairwise', or 'combined'
        percentage: fraction of sequences to keep by lowest energy (None = keep all)
        topN: number of sequences to return
    """
    scores = compute_all_scores(dca_seqs, posterior)
    
    if scoring == "yang":
        if scores["yang"] is None:
            raise ValueError("posterior needed for yang scoring")
        base_scores = scores["yang"]
        higher_is_better = True
    elif scoring == "pairwise":
        base_scores = scores['avg_pairwise']
        higher_is_better = False
    elif scoring == "combined":
        if scores["yang"] is None:
            raise ValueError("posterior needed for combined scoring")
        norm_yang = normalize(scores["yang"])
        norm_min = normalize(scores["min_pairwise"])
        base_scores = (norm_yang + (1 - norm_min)) / 2.0
        higher_is_better = True
    else:
        raise ValueError("scoring must be 'yang'|'pairwise'|'combined'")
    
    if percentage is None:
        n_energy_keep = len(energies)
    else:
        n_energy_keep = max(1, int(np.floor(len(energies) * float(percentage))))
    
    energy_keep = np.argsort(energies)[:n_energy_keep]
    surv_scores = base_scores[energy_keep]
    
    if higher_is_better:
        order = np.argsort(-surv_scores)
    else:
        order = np.argsort(surv_scores)
    
    final_sorted = energy_keep[order]
    return final_sorted[:topN]


def _resolve_candidate_wts(
    sequences: list,
    wt_indices: list | tuple | None = None,
    include_wt24: bool = False,
):
    """Resolve selected WT sequence names from 1-based WT indices."""
    if wt_indices is None:
        wt_indices = [1, 3, 5]

    ordered = []
    for idx in wt_indices:
        try:
            i = int(idx)
        except Exception:
            continue
        if i >= 1 and i not in ordered:
            ordered.append(i)

    if include_wt24:
        for i in (2, 4):
            if i not in ordered:
                ordered.append(i)

    selected = []
    for i in ordered:
        if i <= len(sequences):
            selected.append(sequences[i - 1])
    return selected


def _collect_top10_candidates_by_wt_mu(
    selected_wts: list,
    mu_values: list,
    M: int,
    T: float,
    msa_folder: str,
    posterior_folder: str,
    GT_sequences: dict,
    fields_: np.ndarray,
    couplings: np.ndarray,
    energy_keep_pct: float | None = None,
    data_prefix: str = "DBD",
    topN: int = 10,
):
    """
    Gather Figure-5-style top-N candidates for each WT and mu.

    Returns:
        dict[wt][mu] with selected sequences and precomputed energy/distance arrays.
    """
    def _to_numpy_1d(x):
        if x is None:
            return np.array([])
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    all_data = {}

    for wt in selected_wts:
        gt = GT_sequences[wt]
        wt_data = {}

        for mu in mu_values:
            reshuffled_path = f"{msa_folder}/{wt}_mu={mu}_depth=None_shuffled_M={M}_T={T}"
            posterior_msa_path = f"{msa_folder}/{wt}_mu={mu}_depth=None_M={M}"
            posterior_path = os.path.join(
                posterior_folder,
                f"{data_prefix}_{wt}_mu{mu}_ancestral_probability",
            )

            if (not os.path.exists(reshuffled_path)) or (not os.path.exists(posterior_path)):
                continue

            try:
                dca_reshuffled = np.loadtxt(reshuffled_path, dtype=int)
                posterior = np.loadtxt(posterior_path)
                energies_reshuffled = energy_of_msa(dca_reshuffled, fields_, couplings)

                dca_idx = select_topN(
                    dca_reshuffled,
                    energies_reshuffled,
                    posterior=posterior,
                    scoring="yang",
                    percentage=energy_keep_pct,
                    topN=topN,
                )
                dca_idx = np.asarray(dca_idx, dtype=int)
                dca_seqs = dca_reshuffled[dca_idx] if dca_idx.size > 0 else np.empty((0, gt.shape[0]), dtype=int)

                si_seqs = np.empty((0, gt.shape[0]), dtype=int)
                if os.path.exists(posterior_msa_path):
                    posterior_msa = np.loadtxt(posterior_msa_path, dtype=int)
                    energies_posterior = energy_of_msa(posterior_msa, fields_, couplings)
                    si_idx = select_topN(
                        posterior_msa,
                        energies_posterior,
                        posterior=posterior,
                        scoring="yang",
                        percentage=None,
                        topN=topN,
                    )
                    si_idx = np.asarray(si_idx, dtype=int)
                    if si_idx.size > 0:
                        si_seqs = posterior_msa[si_idx]

                ml_seq = np.argmax(posterior, axis=1)

                dca_energy = _to_numpy_1d(energy_of_msa(dca_seqs, fields_, couplings)) if dca_seqs.shape[0] > 0 else np.array([])
                dca_dist = np.mean(dca_seqs != gt, axis=1) if dca_seqs.shape[0] > 0 else np.array([])

                si_energy = _to_numpy_1d(energy_of_msa(si_seqs, fields_, couplings)) if si_seqs.shape[0] > 0 else np.array([])
                si_dist = np.mean(si_seqs != gt, axis=1) if si_seqs.shape[0] > 0 else np.array([])

                wt_data[mu] = {
                    "dca_seqs": dca_seqs,
                    "si_seqs": si_seqs,
                    "dca_energy": dca_energy,
                    "dca_dist": dca_dist,
                    "si_energy": si_energy,
                    "si_dist": si_dist,
                    "starting_seq": np.asarray(gt, dtype=int),
                    "ml_seq": np.asarray(ml_seq, dtype=int),
                }
            except Exception as exc:
                print(f"    Warning: failed candidate extraction for WT={wt}, mu={mu}: {exc}")

        all_data[wt] = wt_data

    return all_data


# ============================================================
# --- FIGURE 2: MERGED FIGURE (Hamming + Pairwise Histogram) ---
# ============================================================

def create_figure2(
    natural_alignment: str,
    fasta_folder: str,
    mu_values: list,
    GT_sequences: dict,
    sequences: list,
    seq_to_label: dict,
    data_prefix: str = "Beta",
    num_bins: int = 77,
    figsize=(16, 8.6),
    include_pca_panels: bool = True,
    pca_wt_indices: tuple = (1, 3, 5),
    pca_mu_values: tuple = (1.0, 10.0, 50.0, 3000.0),
    pca_natural_seq_limit: int = 10000,
    pca_per_mu_seq_limit: int = 10000,
    pca_gt_in_legend: bool = False,
    pca_alignment: str | None = None,
    cache_dir: str = "figure2_precomputed_data",
    use_cache: bool = True,
    refresh_cache: bool = False,
):
    """
    Create Figure 2 with up to five panels:
    - Panel A: Hamming distance vs mu_gen
    - Panel B: Pairwise Hamming histogram
    - Panels C/D/E: PCA overlays for GT1/GT3/GT5 (natural density + mu outlines)
    
    Returns: matplotlib figure object
    """
    L = len(GT_sequences[sequences[0]])
    
    # Determine file pattern based on data_prefix
    if data_prefix == "DBD":
        file_pattern = "{prefix}_{seq}_mu{mu}_amino_DBDtree_collapsed_noonlychild_midpointrooted_mean1.fa"
        natural_label = "Natural DBD"
    else:
        file_pattern = "{prefix}_{seq}_mu{mu}_amino_betaLactree_collapsed_noonlychild_midpointrooted.fa"
        natural_label = r'$\bm{\mathcal{D}}_\text{extant}$'
    
    # Color scheme
    cmap_seq = cm.get_cmap('plasma')
    color_values = np.linspace(0.1, 0.9, len(sequences))
    colors_dict = {seq: cmap_seq(val) for seq, val in zip(sequences, color_values)}

    # Cache setup (shared by Figure 2 and Figure 2 bis)
    pca_fit_alignment = pca_alignment if pca_alignment is not None else natural_alignment
    preferred_mu_values = [0.2, 1.0, 10.0, 2000.0]
    selected_mu_values = [mu for mu in preferred_mu_values if mu in mu_values]
    if len(selected_mu_values) == 0 and len(mu_values) > 0:
        selected_indices = np.unique(np.logspace(0, np.log10(len(mu_values) - 1), 4, dtype=int))
        selected_mu_values = [mu_values[i] for i in selected_indices]

    cache_payload = None
    cache_file = None
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)

        referenced_files = [natural_alignment, pca_fit_alignment]
        for seq in sequences:
            for mu in mu_values:
                referenced_files.append(os.path.join(fasta_folder, file_pattern.format(prefix=data_prefix, seq=seq, mu=mu)))
        for seq in sequences:
            for mu_local in pca_mu_values:
                referenced_files.append(os.path.join(fasta_folder, file_pattern.format(prefix=data_prefix, seq=seq, mu=mu_local)))

        key_payload = {
            "version": 2,
            "natural_alignment": _file_signature(natural_alignment),
            "pca_alignment": _file_signature(pca_fit_alignment),
            "fasta_folder": os.path.abspath(fasta_folder),
            "data_prefix": data_prefix,
            "mu_values": [float(m) for m in mu_values],
            "selected_mu_values": [float(m) for m in selected_mu_values],
            "sequences": list(sequences),
            "pca_wt_indices": list(pca_wt_indices),
            "pca_mu_values": [float(m) for m in pca_mu_values],
            "pca_natural_seq_limit": int(pca_natural_seq_limit),
            "pca_per_mu_seq_limit": int(pca_per_mu_seq_limit),
            "files": [_file_signature(p) for p in referenced_files],
        }
        cache_key = _build_cache_key(key_payload)
        cache_file = os.path.join(cache_dir, f"figure2_data_{cache_key}.pkl")

        if (not refresh_cache) and os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cache_payload = pickle.load(f)
                print(f"Loaded Figure 2 cache: {cache_file}")
            except Exception:
                cache_payload = None
    
    # Figure layout: A/B on top row, C/D/E on bottom row
    if include_pca_panels:
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=figsize)
        # 3-column layout: A over C, B spans above D+E
        gs = GridSpec(2, 3, figure=fig, wspace=0.30, hspace=0.50)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1:3])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax3)
        ax5 = fig.add_subplot(gs[1, 2], sharex=ax3, sharey=ax3)
        pca_axes = [ax3, ax4, ax5]
    else:
        figsize_adjusted = (12, 5.4)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_adjusted, gridspec_kw={'wspace': 0.3})
        pca_axes = []
    
    # ---------------------------
    # PANEL A: Hamming vs \mu_gen
    # ---------------------------
    all_results = None if cache_payload is None else cache_payload.get("all_results")
    if all_results is None:
        all_results = {seq: {'mu': [], 'avg_hamming': []} for seq in sequences}

        for seq in sequences:
            root_sequence = GT_sequences[seq]
            for mu in mu_values:
                filename = file_pattern.format(prefix=data_prefix, seq=seq, mu=mu)
                filepath = os.path.join(fasta_folder, filename)
                if not os.path.exists(filepath):
                    continue
                msa = read_fasta1(filepath)
                distances = [calculate_hamming_distance(s, root_sequence) for s in msa]
                all_results[seq]['mu'].append(mu)
                all_results[seq]['avg_hamming'].append(float(np.mean(distances)))
    
    for seq in sequences:
        if all_results[seq]['mu']:
            # Normalize by sequence length
            L_seq = len(GT_sequences[seq])
            normalized_hamming = [h / L_seq for h in all_results[seq]['avg_hamming']]
            legend_label = _format_gt_legend_label(seq_to_label.get(seq, seq))
            ax1.plot(
                all_results[seq]['mu'],
                normalized_hamming,
                'o-', color=colors_dict[seq],
                label=legend_label, linewidth=2.5, markersize=7, alpha=0.85
            )
    
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$\mu_{\mathrm{gen}}$')
    ax1.set_ylabel(r'$\langle d_H(\bm{s}^\mathrm{GT}, \bm{s}_\mathrm{leaf}) \rangle$')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax1.text(-0.10, 1.08, r'\textbf{A}', transform=ax1.transAxes, fontsize=22, fontweight='bold', va='bottom')
    
    # ---------------------------
    # PANEL B: Pairwise Hamming histogram (normalized by sequence length)
    # ---------------------------
    seq = sequences[0]
    panel_b_cached = None if cache_payload is None else cache_payload.get("panel_b")

    # Normalize: bin_centers go from 0 to 1 (fraction of sequence length).
    # Figure 2 should always use exactly L+1 bins.
    num_bins_norm = L + 1 if num_bins != L + 1 else num_bins
    bin_edges = np.linspace(0, 1, num_bins_norm + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    if panel_b_cached is None:
        natural_seqs = read_fasta1(natural_alignment)
        natural_tensor = torch.tensor(natural_seqs, dtype=torch.long)
        natural_hist_raw = get_pairwise_hamming_dist(natural_tensor, dist_matrix=False).cpu().numpy()
        L_nat = natural_tensor.shape[1]

        # Convert raw histogram to normalized bins
        raw_bin_centers = np.arange(len(natural_hist_raw)) / L_nat
        natural_hist, _ = np.histogram(raw_bin_centers, bins=bin_edges, weights=natural_hist_raw)
        natural_hist = natural_hist / natural_hist.sum() if natural_hist.sum() > 0 else natural_hist

        panel_b_mu_hists = {}
        for mu in selected_mu_values:
            filename = file_pattern.format(prefix=data_prefix, seq=seq, mu=mu)
            filepath = os.path.join(fasta_folder, filename)
            if not os.path.exists(filepath):
                continue
            artificial_seqs = read_fasta1(filepath)
            tensor = torch.tensor(artificial_seqs, dtype=torch.long)
            hist_raw = get_pairwise_hamming_dist(tensor, dist_matrix=False).cpu().numpy()
            L_art = tensor.shape[1]
            raw_centers = np.arange(len(hist_raw)) / L_art
            hist, _ = np.histogram(raw_centers, bins=bin_edges, weights=hist_raw)
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            panel_b_mu_hists[float(mu)] = hist
    else:
        natural_hist = np.asarray(panel_b_cached["natural_hist"])
        panel_b_mu_hists = {float(k): np.asarray(v) for k, v in panel_b_cached["mu_hists"].items()}
    
    ax2.bar(bin_centers, natural_hist, width=bin_width, color='lightgray', edgecolor='darkgrey', linewidth=1.2, alpha=0.6)
    ax2.step(bin_centers, natural_hist, where='mid', color='black', linewidth=2, label=natural_label)
    
    base_color = colors_dict[seq]
    panel_cmap = LinearSegmentedColormap.from_list(f'{seq}_gradient', [(1,1,1,1), base_color], N=256)
    n_mu = len(selected_mu_values)
    
    for i, mu in enumerate(selected_mu_values):
        hist = panel_b_mu_hists.get(float(mu))
        if hist is not None:
            m_label = rf'$\mu_{{\mathrm{{gen}}}}$={mu:g}'
            color = panel_cmap((i+1)/n_mu)
            # Bar plot with no edge
            ax2.bar(bin_centers, hist, width=bin_width, alpha=0.6, edgecolor='none', color=color, label=m_label)
            # Add dark grey outline of the histogram shape
            ax2.step(bin_centers, hist, where='mid', color='dimgray', linewidth=1.2, alpha=0.9)
    
    ax2.set_xlabel('Normalized Pairwise Hamming Distance')
    ax2.set_ylabel('Frequency')
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=1e-6)
    # Set xlim to 1.0 (no extra space for external legend)
    ax2.set_xlim(-0.02, 1.02)
    
    # Set nice ticks for 0-1 range
    ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)

    # Fixed y-range up to 10^0 to leave clear space for the in-panel legend
    ax2.set_ylim(1e-6, 1e0)
    # Hide the 10^0 tick to avoid overlap with the panel label
    ax2.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Enforce requested legend order on a single line
    ordered_labels = [
        natural_label,
        r'$\mu_{\mathrm{gen}}$=0.2',
        r'$\mu_{\mathrm{gen}}$=1',
        r'$\mu_{\mathrm{gen}}$=10',
        r'$\mu_{\mathrm{gen}}$=2000',
    ]
    ordered_labels = [lbl for lbl in ordered_labels if lbl in by_label]
    ordered_handles = [by_label[lbl] for lbl in ordered_labels]

    # Compact legend that fits inside panel
    ax2.legend(ordered_handles, ordered_labels, loc='upper left', ncol=max(1, len(ordered_labels)),
               frameon=True, fancybox=True, shadow=True,
               borderpad=0.35, handlelength=1.2, handletextpad=0.4,
               labelspacing=0.25, columnspacing=0.8)
    ax2.text(-0.05, 1.08, r'\textbf{B}', transform=ax2.transAxes, fontsize=22, fontweight='bold', va='bottom')

    # ---------------------------
    # PANELS C/D/E: PCA overlays for GT1, GT3, GT5
    # ---------------------------
    if include_pca_panels and len(pca_axes) == 3:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        def _subsample_rows(arr: np.ndarray, limit: int) -> np.ndarray:
            if (limit is None) or (limit <= 0) or (arr.shape[0] <= limit):
                return arr
            idx = np.random.choice(arr.shape[0], size=int(limit), replace=False)
            return arr[idx]

        def _one_hot_flat(msa_int: np.ndarray, q_: int) -> np.ndarray:
            oh = np.eye(q_, dtype=np.float32)[msa_int]
            return oh.reshape(msa_int.shape[0], -1)

        def _plot_density(ax, x, y, cmap_local, levels=24):
            points = np.vstack([x, y])
            if points.shape[1] < 5:
                return
            kde = gaussian_kde(points)
            xx, yy = np.mgrid[x.min()-1.5:x.max()+1.5:320j, y.min()-1.5:y.max()+1.5:320j]
            grid = np.vstack([xx.ravel(), yy.ravel()])
            z = kde(grid).reshape(xx.shape)
            # Filled density (background extant layer)
            ax.contourf(xx, yy, z, levels=levels, cmap=cmap_local, alpha=0.95, zorder=1)
            # Add isolines to improve contrast/readability between density bands
            ax.contour(xx, yy, z, levels=10, colors=['#4a4a4a'], linewidths=0.45, alpha=0.55, zorder=2)

        def _plot_outline(ax, x, y, color, lw=2.6, percentile=76):
            points = np.vstack([x, y])
            if points.shape[1] < 5:
                return
            kde = gaussian_kde(points)
            # Use a larger, adaptive grid margin to avoid contours being cut at grid edges
            x_span = max(1e-6, float(x.max() - x.min()))
            y_span = max(1e-6, float(y.max() - y.min()))
            x_margin = max(2.0, 0.35 * x_span)
            y_margin = max(2.0, 0.35 * y_span)
            xx, yy = np.mgrid[
                x.min() - x_margin:x.max() + x_margin:260j,
                y.min() - y_margin:y.max() + y_margin:260j,
            ]
            z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            thr = np.percentile(z, percentile)
            # Plot main colored contour with crisp linewidth
            ax.contour(xx, yy, z, levels=[thr], colors=[color], linewidths=lw, zorder=10)
            # Add a subtle dark edge for sharpness
            ax.contour(xx, yy, z, levels=[thr], colors=['black'], linewidths=0.25, alpha=0.45, zorder=9)

        def _resolve_msa_path(seq_name, mu_val):
            cand_names = [
                file_pattern.format(prefix=data_prefix, seq=seq_name, mu=mu_val),
                file_pattern.format(prefix=data_prefix, seq=seq_name, mu=f"{float(mu_val):.1f}"),
            ]
            for name in cand_names:
                path = os.path.join(fasta_folder, name)
                if os.path.exists(path):
                    return path
            # fallback: broad search
            mu_str = str(mu_val)
            for fn in os.listdir(fasta_folder):
                if (seq_name in fn) and (f"mu{mu_str}" in fn) and fn.endswith((".fa", ".fasta")):
                    return os.path.join(fasta_folder, fn)
            return None

        pca_cached = None if cache_payload is None else cache_payload.get("pca")
        if pca_cached is None:
            pca_msa = np.asarray(read_fasta1(pca_fit_alignment), dtype=int)
            pca_msa = _subsample_rows(pca_msa, pca_natural_seq_limit)
            q = max(21, int(np.max(pca_msa)) + 1)

            pca_flat = _one_hot_flat(pca_msa, q)
            scaler = StandardScaler().fit(pca_flat)
            pca_sc = scaler.transform(pca_flat)
            pca = PCA(n_components=2)
            pca.fit(pca_sc)

            # Load natural alignment for gray density display (always use natural_alignment)
            natural_msa = np.asarray(read_fasta1(natural_alignment), dtype=int)
            natural_msa = _subsample_rows(natural_msa, pca_natural_seq_limit)
            # Pad or truncate to match PCA alignment length if needed
            if natural_msa.shape[1] != pca_msa.shape[1]:
                if natural_msa.shape[1] < pca_msa.shape[1]:
                    pad_width = ((0, 0), (0, pca_msa.shape[1] - natural_msa.shape[1]))
                    natural_msa = np.pad(natural_msa, pad_width, mode='constant', constant_values=0)
                else:
                    natural_msa = natural_msa[:, :pca_msa.shape[1]]
            natural_flat = _one_hot_flat(natural_msa, q)
            natural_sc = scaler.transform(natural_flat)
            natural_proj = pca.transform(natural_sc)

            seq_mu_proj = {}
            for seq_name in sequences:
                for mu_local in pca_mu_values:
                    msa_path = _resolve_msa_path(seq_name, mu_local)
                    if msa_path is None:
                        continue
                    try:
                        msa_mu = np.asarray(read_fasta1(msa_path), dtype=int)
                        msa_mu = _subsample_rows(msa_mu, pca_per_mu_seq_limit)
                        if msa_mu.shape[1] != natural_msa.shape[1]:
                            continue
                        if int(np.max(msa_mu)) >= q:
                            continue
                        flat_mu = _one_hot_flat(msa_mu, q)
                        seq_mu_proj[(seq_name, float(mu_local))] = pca.transform(scaler.transform(flat_mu))
                    except Exception:
                        continue

            gt_proj_by_seq = {}
            for seq_name in sequences:
                try:
                    gt_seq = np.asarray(GT_sequences[seq_name], dtype=int).reshape(1, -1)
                    if gt_seq.shape[1] == natural_msa.shape[1] and int(np.max(gt_seq)) < q:
                        gt_proj_by_seq[seq_name] = pca.transform(scaler.transform(_one_hot_flat(gt_seq, q)))
                except Exception:
                    continue
        else:
            natural_proj = np.asarray(pca_cached["natural_proj"])
            seq_mu_proj = {
                tuple(k): np.asarray(v)
                for k, v in pca_cached["seq_mu_proj"].items()
            }
            gt_proj_by_seq = {
                k: np.asarray(v)
                for k, v in pca_cached["gt_proj_by_seq"].items()
            }

        panel_letters = ['C', 'D', 'E']
        for ax, wt_idx_1based, letter in zip(pca_axes, pca_wt_indices, panel_letters):
            if wt_idx_1based < 1 or wt_idx_1based > len(sequences):
                ax.axis('off')
                continue

            seq_name = sequences[wt_idx_1based - 1]
            wt_color = colors_dict[seq_name]
            panel_cmap = LinearSegmentedColormap.from_list(
                f'pca_{seq_name}_cmap',
                [(1, 1, 1, 1), wt_color],
                N=256
            )
            # Grey colormap for natural sequences (independent of WT color) - darker gray for visibility
            grey_cmap = LinearSegmentedColormap.from_list(
                f'pca_{seq_name}_grey',
                [(1, 1, 1, 1), (0.2, 0.2, 0.2)],
                N=256
            )

            # Natural density in shades of grey
            _plot_density(ax, natural_proj[:, 0], natural_proj[:, 1], grey_cmap, levels=18)

            handles_local = []
            n_mu_local = max(1, len(pca_mu_values))
            for i_mu, mu_local in enumerate(pca_mu_values):
                proj_mu = seq_mu_proj.get((seq_name, float(mu_local)))
                if proj_mu is None:
                    continue
                # Better color spacing: start from 0.2 instead of normalized (i_mu+1)/(n_mu_local+1)
                # This provides more saturated, distinct colors
                contour_color = panel_cmap(0.2 + (i_mu / max(1, n_mu_local - 1)) * 0.75)
                _plot_outline(ax, proj_mu[:, 0], proj_mu[:, 1], contour_color, lw=2.6, percentile=76)
                handles_local.append(
                    Line2D([0], [0], color=contour_color, lw=3.0,
                           label=rf'$\mu_{{\mathrm{{gen}}}}={float(mu_local):g}$')
                )

            # Plot GT as diamond; either annotate directly (default) or add to legend (Figure 2 bis)
            gt_proj = gt_proj_by_seq.get(seq_name)
            if gt_proj is not None:
                gt_label = rf'$\bm{{s}}^{{\text{{GT{wt_idx_1based}}}}}$'
                gt_col = panel_cmap(0.97)
                ax.scatter(gt_proj[0, 0], gt_proj[0, 1], marker='D', s=85,
                           color=gt_col, edgecolor='black', linewidth=0.8, zorder=15)
                if pca_gt_in_legend:
                    handles_local.append(
                        Line2D([0], [0], marker='D', linestyle='None', markersize=8,
                               markerfacecolor=gt_col, markeredgecolor='black', markeredgewidth=0.8,
                               label=gt_label)
                    )
                else:
                    # Place label above diamond, with bold formatting and offset upward
                    ax.annotate(gt_label, xy=(gt_proj[0, 0], gt_proj[0, 1]),
                                xytext=(0, 12), textcoords='offset points',
                                fontsize=13, fontweight='bold', ha='center', va='bottom',
                                color='black', zorder=16, annotation_clip=True)

            # Panel formatting: no panel title, only legend for mu values (no GT in legend title)
            ax.set_title('')
            ax.set_xlabel('PC1')
            if letter == 'C':
                ax.set_ylabel('PC2')
                # Position C label above the panel, aligned left
                ax.text(-0.08, 1.08, rf'\textbf{{{letter}}}', transform=ax.transAxes,
                        fontsize=22, fontweight='bold', va='bottom', ha='left')
                # Keep y ticks visible on panel C
                ax.tick_params(axis='y', which='both', left=True, labelleft=True)
            else:
                ax.set_ylabel('')
                # Position D and E labels above their panels
                ax.text(-0.08, 1.08, rf'\textbf{{{letter}}}', transform=ax.transAxes,
                        fontsize=22, fontweight='bold', va='bottom', ha='left')
                # Remove y-ticks for D and E panels
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax.grid(alpha=0.25, linestyle='--')

            # For Figure 2 bis, expand the legend by 1/4 of a y-tick and set bounds (bottom <= -15, top <= 25)
            if pca_gt_in_legend:
                y0, y1 = ax.get_ylim()
                yt = ax.get_yticks()
                if len(yt) >= 2:
                    step = float(np.median(np.diff(yt)))
                else:
                    step = 0.25 * (y1 - y0 if y1 > y0 else 1.0)
                ax.set_ylim(min(-15.0, y0), min(25.0, y1 + 0.25 * step))
                
                # Extend x-axis to at least 25 to prevent clipping
                x0, x1 = ax.get_xlim()
                ax.set_xlim(x0, max(25.0, x1))

            # Legend without GT in title (just show mu values)
            if len(handles_local) > 0:
                leg = ax.legend(
                    handles=handles_local,
                    loc='upper right',
                    bbox_to_anchor=(0.98, 0.98),
                    borderaxespad=0.2,
                    frameon=True,
                    fancybox=True,
                    shadow=False,
                    framealpha=1.0,
                    facecolor='white',
                    edgecolor='0.3',
                    handlelength=1.4,
                    handletextpad=0.4,
                    labelspacing=0.25,
                )
                leg.set_zorder(30)
    
    if use_cache and (cache_payload is None) and (cache_file is not None):
        try:
            payload_to_save = {
                "all_results": all_results,
                "panel_b": {
                    "natural_hist": np.asarray(natural_hist),
                    "mu_hists": {float(k): np.asarray(v) for k, v in panel_b_mu_hists.items()},
                },
            }
            if include_pca_panels and len(pca_axes) == 3:
                payload_to_save["pca"] = {
                    "natural_proj": np.asarray(natural_proj),
                    "seq_mu_proj": {tuple(k): np.asarray(v) for k, v in seq_mu_proj.items()},
                    "gt_proj_by_seq": {k: np.asarray(v) for k, v in gt_proj_by_seq.items()},
                }
            with open(cache_file, "wb") as f:
                pickle.dump(payload_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved Figure 2 cache: {cache_file}")
        except Exception:
            pass

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def create_supplementary_figure2(
    natural_alignment: str,
    fasta_folder: str,
    mu_values: list,
    GT_sequences: dict,
    sequences: list,
    seq_to_label: dict,
    data_prefix: str = "Beta",
    num_bins: int = 77,
    figsize=(12, 5)
):
    """
    Create Supplementary Figure 2 with the same quantities as Figure 2,
    but parameterized by average GT-to-leaf Hamming distance.

    - Panel A: mu_gen as a function of <d_H(s^GT, s_leaf)>_D
    - Panel B: Pairwise Hamming histogram (same as Figure 2), with labels
               expressed by <d_H(s^GT, s_leaf)>_D instead of mu_gen.

    Returns: matplotlib figure object
    """
    L = len(GT_sequences[sequences[0]])

    if data_prefix == "DBD":
        file_pattern = "{prefix}_{seq}_mu{mu}_amino_DBDtree_collapsed_noonlychild_midpointrooted_mean1.fa"
        natural_label = "Natural DBD"
    else:
        file_pattern = "{prefix}_{seq}_mu{mu}_amino_betaLactree_collapsed_noonlychild_midpointrooted.fa"
        natural_label = r'$\mathcal{D}_{\text{extant}}$'

    cmap_seq = cm.get_cmap('plasma')
    color_values = np.linspace(0.1, 0.9, len(sequences))
    colors_dict = {seq: cmap_seq(val) for seq, val in zip(sequences, color_values)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace': 0.3})

    # ---------------------------
    # PANEL A: mu_gen vs average GT-to-leaf distance
    # ---------------------------
    all_results = {seq: {'mu': [], 'avg_hamming': []} for seq in sequences}

    for seq in sequences:
        root_sequence = GT_sequences[seq]
        for mu in mu_values:
            filename = file_pattern.format(prefix=data_prefix, seq=seq, mu=mu)
            filepath = os.path.join(fasta_folder, filename)
            if not os.path.exists(filepath):
                continue
            msa = read_fasta1(filepath)
            distances = [hamming_distance(s, root_sequence) for s in msa]
            all_results[seq]['mu'].append(mu)
            all_results[seq]['avg_hamming'].append(np.mean(distances))

    for seq in sequences:
        if all_results[seq]['mu']:
            L_seq = len(GT_sequences[seq])
            normalized_hamming = [h / L_seq for h in all_results[seq]['avg_hamming']]
            legend_label = _format_gt_legend_label(seq_to_label.get(seq, seq))
            ax1.plot(
                normalized_hamming,
                all_results[seq]['mu'],
                'o-', color=colors_dict[seq],
                label=legend_label, linewidth=2.5, markersize=7, alpha=0.85
            )

    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\langle d_H(\bm{s}^\mathrm{GT}, \bm{s}_\mathrm{leaf}) \rangle$')
    ax1.set_ylabel(r'$\mu_{\mathrm{gen}}$')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax1.text(-0.12, 1.05, r'\textbf{A}', transform=ax1.transAxes, fontsize=22, fontweight='bold', va='top')

    # ---------------------------
    # PANEL B: Pairwise Hamming histogram (same values as Figure 2)
    # ---------------------------
    seq = sequences[0]
    natural_seqs = read_fasta1(natural_alignment)
    natural_tensor = torch.tensor(natural_seqs, dtype=torch.long)
    natural_hist_raw = get_pairwise_hamming_dist(natural_tensor, dist_matrix=False).cpu().numpy()

    L_nat = natural_tensor.shape[1]
    # Figure 2 should always use exactly L+1 bins.
    num_bins_norm = L + 1 if num_bins != L + 1 else num_bins
    bin_edges = np.linspace(0, 1, num_bins_norm + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    raw_bin_centers = np.arange(len(natural_hist_raw)) / L_nat
    natural_hist, _ = np.histogram(raw_bin_centers, bins=bin_edges, weights=natural_hist_raw)
    natural_hist = natural_hist / natural_hist.sum() if natural_hist.sum() > 0 else natural_hist

    ax2.bar(bin_centers, natural_hist, width=bin_width, color='lightgray', edgecolor='darkgrey', linewidth=1.2, alpha=0.6)
    ax2.step(bin_centers, natural_hist, where='mid', color='black', linewidth=2, label=natural_label)

    selected_indices = np.unique(np.logspace(0, np.log10(len(mu_values)-1), 5, dtype=int))
    selected_indices = [idx for idx in selected_indices if idx != selected_indices[1]]
    selected_mu_values = [mu_values[i] for i in selected_indices]

    # Map mu -> normalized average GT-to-leaf distance for the displayed sequence
    mu_to_avg = {}
    if all_results[seq]['mu']:
        L_seq = len(GT_sequences[seq])
        mu_to_avg = {
            mu: avg_h / L_seq
            for mu, avg_h in zip(all_results[seq]['mu'], all_results[seq]['avg_hamming'])
        }

    base_color = colors_dict[seq]
    panel_cmap = LinearSegmentedColormap.from_list(f'{seq}_gradient', [(1,1,1,1), base_color], N=256)
    n_mu = len(selected_mu_values)

    for i, mu in enumerate(selected_mu_values):
        filename = file_pattern.format(prefix=data_prefix, seq=seq, mu=mu)
        filepath = os.path.join(fasta_folder, filename)
        if os.path.exists(filepath):
            artificial_seqs = read_fasta1(filepath)
            tensor = torch.tensor(artificial_seqs, dtype=torch.long)
            hist_raw = get_pairwise_hamming_dist(tensor, dist_matrix=False).cpu().numpy()
            L_art = tensor.shape[1]
            raw_centers = np.arange(len(hist_raw)) / L_art
            hist, _ = np.histogram(raw_centers, bins=bin_edges, weights=hist_raw)
            hist = hist / hist.sum() if hist.sum() > 0 else hist

            avg_val = mu_to_avg.get(mu, None)
            if avg_val is None:
                m_label = rf'$\langle d_H\rangle_{{\mathcal{{D}}}}={mu:g}$'
            else:
                m_label = rf'$\langle d_H\rangle_{{\mathcal{{D}}}}={avg_val:.3f}$'

            color = panel_cmap((i+1)/n_mu)
            ax2.bar(bin_centers, hist, width=bin_width, alpha=0.6, edgecolor='none', color=color, label=m_label)
            ax2.step(bin_centers, hist, where='mid', color='dimgray', linewidth=1.2, alpha=0.9)

    ax2.set_xlabel('Normalized Pairwise Hamming Distance')
    ax2.set_ylabel('Frequency\n' + r'$d_\text{H}(\bm{s}_i, \bm{s}_j), \bm{s}_i, \bm{s}_j \in \mathcal{D}$')
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=1e-6)
    ax2.set_xlim(-0.02, 1.55)
    ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)

    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(0.995, 0.99),
               frameon=True, fancybox=True, shadow=True)
    ax2.text(-0.12, 1.05, r'\textbf{B}', transform=ax2.transAxes, fontsize=22, fontweight='bold', va='top')

    plt.tight_layout()
    return fig


# ============================================================
# --- FIGURE 3: ML DISTANCES ---
# ============================================================

def create_figure3(
    mu_values: list,
    ancestral_save_folder: str,
    GT_sequences: dict,
    consensus_directory: str,
    sequences: list,
    seq_to_label: dict,
    colors_dict: dict,
    count_gaps: bool = True,
    figsize=(12, 10),
    data_prefix: str = "DBD"
):
    """
    Create Figure 3 with four panels (2x2 grid):
    - Panel A: ML vs GT distance
    - Panel B: Consensus vs GT distance  
    - Panel C: ML vs Consensus distance
    - Panel D: Consensus-GT minus ML-GT distance
    
    Returns: matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])  # A: top-left
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)  # B: top-right, share y-axis with A
    ax3 = fig.add_subplot(gs[1, 0])  # C: bottom-left
    ax4 = fig.add_subplot(gs[1, 1])  # D: bottom-right
    
    L = len(GT_sequences[sequences[0]])
    if data_prefix == "DBD":
        axis_label_fs = 16
        tick_fs = 13
        panel_label_fs = 24
        panel_label_x = -0.16
        panel_label_y = 1.03
    else:
        axis_label_fs = 14
        tick_fs = None
        panel_label_fs = 20
        panel_label_x = -0.15
        panel_label_y = 1.03
    
    # Build consensus_dict from files (excluding reweighted)
    consensus_dict = {}
    for file in get_all_file_paths(consensus_directory):
        if "reweighted" in file:
            continue
        # Extract just the filename for matching
        filename = os.path.basename(file)
        for seq in sequences:
            if seq in filename:
                for mu in mu_values:
                    # Use more precise pattern: _mu{value}_ to avoid mu1.0 matching mu10.0
                    mu_pattern = f"_mu{float(mu):.1f}_"
                    if mu_pattern in filename:
                        consensus_dict[(seq, mu)] = np.loadtxt(file, dtype=int)
                        break  # Found the mu for this file, no need to check others
    
    print(f"    Built consensus_dict with {len(consensus_dict)} entries")
    
    # PANEL A: ML vs GT
    results_gt = []
    for seq in sequences:
        GT = np.asarray(GT_sequences[seq])
        for mu in mu_values:
            path = f"{ancestral_save_folder}/{data_prefix}_{seq}_mu{mu}_ancestral_probability"
            if not os.path.exists(path):
                continue
            try:
                ancestral_dist = np.loadtxt(path)
                ML_seq = np.argmax(ancestral_dist, axis=1)
                dist = hamming_distance(ML_seq, GT)
                results_gt.append({"sequence": seq, "mu": mu, "ML_GT_distance": dist})
            except Exception:
                continue
    
    df_gt = pd.DataFrame(results_gt)

    for seq in sequences:
        seq_data = df_gt[df_gt['sequence'] == seq]
        if not seq_data.empty:
            # Normalize by sequence length
            normalized_dist = seq_data['ML_GT_distance'] / L
            x_vals = np.asarray(seq_data['mu'])
            y_vals = normalized_dist.values
            ax1.plot(x_vals, y_vals,
                     'o-', color=colors_dict[seq], label=seq_to_label[seq],
                     linewidth=2.5, markersize=7, alpha=0.85)
    
    ax1.set_xlabel(r'$\mu_{\mathrm{gen}}$', fontsize=axis_label_fs)
    ax1.set_xscale("log")
    ax1.set_ylabel(r'$d_\text{H}(\bm{s}^\text{MAP},\bm{s}^\text{GT})$', fontsize=axis_label_fs, labelpad=8)
    if tick_fs is not None:
        ax1.tick_params(axis='both', labelsize=tick_fs)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.text(panel_label_x, panel_label_y, r'\textbf{A}', transform=ax1.transAxes, fontsize=panel_label_fs, fontweight='bold', va='top')

    # Compute Consensus vs GT (needed for panels B and D)
    results_consensus_gt = []
    for seq in sequences:
        GT = np.asarray(GT_sequences[seq])
        for mu in mu_values:
            key = (seq, mu)
            if key not in consensus_dict:
                continue
            consensus_seq = np.asarray(consensus_dict[key])
            dist = calculate_hamming_distance(consensus_seq, GT)
            results_consensus_gt.append({"sequence": seq, "mu": mu, "Consensus_GT_distance": dist})
    
    df_consensus_gt = pd.DataFrame(results_consensus_gt)
    
    # PANEL B: Consensus vs GT
    for seq in sequences:
        seq_data = df_consensus_gt[df_consensus_gt['sequence'] == seq]
        if not seq_data.empty:
            # Normalize by sequence length
            normalized_dist = seq_data['Consensus_GT_distance'] / L
            x_vals = np.asarray(seq_data['mu'])
            y_vals = normalized_dist.values
            ax2.plot(x_vals, y_vals,
                     'o-', color=colors_dict[seq], label=seq_to_label[seq],
                     linewidth=2.5, markersize=7, alpha=0.85)
    
    ax2.set_xlabel(r'$\mu_{\mathrm{gen}}$', fontsize=axis_label_fs)
    ax2.set_xscale("log")
    ax2.set_ylabel(r'$d_\text{H}(\bm{s}^\text{cons},\bm{s}^\text{GT})$', fontsize=axis_label_fs, labelpad=8)
    if tick_fs is not None:
        ax2.tick_params(axis='both', labelsize=tick_fs)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.text(panel_label_x, panel_label_y, r'\textbf{B}', transform=ax2.transAxes, fontsize=panel_label_fs, fontweight='bold', va='top')
    
    # PANEL C: ML vs Consensus
    results_consensus = []
    for seq in sequences:
        for mu in mu_values:
            key = (seq, mu)
            if key not in consensus_dict:
                continue
            path = f"{ancestral_save_folder}/{data_prefix}_{seq}_mu{mu}_ancestral_probability"
            if not os.path.exists(path):
                continue
            try:
                ancestral_dist = np.loadtxt(path)
                ML_seq = np.argmax(ancestral_dist, axis=1)
                consensus_seq = np.asarray(consensus_dict[key])
                dist = calculate_hamming_distance(ML_seq, consensus_seq)
                results_consensus.append({"sequence": seq, "mu": mu, "ML_consensus_distance": dist})
            except Exception:
                continue
    
    df_consensus = pd.DataFrame(results_consensus)
    
    for seq in sequences:
        seq_data = df_consensus[df_consensus['sequence'] == seq]
        if not seq_data.empty:
            # Normalize by sequence length
            normalized_dist = seq_data['ML_consensus_distance'] / L
            ax3.plot(np.asarray(seq_data['mu']), normalized_dist,
                     'o-', color=colors_dict[seq], label=seq_to_label[seq],
                     linewidth=2.5, markersize=7, alpha=0.85)
    
    ax3.set_xlabel(r'$\mu_{\mathrm{gen}}$', fontsize=axis_label_fs)
    ax3.set_xscale("log")
    ax3.set_ylabel(r'$d_\text{H}(\bm{s}^\text{MAP},\bm{s}^\text{cons})$', fontsize=axis_label_fs, labelpad=8)
    if tick_fs is not None:
        ax3.tick_params(axis='both', labelsize=tick_fs)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.text(panel_label_x, panel_label_y, r'\textbf{C}', transform=ax3.transAxes, fontsize=panel_label_fs, fontweight='bold', va='top')
    
    # PANEL D: Consensus-GT minus ML-GT
    df_diff = df_gt.merge(df_consensus_gt, on=['sequence', 'mu'])
    df_diff['Consensus_GT_minus_ML_GT'] = df_diff['Consensus_GT_distance'] - df_diff['ML_GT_distance']
    
    for seq in sequences:
        seq_data = df_diff[df_diff['sequence'] == seq]
        if not seq_data.empty:
            # Normalize difference by sequence length
            normalized_diff = seq_data['Consensus_GT_minus_ML_GT'] / L
            ax4.plot(np.asarray(seq_data['mu']), normalized_diff,
                     'o-', color=colors_dict[seq], label=seq_to_label[seq],
                     linewidth=2.5, markersize=7, alpha=0.85)
    
    ax4.set_xlabel(r'$\mu_{\mathrm{gen}}$', fontsize=axis_label_fs)
    ax4.set_xscale("log")
    ax4.set_ylabel(r'$d_\text{H}(\bm{s}^\text{cons},\bm{s}^\text{GT}) - d_\text{H}(\bm{s}^\text{MAP},\bm{s}^\text{GT})$', fontsize=axis_label_fs, labelpad=8)
    if tick_fs is not None:
        ax4.tick_params(axis='both', labelsize=tick_fs)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax4.text(panel_label_x, panel_label_y, r'\textbf{D}', transform=ax4.transAxes, fontsize=panel_label_fs, fontweight='bold', va='top')
    
    # Shared legend at bottom (closer to panels, with GT-style labels)
    handles, labels = ax4.get_legend_handles_labels()
    legend_labels = []
    for lbl in labels:
        legend_labels.append(_format_gt_legend_label(lbl))

    fig.legend(handles, legend_labels, loc='lower center',
               bbox_to_anchor=(0.5, 0.01), ncol=5, frameon=True, fancybox=True,
               shadow=True, fontsize=11)

    plt.tight_layout(rect=[0, 0.09, 1, 1])
    return fig


def create_supplementary_figure3(
    mu_values: list,
    ancestral_save_folder: str,
    GT_sequences: dict,
    consensus_directory: str,
    sequences: list,
    seq_to_label: dict,
    colors_dict: dict,
    fasta_folder: str,
    count_gaps: bool = True,
    figsize=(12, 10),
    data_prefix: str = "DBD",
    panels_to_show: tuple = ("A", "B", "C", "D"),
):
    """
    Create Supplementary Figure 3 from Figure 3, but with all panel x-axes set to
    average (normalized) Hamming distance between generated leaves and GT root,
    computed for each (sequence, mu) from generated leaf MSAs.
    """
    # Build the base 4-panel figure first
    fig = create_figure3(
        mu_values=mu_values,
        ancestral_save_folder=ancestral_save_folder,
        GT_sequences=GT_sequences,
        consensus_directory=consensus_directory,
        sequences=sequences,
        seq_to_label=seq_to_label,
        colors_dict=colors_dict,
        count_gaps=count_gaps,
        figsize=figsize,
        data_prefix=data_prefix,
    )

    # File pattern for generated leaves
    if data_prefix == "DBD":
        file_pattern = "{prefix}_{seq}_mu{mu}_amino_DBDtree_collapsed_noonlychild_midpointrooted_mean1.fa"
    else:
        file_pattern = "{prefix}_{seq}_mu{mu}_amino_betaLactree_collapsed_noonlychild_midpointrooted.fa"

    # Precompute x-mapping: (seq, mu) -> avg normalized d_H(leaf, GT)
    avg_leaf_root_dist = {}
    for seq in sequences:
        GT = np.asarray(GT_sequences[seq])
        L_seq = len(GT)
        for mu in mu_values:
            filename = file_pattern.format(prefix=data_prefix, seq=seq, mu=mu)
            filepath = os.path.join(fasta_folder, filename)
            if not os.path.exists(filepath):
                continue
            try:
                msa = read_fasta1(filepath)
                if len(msa) == 0:
                    continue
                dists = [hamming_distance(s, GT) / L_seq for s in msa]
                avg_leaf_root_dist[(seq, mu)] = float(np.mean(dists))
            except Exception:
                continue

    # Apply transformed x-axis to all 4 panels in create_figure3
    ax1, ax2, ax3, ax4 = fig.axes[:4]
    panel_axes = {"A": ax1, "B": ax2, "C": ax3, "D": ax4}
    label_to_seq = {v: k for k, v in seq_to_label.items()}

    def _map_mu_to_avg(seq_name, x_mu):
        # Prefer exact key, then fallback to isclose match across provided mu values
        key = (seq_name, float(x_mu))
        if key in avg_leaf_root_dist:
            return avg_leaf_root_dist[key]
        for mu in mu_values:
            if np.isclose(float(x_mu), float(mu), rtol=1e-8, atol=1e-10):
                key2 = (seq_name, mu)
                if key2 in avg_leaf_root_dist:
                    return avg_leaf_root_dist[key2]
        return np.nan

    for ax in [ax1, ax2, ax3, ax4]:
        for line in ax.get_lines():
            seq_label = line.get_label()
            seq = label_to_seq.get(seq_label)
            if seq is None:
                continue

            old_x = np.asarray(line.get_xdata(), dtype=float)
            new_x = [_map_mu_to_avg(seq, x) for x in old_x]
            line.set_xdata(np.asarray(new_x, dtype=float))

        ax.set_xscale("linear")
        ax.set_xlabel(r'$\langle d_H(\bm{s}^\mathrm{GT}, \bm{s}_\mathrm{leaf}) \rangle$', fontsize=28)
        ax.tick_params(labelsize=24)
        ax.relim()
        ax.autoscale_view()

    # Optional panel filtering (e.g., keep only A and B)
    show = tuple(p.upper() for p in panels_to_show if str(p).upper() in panel_axes)
    if not show:
        show = ("A", "B", "C", "D")

    if set(show) != {"A", "B", "C", "D"}:
        # Remove the original shared legend (created in create_figure3)
        for leg in list(fig.legends):
            leg.remove()

        # Remove hidden axes
        for key, ax in panel_axes.items():
            if key not in show:
                ax.remove()

        # Re-layout for the common use case requested in notebook: keep A and B only
        if set(show) == {"A", "B"}:
            ax1.set_position([0.08, 0.35, 0.40, 0.60])
            ax2.set_position([0.56, 0.35, 0.40, 0.60])
            handles, labels = ax2.get_legend_handles_labels()
            legend_labels = [_format_gt_legend_label(lbl) for lbl in labels]
            fig.legend(handles, legend_labels, loc='lower center',
                       bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=True, fancybox=True,
                       shadow=True, fontsize=22)

    return fig


def create_supplementary_figure3_bis(
    mu_values: list,
    ancestral_save_folder: str,
    GT_sequences: dict,
    consensus_directory: str,
    sequences: list,
    seq_to_label: dict,
    colors_dict: dict,
    fasta_folder: str,
    count_gaps: bool = True,
    figsize=(16, 5),
    data_prefix: str = "DBD",
    panels_to_show: tuple = ("A", "B", "C", "D"),
):
    """
    Create Supplementary Figure 3 bis from Figure 3 with transformed x-axis
    and only panels A/B shown, while preserving Figure 3 panel sizing and
    typography.
    """
    # Build the base 4-panel figure first
    fig = create_figure3(
        mu_values=mu_values,
        ancestral_save_folder=ancestral_save_folder,
        GT_sequences=GT_sequences,
        consensus_directory=consensus_directory,
        sequences=sequences,
        seq_to_label=seq_to_label,
        colors_dict=colors_dict,
        count_gaps=count_gaps,
        figsize=figsize,
        data_prefix=data_prefix,
    )

    # File pattern for generated leaves
    if data_prefix == "DBD":
        file_pattern = "{prefix}_{seq}_mu{mu}_amino_DBDtree_collapsed_noonlychild_midpointrooted_mean1.fa"
    else:
        file_pattern = "{prefix}_{seq}_mu{mu}_amino_betaLactree_collapsed_noonlychild_midpointrooted.fa"

    # Precompute x-mapping: (seq, mu) -> avg normalized d_H(leaf, GT)
    avg_leaf_root_dist = {}
    for seq in sequences:
        GT = np.asarray(GT_sequences[seq])
        L_seq = len(GT)
        for mu in mu_values:
            filename = file_pattern.format(prefix=data_prefix, seq=seq, mu=mu)
            filepath = os.path.join(fasta_folder, filename)
            if not os.path.exists(filepath):
                continue
            try:
                msa = read_fasta1(filepath)
                if len(msa) == 0:
                    continue
                dists = [hamming_distance(s, GT) / L_seq for s in msa]
                avg_leaf_root_dist[(seq, mu)] = float(np.mean(dists))
            except Exception:
                continue

    # Apply transformed x-axis to all 4 panels in create_figure3
    ax1, ax2, ax3, ax4 = fig.axes[:4]
    panel_axes = {"A": ax1, "B": ax2, "C": ax3, "D": ax4}
    label_to_seq = {v: k for k, v in seq_to_label.items()}

    def _map_mu_to_avg(seq_name, x_mu):
        # Prefer exact key, then fallback to isclose match across provided mu values
        key = (seq_name, float(x_mu))
        if key in avg_leaf_root_dist:
            return avg_leaf_root_dist[key]
        for mu in mu_values:
            if np.isclose(float(x_mu), float(mu), rtol=1e-8, atol=1e-10):
                key2 = (seq_name, mu)
                if key2 in avg_leaf_root_dist:
                    return avg_leaf_root_dist[key2]
        return np.nan

    for ax in [ax1, ax2, ax3, ax4]:
        for line in ax.get_lines():
            seq_label = line.get_label()
            seq = label_to_seq.get(seq_label)
            if seq is None:
                continue

            old_x = np.asarray(line.get_xdata(), dtype=float)
            new_x = [_map_mu_to_avg(seq, x) for x in old_x]
            line.set_xdata(np.asarray(new_x, dtype=float))

        ax.set_xscale("linear")
        # Keep Figure 3 typography (xlabel size 14)
        ax.set_xlabel(r'$\langle d_H(\bm{s}^\mathrm{GT}, \bm{s}_\mathrm{leaf}) \rangle$', fontsize=14)
        ax.relim()
        ax.autoscale_view()

    # Optional panel filtering - for S3bis we always show A and B only
    show = ("A", "B")

    if set(show) != {"A", "B", "C", "D"}:
        # Remove the original shared legend (created in create_figure3)
        for leg in list(fig.legends):
            leg.remove()

        # Remove hidden axes
        for key, ax in panel_axes.items():
            if key not in show:
                ax.remove()

        # Custom A/B layout and typography for Supplementary Figure 3 bis
        if set(show) == {"A", "B"}:
            # Fill more vertical space while reserving room for legend
            ax1.set_position([0.08, 0.22, 0.40, 0.68])
            ax2.set_position([0.56, 0.22, 0.40, 0.68])

            # Make axis-label sizes more balanced
            ax1.set_xlabel(ax1.get_xlabel(), fontsize=20)
            ax2.set_xlabel(ax2.get_xlabel(), fontsize=20)
            ax1.set_ylabel(ax1.get_ylabel(), fontsize=20)
            ax2.set_ylabel(ax2.get_ylabel(), fontsize=20)

            # Reduce oversized panel letters A/B
            for text_obj in list(ax1.texts):
                if 'textbf' in text_obj.get_text():
                    text_obj.remove()
            for text_obj in list(ax2.texts):
                if 'textbf' in text_obj.get_text():
                    text_obj.remove()
            ax1.text(-0.18, 1.08, r'\textbf{A}', transform=ax1.transAxes, fontsize=28, fontweight='bold', va='top')
            ax2.text(-0.18, 1.08, r'\textbf{B}', transform=ax2.transAxes, fontsize=28, fontweight='bold', va='top')

            handles, labels = ax2.get_legend_handles_labels()
            legend_labels = [_format_gt_legend_label(lbl) for lbl in labels]
            fig.legend(handles, legend_labels, loc='lower center',
                       bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=True, fancybox=True,
                       shadow=True, fontsize=22)

    return fig


def create_supplementary_figure4(
    mu_values: list,
    ancestral_save_folder: str,
    GT_sequences: dict,
    consensus_directory: str,
    sequences: list,
    seq_to_label: dict,
    colors_dict: dict,
    fasta_folder: str,
    figsize=(12, 9),
    data_prefix: str = "DBD"
):
    """
    Create Supplementary Figure 4 from panels C and D of Supplementary Figure 3,
    using y/x values where:
        x = average normalized Hamming distance between generated leaves and GT root
        y_C = d_H(MAP, cons)
        y_D = d_H(cons, GT) - d_H(MAP, GT)

    The figure contains two indexed-by-order panels:
        - C: one y/x curve per sGT
        - D: one y/x curve per sGT
    """
    # ------------------------------------------------------------
    # Build consensus dict
    # ------------------------------------------------------------
    consensus_dict = {}
    for file in get_all_file_paths(consensus_directory):
        if "reweighted" in file:
            continue
        filename = os.path.basename(file)
        for seq in sequences:
            if seq in filename:
                for mu in mu_values:
                    mu_pattern = f"_mu{float(mu):.1f}_"
                    if mu_pattern in filename:
                        consensus_dict[(seq, mu)] = np.loadtxt(file, dtype=int)
                        break

    # ------------------------------------------------------------
    # x-mapping: (seq, mu) -> avg normalized d_H(leaf, GT)
    # ------------------------------------------------------------
    if data_prefix == "DBD":
        file_pattern = "{prefix}_{seq}_mu{mu}_amino_DBDtree_collapsed_noonlychild_midpointrooted_mean1.fa"
    else:
        file_pattern = "{prefix}_{seq}_mu{mu}_amino_betaLactree_collapsed_noonlychild_midpointrooted.fa"

    avg_leaf_root_dist = {}
    for seq in sequences:
        GT = np.asarray(GT_sequences[seq])
        L_seq = len(GT)
        for mu in mu_values:
            filepath = os.path.join(fasta_folder, file_pattern.format(prefix=data_prefix, seq=seq, mu=mu))
            if not os.path.exists(filepath):
                continue
            try:
                msa = read_fasta1(filepath)
                if len(msa) == 0:
                    continue
                avg_leaf_root_dist[(seq, mu)] = float(np.mean([calculate_hamming_distance(s, GT) / L_seq for s in msa]))
            except Exception:
                continue

    # ------------------------------------------------------------
    # Build C and D y-data (normalized, as in Figure 3)
    # ------------------------------------------------------------
    L = len(GT_sequences[sequences[0]])

    # ML vs GT
    results_gt = []
    for seq in sequences:
        GT = np.asarray(GT_sequences[seq])
        for mu in mu_values:
            path = f"{ancestral_save_folder}/{data_prefix}_{seq}_mu{mu}_ancestral_probability"
            if not os.path.exists(path):
                continue
            try:
                ancestral_dist = np.loadtxt(path)
                ML_seq = np.argmax(ancestral_dist, axis=1)
                dist = hamming_distance(ML_seq, GT)
                results_gt.append({"sequence": seq, "mu": mu, "ML_GT_distance": dist / L})
            except Exception:
                continue
    df_gt = pd.DataFrame(results_gt)

    # Consensus vs GT
    results_cons_gt = []
    for seq in sequences:
        GT = np.asarray(GT_sequences[seq])
        for mu in mu_values:
            key = (seq, mu)
            if key not in consensus_dict:
                continue
            cons = np.asarray(consensus_dict[key])
            dist = calculate_hamming_distance(cons, GT)
            results_cons_gt.append({"sequence": seq, "mu": mu, "Consensus_GT_distance": dist / L})
    df_cons_gt = pd.DataFrame(results_cons_gt)

    # Panel C: ML vs consensus
    results_c = []
    for seq in sequences:
        for mu in mu_values:
            key = (seq, mu)
            if key not in consensus_dict:
                continue
            path = f"{ancestral_save_folder}/{data_prefix}_{seq}_mu{mu}_ancestral_probability"
            if not os.path.exists(path):
                continue
            try:
                ancestral_dist = np.loadtxt(path)
                ML_seq = np.argmax(ancestral_dist, axis=1)
                cons = np.asarray(consensus_dict[key])
                y_val = calculate_hamming_distance(ML_seq, cons) / L
                x_val = avg_leaf_root_dist.get((seq, mu), np.nan)
                results_c.append({"sequence": seq, "mu": mu, "x": x_val, "y": y_val})
            except Exception:
                continue
    df_c = pd.DataFrame(results_c)

    # Panel D: Cons-GT minus ML-GT
    df_d = pd.DataFrame()
    if not df_gt.empty and not df_cons_gt.empty:
        df_d = df_gt.merge(df_cons_gt, on=["sequence", "mu"])
        df_d["y"] = df_d["Consensus_GT_distance"] - df_d["ML_GT_distance"]
        df_d["x"] = df_d.apply(lambda r: avg_leaf_root_dist.get((r["sequence"], r["mu"]), np.nan), axis=1)

    # Ratios y/x
    df_c = df_c.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
    df_d = df_d.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
    df_c = df_c[df_c["x"] > 0].copy()
    df_d = df_d[df_d["x"] > 0].copy()
    df_c["ratio"] = df_c["y"] / df_c["x"]
    df_d["ratio"] = df_d["y"] / df_d["x"]

    # ------------------------------------------------------------
    # Plot: indexed-by-order only, one curve per sGT
    # ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"wspace": 0.25})

    for seq in sequences:
        label = _format_gt_legend_label(seq_to_label.get(seq, seq))
        color = colors_dict.get(seq, None)

        seq_c = df_c[df_c["sequence"] == seq] if not df_c.empty else pd.DataFrame()
        if not seq_c.empty:
            c_sorted = np.sort(seq_c["ratio"].values)
            ax1.plot(np.arange(1, len(c_sorted) + 1), c_sorted, 'o-',
                     color=color, markersize=3, linewidth=1.2, alpha=0.9, label=label)

        seq_d = df_d[df_d["sequence"] == seq] if not df_d.empty else pd.DataFrame()
        if not seq_d.empty:
            d_sorted = np.sort(seq_d["ratio"].values)
            ax2.plot(np.arange(1, len(d_sorted) + 1), d_sorted, 'o-',
                     color=color, markersize=3, linewidth=1.2, alpha=0.9, label=label)

    ax1.set_title("C: y/x indexed by order")
    ax1.set_xlabel("Ordered index")
    ax1.set_ylabel("y/x")
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2.set_title("D: y/x indexed by order")
    ax2.set_xlabel("Ordered index")
    ax2.set_ylabel("y/x")
    ax2.grid(True, alpha=0.3, linestyle='--')

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(handles, labels, loc='best', fontsize=10, frameon=True)

    fig.suptitle(r"Supplementary Figure 4: Panels C and D from S3 as $y/x$ (one curve per $\bm{s}^{\mathrm{GT}}$)", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def create_supplementary_map_confidence_vs_gt_distance(
    mu_values: list,
    ancestral_save_folder: str,
    GT_sequences: dict,
    sequences: list,
    seq_to_label: dict,
    colors_dict: dict | None = None,
    data_prefix: str = "DBD",
    figsize: tuple = (16, 9),
    use_latex: bool = True,
):
    """
    Supplementary figure:
        One panel per WT. For each mu, plot one point where:
            x = normalized Hamming distance d_H(MAP, GT)
            y = 1 - average sitewise confidence mean_i p_i(argmax_a p_i(a))
    """
    if colors_dict is None:
        cmap_seq = cm.get_cmap("plasma")
        color_values = np.linspace(0.1, 0.9, len(sequences))
        colors_dict = {seq: cmap_seq(val) for seq, val in zip(sequences, color_values)}

    if use_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })

    title_fs = 28
    axis_label_fs = 28
    tick_fs = 24
    panel_letter_fs = 32
    cbar_label_fs = 24
    cbar_tick_fs = 20

    n_wt = len(sequences)

        # Special article layout for 5 panels: 3 on top, 2 centered below.
    if n_wt == 5:
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 6, figure=fig)
        axes = [
            fig.add_subplot(gs[0, 0:2]),
            fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[0, 4:6]),
            fig.add_subplot(gs[1, 1:3]),
            fig.add_subplot(gs[1, 3:5]),
        ]
        n_cols = 3
        n_rows = 2
    else:
        n_cols = min(3, max(1, n_wt))
        n_rows = int(np.ceil(n_wt / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel().tolist()

    global_x_min, global_x_max = np.inf, -np.inf
    global_y_min, global_y_max = np.inf, -np.inf
    last_scatter = None
    pearson_by_seq = {}

    used_axes = []
    for idx, seq in enumerate(sequences):
        ax = axes[idx]
        used_axes.append(ax)
        gt = np.asarray(GT_sequences[seq], dtype=int)
        wt_color = colors_dict.get(seq, "#1f77b4")

        rows = []
        for mu in mu_values:
            ppath = os.path.join(ancestral_save_folder, f"{data_prefix}_{seq}_mu{mu}_ancestral_probability")
            if not os.path.exists(ppath):
                continue

            try:
                posterior = np.asarray(np.loadtxt(ppath), dtype=float)
                if posterior.ndim != 2 or posterior.shape[0] != gt.shape[0]:
                    continue

                map_seq = np.argmax(posterior, axis=1)
                x_dist = float(np.mean(map_seq != gt))
                y_inv_conf = 1.0 - float(np.mean(np.max(posterior, axis=1)))
                rows.append((float(mu), x_dist, y_inv_conf))
            except Exception:
                continue

        if len(rows) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=tick_fs)
            ax.set_title(seq_to_label.get(seq, seq), fontsize=title_fs)
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.tick_params(axis="both", labelsize=tick_fs)
            continue

        rows = sorted(rows, key=lambda t: t[0])
        mu_arr = np.asarray([r[0] for r in rows], dtype=float)
        x_arr = np.asarray([r[1] for r in rows], dtype=float)
        y_arr = np.asarray([r[2] for r in rows], dtype=float)

        # Draw a light trajectory to emphasize ordering in mu while keeping one point per mu.
        ax.plot(x_arr, y_arr, color=wt_color, alpha=0.35, linewidth=1.6, zorder=1)
        last_scatter = ax.scatter(
            x_arr,
            y_arr,
            c=np.log10(mu_arr),
            cmap="viridis",
            s=88,
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )

        if len(x_arr) >= 2 and (not np.allclose(x_arr, x_arr[0])) and (not np.allclose(y_arr, y_arr[0])):
            try:
                r_val, p_val = pearsonr(x_arr, y_arr)
            except Exception:
                r_val, p_val = np.nan, np.nan
        else:
            r_val, p_val = np.nan, np.nan
        pearson_by_seq[seq] = {"r": float(r_val), "p": float(p_val), "n": int(len(x_arr))}

        global_x_min = min(global_x_min, float(np.min(x_arr)))
        global_x_max = max(global_x_max, float(np.max(x_arr)))
        global_y_min = min(global_y_min, float(np.min(y_arr)))
        global_y_max = max(global_y_max, float(np.max(y_arr)))

        gt_title = str(seq_to_label.get(seq, seq)).replace("WT", "GT")
        if np.isfinite(r_val):
            gt_title = f"{gt_title} (r={r_val:.3f})"
        else:
            gt_title = f"{gt_title} (r=NA)"

        ax.set_title(gt_title, fontsize=title_fs, pad=10)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.tick_params(axis="both", labelsize=tick_fs)

        panel_letter = chr(ord("A") + idx)
        panel_text = rf"\textbf{{{panel_letter}}}" if plt.rcParams.get("text.usetex", False) else panel_letter
        ax.text(
            -0.055,
            1.035,
            panel_text,
            transform=ax.transAxes,
            fontsize=panel_letter_fs,
            fontweight="bold",
            va="bottom",
            ha="left",
            clip_on=False,
            zorder=50,
        )

    for k in range(n_wt, len(axes)):
        axes[k].set_visible(False)

    if np.isfinite(global_x_min) and np.isfinite(global_x_max):
        dx = max(1e-3, global_x_max - global_x_min)
        for ax in used_axes:
            ax.set_xlim(global_x_min - 0.08 * dx, global_x_max + 0.08 * dx)
    if np.isfinite(global_y_min) and np.isfinite(global_y_max):
        dy = max(1e-3, global_y_max - global_y_min)
        for ax in used_axes:
            ax.set_ylim(max(0.0, global_y_min - 0.10 * dy), min(1.0, global_y_max + 0.10 * dy))

    for ax in used_axes:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lo = max(xlim[0], ylim[0])
        hi = min(xlim[1], ylim[1])
        if hi > lo:
            ax.plot([lo, hi], [lo, hi], linestyle=":", color="0.35", linewidth=1.2, zorder=0)

    keep_y_tick_panels = {"A", "D"}
    for i, ax in enumerate(used_axes):
        panel_letter = chr(ord("A") + i)
        if panel_letter in keep_y_tick_panels:
            ax.set_ylabel(r"$1-\langle \max_a\, p_i(a) \rangle_i$", fontsize=axis_label_fs)
            ax.tick_params(axis="y", left=True, labelleft=True, labelsize=tick_fs)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)

    for i, ax in enumerate(used_axes):
        row_id = i // n_cols
        if row_id == n_rows - 1:
            ax.set_xlabel(r"$d_\mathrm{H}(\bm{s}^{\mathrm{MAP}}, \bm{s}^{\mathrm{GT}})$", fontsize=axis_label_fs)

    if last_scatter is not None:
        cax = fig.add_axes([0.955, 0.18, 0.020, 0.68])
        cbar = fig.colorbar(last_scatter, cax=cax, orientation="vertical")
        cbar.set_label(r"$\log_{10}(\mu_\mathrm{gen})$", fontsize=cbar_label_fs)
        cbar.ax.tick_params(labelsize=cbar_tick_fs)

    fig.subplots_adjust(left=0.08, right=0.93, top=0.90, bottom=0.14, wspace=0.22, hspace=0.30)

    print("Pearson correlation per WT (x=d_H(MAP,GT), y=1-avg MAP confidence):")
    for seq in sequences:
        stats = pearson_by_seq.get(seq)
        label = str(seq_to_label.get(seq, seq)).replace("WT", "GT")
        if stats is None:
            print(f"  {label}: r=NA, p=NA, n=0")
            continue
        if np.isfinite(stats["r"]) and np.isfinite(stats["p"]):
            print(f"  {label}: r={stats['r']:.4f}, p={stats['p']:.3e}, n={stats['n']}")
        else:
            print(f"  {label}: r=NA, p=NA, n={stats['n']}")

    return fig


# ============================================================
# --- FIGURE 4: HAMMING VS POTTS GRID ---
# ============================================================

def create_figure4(
    sequences: list,
    mu_values: list,
    M: int,
    msa_save_folder: str,
    ancestral_probabilities_folder: str,
    consensus_directory: str,
    GT_sequences: dict,
    couplings: np.ndarray,
    fields_: np.ndarray,
    T: float,
    seq_to_label: dict,
    subset: str = "all",
    data_prefix: str = "DBD"
):
    """
    Create Figure 4: Grid plot of Hamming distance to GT vs Potts energy.
    
    Returns: matplotlib figure object
    """
    # Smaller, appropriate font sizes for the figure
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 21,
        "ytick.labelsize": 21,
        "figure.titlesize": 12,
        "figure.titleweight": "bold",
    })
    
    cmap = cm.get_cmap("plasma")
    color_values = np.linspace(0.1, 0.9, len(sequences))
    colors_dict = {seq: cmap(val) for seq, val in zip(sequences, color_values)}
    
    def lighten_color(color, factor=0.45):
        r, g, b = color[:3]
        return (r + (1-r)*factor, g + (1-g)*factor, b + (1-b)*factor)
    
    # Filter mus
    mu_values = [mu for mu in mu_values if 1.0 <= mu <= 100.0]
    print(f"  Figure 4: Filtered mu_values = {mu_values}")
    if subset == "first_last":
        sequences = [sequences[0], sequences[-1]]
    
    # Load consensus
    consensus_dict = {}
    for file in get_all_file_paths(consensus_directory):
        if "reweighted" in file:
            continue
        for seq in sequences:
            if seq in file:
                for mu in mu_values:
                    # Try both .1f and exact match for flexibility
                    mu_str1 = f"mu{float(mu):.1f}"
                    mu_str2 = f"mu{mu}"
                    if mu_str1 in file or mu_str2 in file:
                        consensus_dict[(seq, mu)] = np.loadtxt(file, dtype=int)
    
    print(f"  Loaded {len(consensus_dict)} consensus sequences")
    
    # Create figure - larger subplots with less spacing
    n_rows = len(sequences)
    n_cols = len(mu_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows), 
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    
    print(f"  Figure 4: {n_rows} rows x {n_cols} cols, sequences={sequences}, mu_values={mu_values}")
    
    # WT legend handles
    wt_legend_handles = []
    for seq in sequences:
        wt_label = _format_gt_legend_label(seq_to_label[seq])
        wt_legend_handles.append(
            Line2D([0], [0], marker='o', color=colors_dict[seq], linestyle='', markersize=8,
                   label=wt_label)
        )
    
    # Main plot loop
    for i, seq in enumerate(sequences):
        wt_color = colors_dict[seq]
        bayes_color = lighten_color(wt_color)
        
        for j, mu in enumerate(mu_values):
            ax = axes[i, j]
            
            msa_file = f"{msa_save_folder}/{seq}_mu={mu}_depth=None_M={M}"
            shuffled_file = f"{msa_save_folder}/{seq}_mu={mu}_depth=None_shuffled_M={M}_T={T}"
            
            if not os.path.exists(msa_file):
                print(f"    ⚠️ Missing MSA file: {msa_file}")
                ax.text(0.5, 0.5, 'No MSA', ha='center', va='center', transform=ax.transAxes, fontsize=8)
                continue
            if not os.path.exists(shuffled_file):
                print(f"    ⚠️ Missing shuffled file: {shuffled_file}")
                ax.text(0.5, 0.5, 'No shuffled', ha='center', va='center', transform=ax.transAxes, fontsize=8)
                continue
            
            MSA = np.loadtxt(msa_file, dtype=int)
            reshuffled_MSA = np.loadtxt(shuffled_file, dtype=int)
            
            path = f"{ancestral_probabilities_folder}/{data_prefix}_{seq}_mu{mu}_ancestral_probability"
            if not os.path.exists(path):
                print(f"    ⚠️ Missing ancestral file: {path}")
                ax.text(0.5, 0.5, 'No posterior', ha='center', va='center', transform=ax.transAxes, fontsize=8)
                continue
            
            ancestral_dist = np.loadtxt(path)
            ML_seq = np.argmax(ancestral_dist, axis=1)
            GT = GT_sequences[seq]
            L = len(GT)
            
            energy_fels = energy_of_msa(MSA, fields_, couplings)
            energy_shuffled = energy_of_msa(reshuffled_MSA, fields_, couplings)
            
            consensus_seq = consensus_dict.get((seq, mu), None)
            if consensus_seq is None:
                print(f"    ⚠️ Missing consensus for ({seq}, {mu})")
                ax.text(0.5, 0.5, 'No consensus', ha='center', va='center', transform=ax.transAxes, fontsize=8)
                continue
            
            GT_energy = energy(GT, couplings, fields_)
            ML_energy = energy(ML_seq, couplings, fields_)
            consensus_energy = energy(consensus_seq, couplings, fields_)
            
            fels_dists = [calculate_hamming_distance(s, GT)/L for s in MSA]
            shuffled_dists = [calculate_hamming_distance(s, GT)/L for s in reshuffled_MSA]
            d_consensus = calculate_hamming_distance(consensus_seq, GT)/L
            d_ML = calculate_hamming_distance(ML_seq, GT)/L
            
            # Site-independent ancestral samples
            # Slightly larger marker so visual size matches S^anc DCA diamonds
            ax.scatter(energy_fels, fels_dists, color=bayes_color, alpha=0.45, s=24,
                      marker='o', edgecolor='gray', linewidths=0.2, zorder=10, rasterized=True
            )
            
            # Potts samples - lighter color for less contrast
            ax.scatter(energy_shuffled, shuffled_dists, color=wt_color, alpha=0.5, s=20,
                      marker='D', edgecolor='gray', linewidths=0.2, zorder=20, rasterized=True
            )
            
            # Consensus
            ax.scatter(consensus_energy, d_consensus, color='#1f77b4', marker='^', s=80,
                      edgecolor='black', linewidths=1.0, label=r'$\bm{s}^{\mathrm{cons}}$', zorder=150)
            
            # ML - more visible with larger size and thicker edge
            ax.scatter(ML_energy, d_ML, color='yellow', marker='*', s=200,
                      edgecolor='black', linewidths=1.2, label=r'$\bm{s}^{\mathrm{MAP}}$', zorder=200)
            
            ax.axvline(GT_energy, color='green', linestyle='--', linewidth=1.5, label='GT Seq.')
            
            if i == 0:
                ax.set_title(rf"$\mu_{{\mathrm{{gen}}}}={mu:g}$", fontsize=21, pad=8)
            
            ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
    
    # Layout with minimal spacing - more room at top for titles
    left_margin = 0.12 if data_prefix == "DBD" else 0.10
    plt.subplots_adjust(left=left_margin, right=0.90, bottom=0.10, top=0.90, wspace=0.05, hspace=0.12)
    fig.canvas.draw()
    
    positions = np.array([ax.get_position().bounds for ax in axes.flatten()])
    xmin, ymin = positions[:, :2].min(axis=0)
    xmax = (positions[:, 0] + positions[:, 2]).max()
    ymax = (positions[:, 1] + positions[:, 3]).max()
    y_center = (ymin + ymax) / 2
    
    arrow_style = dict(arrowstyle='-|>', mutation_scale=20, linewidth=2, color='black', zorder=100)
    x_arrow_pos = xmax + 0.02
    fig.add_artist(FancyArrowPatch(posA=(x_arrow_pos, ymax - 0.02), posB=(x_arrow_pos, ymin + 0.02), 
                                   transform=fig.transFigure, **arrow_style))
    fig.text(x_arrow_pos + 0.035, y_center, r'Increasing $\bm{s}^{\mathrm{GT}}$ mutability', rotation='vertical',
             fontsize=22, fontstyle='italic', ha='center', va='center')
    
    # Global labels - bigger size (about 2x)
    fig.text(0.5, 0.03, r'$E_{\mathrm{DCA}}$', ha='center', fontsize=24)
    ylab_x = 0.01 if data_prefix == "DBD" else 0.03
    fig.text(ylab_x, 0.5, r'$d_\text{H}(\bm{s}_R,\bm{s}^{\mathrm{GT}})$', va='center', rotation='vertical', fontsize=24)
    
    # Legend
    sample_handles = [
        Line2D([0], [0], marker='o', color='lightgray', markerfacecolor='lightgray',
               markeredgecolor='black', markersize=7, linestyle='', alpha=0.8,
               label=r'$\mathcal{\bm{S}}_\text{anc}$'),
        Line2D([0], [0], marker='D', color='dimgray', markerfacecolor='dimgray',
               markeredgecolor='gray', markersize=7, linestyle='', alpha=0.9,
             label=r'$\mathcal{\bm{S}}_\text{anc}^\text{DCA}$')
    ]
    special_handles = [
        Line2D([0], [0], color='green', linestyle='--', linewidth=1.5, label=r'$\bm{s}^{\mathrm{GT}}$'),
        Line2D([0], [0], marker='*', color='yellow', markerfacecolor='yellow',
               markeredgecolor='black', markersize=12, linestyle='', label=r'$\bm{s}^{\mathrm{MAP}}$'),
        Line2D([0], [0], marker='^', color='#1f77b4', markerfacecolor='#1f77b4',
               markeredgecolor='black', markersize=8, linestyle='', label=r'$\bm{s}^{\mathrm{cons}}$'),
    ]
    handles = wt_legend_handles + special_handles + sample_handles
    labels = [str(h.get_label()) for h in handles]
    
    return fig, handles, labels


# ============================================================
# --- FIGURE 5: BOXPLOTS PER WT ---
# ============================================================

def create_figure5_boxplots(
    sequences: list,
    mu_values: list,
    M: int,
    T: float,
    msa_folder: str,
    posterior_folder: str,
    GT_sequences: dict,
    fields_: np.ndarray,
    couplings: np.ndarray,
    seq_to_label: dict,
    colors_dict: dict = None,
    energy_keep_pct: float = None,
    figsize: tuple = (16, 5),
    data_prefix: str = "DBD"
):
    """
    Create Figure 5: Single figure with boxplots for WT1, WT3, WT5 side by side.
    Color-coded by WT (darker for DCA, lighter for site-independent).
    
    Returns: list of (label, figure) tuples (single figure)
    """
    from matplotlib import cm
    
    # Filter to keep only WT1, WT3, WT5 (indices 0, 2, 4)
    wt_indices = [0, 2, 4]
    selected_wts = [sequences[i] for i in wt_indices if i < len(sequences)]
    
    # Build colors_dict if not provided
    if colors_dict is None:
        cmap = cm.get_cmap('plasma')
        color_values = np.linspace(0.1, 0.9, len(sequences))
        colors_dict = {seq: cmap(val) for seq, val in zip(sequences, color_values)}
    
    def lighten_color(color, factor=0.5):
        r, g, b = color[:3]
        return (r + (1-r)*factor, g + (1-g)*factor, b + (1-b)*factor)
    
    # Map internal keys to display labels
    label_map = {
        "yang_top10": "10 best DCA",
        "posterior_yang_top10": "10 best s-i"
    }
    
    # Symbols for each category
    symbol_map = {
        "10 best DCA": r"$\mathbf{D_{10}}$",
        "10 best s-i": r"$\mathbf{Y_{10}}$"
    }
    
    # Gather data for all WTs
    all_wt_data = {}
    
    for wt in selected_wts:
        wt_label = seq_to_label.get(wt, wt)
        print(f"  Processing boxplots for {wt_label} ({wt})...")
        GT = GT_sequences[wt]
        L = len(GT)
        wt_results = {}
        
        for mu in mu_values:
            try:
                # Load reshuffled MSA (DCA-informed)
                reshuffled_path = f"{msa_folder}/{wt}_mu={mu}_depth=None_shuffled_M={M}_T={T}"
                if not os.path.exists(reshuffled_path):
                    continue
                dca_reshuffled = np.loadtxt(reshuffled_path, dtype=int)
                
                # Load posterior-sampled MSA (site-independent)
                posterior_msa_path = f"{msa_folder}/{wt}_mu={mu}_depth=None_M={M}"
                dca_posterior_msa = None
                if os.path.exists(posterior_msa_path):
                    dca_posterior_msa = np.loadtxt(posterior_msa_path, dtype=int)
                
                # Compute energies for reshuffled MSA
                energies_reshuffled = energy_of_msa(dca_reshuffled, fields_, couplings)
                
                # Load posterior probability matrix
                posterior = None
                ppath = os.path.join(posterior_folder, f"{data_prefix}_{wt}_mu{mu}_ancestral_probability")
                if os.path.exists(ppath):
                    posterior = np.loadtxt(ppath)
                
                # Select sequences using yang scoring on RESHUFFLED MSA (with energy filter)
                selections = {}
                if posterior is not None:
                    try:
                        sel_idx = select_topN(dca_reshuffled, energies_reshuffled, posterior=posterior,
                                             scoring="yang", percentage=energy_keep_pct, topN=10)
                        selections["yang_top10"] = list(sel_idx)
                    except Exception:
                        selections["yang_top10"] = []
                else:
                    selections["yang_top10"] = []
                
                # Site-independent selection from POSTERIOR MSA (no energy filter)
                selections["posterior_yang_top10"] = []
                posterior_seqs_for_plot = None
                if dca_posterior_msa is not None and posterior is not None:
                    try:
                        energies_posterior = energy_of_msa(dca_posterior_msa, fields_, couplings)
                        sel_idx = select_topN(dca_posterior_msa, energies_posterior, posterior=posterior,
                                             scoring="yang", percentage=None, topN=10)
                        selections["posterior_yang_top10"] = list(sel_idx)
                        posterior_seqs_for_plot = dca_posterior_msa
                    except Exception:
                        pass
                
                # Load ML seq
                ML_seq = None
                if posterior is not None:
                    ML_seq = np.argmax(posterior, axis=1)
                
                wt_results[mu] = {
                    "all_dca_seqs": dca_reshuffled,
                    "posterior_dca_seqs": posterior_seqs_for_plot,
                    "selections": selections,
                    "ML_seq": ML_seq,
                    "GT": GT
                }
            except Exception as e:
                print(f"    Error at μ={mu}: {e}")
                continue
        
        all_wt_data[wt] = {"results": wt_results, "L": L, "GT": GT}
    
    # Create a 2-row layout: WT1 and WT3 in top row, WT5 centered in bottom
    # Taller figure (14, 12) to accommodate y-axis up to 1.0 without compression
    fig = plt.figure(figsize=(14, 12))
    
    # Use GridSpec for better control - 2 rows, 4 columns (to allow centering bottom plot)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 4, figure=fig, hspace=0.15, wspace=0.05)
    
    # WT1 and WT3 in top row (each spans 2 columns)
    ax1 = fig.add_subplot(gs[0, 0:2])  # WT1 top-left
    ax2 = fig.add_subplot(gs[0, 2:4], sharey=ax1)  # WT3 top-right
    # WT5 centered in bottom row (spans middle 2 columns = same width as top plots)
    ax3 = fig.add_subplot(gs[1, 1:3])  # WT5 centered, same width as A and B
    
    axes = [[ax1, ax2], [ax3, None]]  # For compatibility
    ax_list = [ax1, ax2, ax3]  # List of actual axes for WT1, WT3, WT5
    
    group_order = ["10 best DCA", "10 best s-i"]
    global_min, global_max = 1e9, -1e9
    
    for wt_idx, wt in enumerate(selected_wts):
        ax = ax_list[wt_idx]
        wt_label = seq_to_label.get(wt, wt)
        wt_color = colors_dict.get(wt, 'gray')
        light_color = lighten_color(wt_color, factor=0.5)
        
        if wt not in all_wt_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        wt_results = all_wt_data[wt]["results"]
        L = all_wt_data[wt]["L"]
        GT = all_wt_data[wt]["GT"]
        
        # Build dataframe for this WT
        rows = []
        for mu in mu_values:
            if mu not in wt_results:
                continue
            r = wt_results[mu]
            sel = r.get("selections", {})
            
            for label_key, pretty in label_map.items():
                idxs = sel.get(label_key, [])
                for idx in idxs:
                    if label_key == "posterior_yang_top10":
                        msa_array = r.get("posterior_dca_seqs")
                        if msa_array is None:
                            continue
                    else:
                        msa_array = r["all_dca_seqs"]
                    seq = msa_array[int(idx)]
                    dist = float(np.mean(seq != GT))
                    rows.append({
                        "mu": mu,
                        "Group": pretty,
                        "Distance": dist
                    })
                    global_min = min(global_min, dist)
                    global_max = max(global_max, dist)
            
            ML_seq = r.get("ML_seq", None)
            if ML_seq is not None:
                ml_dist = float(np.mean(ML_seq != GT))
                global_min = min(global_min, ml_dist)
                global_max = max(global_max, ml_dist)
        
        if len(rows) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        df = pd.DataFrame(rows)
        df["Group"] = pd.Categorical(df["Group"], categories=group_order, ordered=True)
        
        # Create x-axis position identifier
        df["x_id"] = df.apply(lambda r: f"{r['mu']}_{r['Group']}", axis=1)
        
        # Build x_order
        x_order = []
        for mu in mu_values:
            for g in group_order:
                x_id = f"{mu}_{g}"
                if x_id in df["x_id"].values:
                    x_order.append(x_id)
        
        df["x_id"] = pd.Categorical(df["x_id"], categories=x_order, ordered=True)
        
        # Build color palette: DCA = dark WT color, s-i = light WT color
        box_colors = []
        for x_id in x_order:
            group_name = x_id.split('_', 1)[1]
            if group_name == "10 best DCA":
                box_colors.append(wt_color)
            else:
                box_colors.append(light_color)
        
        # Create boxplot with WT-specific colors and narrower boxes
        bp = sns.boxplot(
            data=df, x="x_id", y="Distance",
            showfliers=False, ax=ax, palette=box_colors,
            width=0.6
        )
        
        # Remove default x tick labels
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', length=0)
        
        # Add symbols above each box
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        
        for i, x_id in enumerate(x_order):
            group_name = x_id.split('_', 1)[1]
            symbol = symbol_map.get(group_name, "")
            box_data = df[df["x_id"] == x_id]["Distance"]
            if len(box_data) > 0:
                box_max = box_data.max()
                symbol_y = box_max + 0.02 * y_range
                ax.text(i, symbol_y, symbol, ha='center', va='bottom', 
                       fontsize=18, fontweight='bold', clip_on=False)
        
        # Draw ML reference lines per μ
        for mu in mu_values:
            labels_for_mu = [lbl for lbl in x_order if lbl.startswith(f"{mu}_")]
            if len(labels_for_mu) == 0:
                continue
            
            ml_seq = wt_results.get(mu, {}).get("ML_seq", None)
            if ml_seq is None:
                continue
            
            ml_dist = float(np.mean(ml_seq != wt_results[mu]["GT"]))
            start = x_order.index(labels_for_mu[0])
            end = x_order.index(labels_for_mu[-1])
            
            ax.hlines(ml_dist, xmin=start - 0.3, xmax=end + 0.3,
                     colors="gray", linestyles="--", linewidth=1.2)
        
        # Set up custom x-axis with \mu_gen values
        mu_positions = []
        mu_labels = []
        
        for mu in mu_values:
            labels_for_mu = [lbl for lbl in x_order if lbl.startswith(f"{mu}_")]
            if len(labels_for_mu) == 0:
                continue
            
            start = x_order.index(labels_for_mu[0])
            end = x_order.index(labels_for_mu[-1])
            center = (start + end) / 2
            
            mu_positions.append(center)
            mu_labels.append(f"{mu:g}")
        
        # Set custom x-ticks
        ax.set_xticks(mu_positions)
        ax.set_xticklabels(mu_labels, fontsize=22)
        ax.set_xlabel("")  # Remove x_id label
        
        ax.set_title(wt_label, fontsize=20, fontweight='bold')
        ax.tick_params(axis="y", labelsize=22)
        ax.grid(alpha=0.3, linestyle="--", axis='y')
        ax.set_ylim(0, 1.0)  # Fixed y-axis to accommodate labels
        
        # Y-label and ticks: on left plots (WT1 and WT5), hide ticks on WT3
        if wt_idx == 0 or wt_idx == 2:  # WT1 (top-left) and WT5 (bottom center)
            ax.set_ylabel(r'$d_\text{H}(\bm{s}_R,\bm{s}^\text{GT})$', fontsize=22)
            ax.tick_params(axis='y', labelleft=True)  # Ensure y-tick labels are visible
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', labelleft=False)  # Hide y-tick labels for WT3
    
    # Common x label with larger font (higher position to avoid legend)
    fig.text(0.5, 0.05, r"$\mu_{\mathrm{gen}}$", ha='center', fontsize=22)
    
    # Tighten the layout and keep room for an in-figure legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Integrate legend directly in the first panel (lower-right)
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.2, label=r'$\bm{s}^{\text{MAP}}$')
    ]
    leg = ax1.legend(
        legend_elements,
        [elem.get_label() for elem in legend_elements],
        loc="lower right",
        fontsize=16,
        frameon=True,
        framealpha=0.95,
        edgecolor='black'
    )
    leg.get_frame().set_facecolor("#f0f0f0")

    return [("WT1_WT3_WT5", fig)]

# ============================================================
# --- FIGURE 5 BIS: 3×3 GRID (GT1, GT3, GT5) × (Hamming, pLDDT, RMSD) ---
# ============================================================

def create_figure5_bis(
    sequences,
    mu_values,
    M,
    T,
    msa_folder,
    posterior_folder,
    GT_sequences,
    fields_,
    couplings,
    seq_to_label,
    plddt_result_files=None,
    rmsd_result_files=None,
    extant_rmsd_files=None,
    extant_plddt_file=None,
    colors_dict=None,
    energy_keep_pct=None,
    natural_alignment=None,
    figsize=(20, 14),
    data_prefix="DBD",
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    from matplotlib import cm

    # ============================================================
    # -------------------- BASIC HELPERS --------------------------
    # ============================================================
    def lighten_color(color, factor=0.5):
        r, g, b = color[:3]
        return (r + (1 - r) * factor,
                g + (1 - g) * factor,
                b + (1 - b) * factor)

    def parse_candidate_df(df, wt_n, metric):
        df = df.set_index("Sequence_ID")
        out = {"DCA": [], "Yang": [], "MAP": None}

        for k in range(1, 11):
            if f"DCA{wt_n}_{k}" in df.index:
                out["DCA"].append(float(df.loc[f"DCA{wt_n}_{k}", metric]))
            if f"Yang{wt_n}_{k}" in df.index:
                out["Yang"].append(float(df.loc[f"Yang{wt_n}_{k}", metric]))

        if f"MAP{wt_n}" in df.index:
            out["MAP"] = float(df.loc[f"MAP{wt_n}", metric])

        return out

    def parse_rmsd_wt(df, wt_n):
        df = df.set_index("Sequence_ID")
        result = {}

        for mu in mu_values:
            out = {"DCA": [], "Yang": [], "MAP": None}

            for k in range(1, 11):
                sid = f"DCA{wt_n}_{k}_mu{mu}"
                if sid in df.index:
                    out["DCA"].append(float(df.loc[sid, "rmsd"]))

                sid = f"Yang{wt_n}_{k}_mu{mu}"
                if sid in df.index:
                    out["Yang"].append(float(df.loc[sid, "rmsd"]))

            sid = f"MAP{wt_n}_mu{mu}"
            if sid in df.index:
                out["MAP"] = float(df.loc[sid, "rmsd"])

            result[mu] = out

        return result

    # ============================================================
    # -------------------- PANEL DRAWING --------------------------
    # ============================================================
    def draw_panel(ax, mdata, wt_color, light_color,
               extant_vals, gt_val=None,
               y_lim=None, metric_type="hamming"):

        group_spacing = 3.0
        bar_offset = 0.35
        box_width = 0.5

        mu_positions = []
        x_lookup = {}

        for i, mu in enumerate(mu_values):
            center = i * group_spacing
            mu_positions.append(center)
            x_lookup[(mu, "DCA")]  = center - bar_offset
            x_lookup[(mu, "Yang")] = center + bar_offset

        extant_xpos = mu_positions[-1] + group_spacing * 1.5

        print("mu_values inside draw_panel:", mu_values)
        print("mu_positions:", [i * group_spacing for i, _ in enumerate(mu_values)])
        print("extant_xpos:", (len(mu_values)-1) * group_spacing + group_spacing * 1.5)

        # ── Draw ALL boxplots with manage_ticks=False so matplotlib
        #    never touches xlim automatically ──────────────────────
        for (mu, typ), xpos in x_lookup.items():
            vals = mdata.get(mu, {}).get(typ, [])
            if not vals:
                continue
            color = wt_color if typ == "DCA" else light_color
            ax.boxplot([vals],
                    positions=[xpos],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,          # <-- KEY: stops xlim/xtick clobbering
                    boxprops=dict(facecolor=color, edgecolor="black"),
                    medianprops=dict(color="black"),
                    whiskerprops=dict(color="black"),
                    capprops=dict(color="black"))

        if extant_vals:
            ax.boxplot([extant_vals],
                    positions=[extant_xpos],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,          # <-- same here
                    boxprops=dict(facecolor="#aaaaaa", edgecolor="black", alpha=0.7),
                    medianprops=dict(color="darkred"))

        # ── y limits first so ylim is correct for everything below ──
        if y_lim:
            ax.set_ylim(*y_lim)
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]

        # ── xlim: left edge flush with left whisker, right edge flush
        #    with extant right whisker, equal padding each side ──────
        margin = box_width * 1.5
        ax.set_xlim(
            mu_positions[0] - bar_offset - margin,
            extant_xpos     + bar_offset + margin,
        )

        # ── Alternating background bands ────────────────────────────
        for i, mu in enumerate(mu_values):
            if i % 2 == 1:
                c = mu_positions[i]
                ax.axvspan(c - group_spacing / 2,
                        c + group_spacing / 2,
                        color="gray", alpha=0.07, zorder=0)

        # ── MAP dashed lines ─────────────────────────────────────────
        for i, mu in enumerate(mu_values):
            m = mdata.get(mu, {}).get("MAP")
            if m is None:
                continue
            c = mu_positions[i]
            ax.hlines(m,
                    c - bar_offset - 0.2,
                    c + bar_offset + 0.2,
                    colors="gray", linestyles="--", linewidth=1)

        # ── GT dotted line across full mu range ─────────────────────
        if gt_val is not None and metric_type in ["plddt", "rmsd"]:
            ax.hlines(gt_val,
                    mu_positions[0]  - bar_offset,
                    mu_positions[-1] + bar_offset,
                    colors="red", linestyles=":", linewidth=1.2)

        # ── Extant "E" label ─────────────────────────────────────────
        if extant_vals:
            ax.text(extant_xpos, ylim[1] - 0.01 * y_range, "E",
                    ha="center", va="top", fontweight="bold", fontsize=10)

        # ── D10 / Y10 symbols above each box ────────────────────────
        for (mu, typ), xpos in x_lookup.items():
            vals = mdata.get(mu, {}).get(typ, [])
            if not vals:
                continue
            ymax = max(vals)
            y_text = ymax + 0.02 * y_range
            if y_text < ylim[1] - 0.05 * y_range:
                label = r"$\mathbf{D_{10}}$" if typ == "DCA" else r"$\mathbf{Y_{10}}$"
                ax.text(xpos, y_text, label,
                        ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

        # ── Ticks: one per mu group, correctly centered ──────────────
        ax.set_xticks(mu_positions)
        ax.set_xticklabels([f"{mu:g}" for mu in mu_values])

        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # ============================================================
        # -------------------- LOAD DATA ------------------------------
        # ============================================================
        EXTANT_COLOR = "#aaaaaa"

        extant_plddt = []
        if extant_plddt_file and os.path.exists(extant_plddt_file):
            extant_plddt = pd.read_csv(extant_plddt_file)["pLDDT"].dropna().tolist()

        extant_rmsd = {}
        if extant_rmsd_files:
            for wt, f in extant_rmsd_files.items():
                if os.path.exists(f):
                    extant_rmsd[wt] = pd.read_csv(f)["rmsd"].dropna().tolist()

        plddt_dfs = {}
        if plddt_result_files:
            for mu, f in plddt_result_files.items():
                if os.path.exists(f):
                    plddt_dfs[mu] = pd.read_csv(f)

        rmsd_wt_files = {
            i: f"final_betaLac/rmsd_results/candidate_pdbs_wt{i}_results.csv"
            for i in range(1, 6)
        }

        rmsd_dfs = {
            wt: pd.read_csv(f)
            for wt, f in rmsd_wt_files.items()
            if os.path.exists(f)
        }

        # ============================================================
        # -------------------- COLORS --------------------------------
        # ============================================================
        if colors_dict is None:
            cmap = cm.get_cmap("plasma")
            vals = np.linspace(0.1, 0.9, len(sequences))
            colors_dict = {s: cmap(v) for s, v in zip(sequences, vals)}

    # ============================================================
    # -------------------- FIGURE --------------------------------
    # ============================================================
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 4, width_ratios=[1,1,1,0.15])

    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]

    wt_indices = [0, 2, 4]
    wt_numbers = [1, 3, 5]

    for r, (idx, wt_n) in enumerate(zip(wt_indices, wt_numbers)):
        wt_seq = sequences[idx]
        wt_label = seq_to_label.get(wt_seq, f"WT{wt_n}")

        color = colors_dict[wt_seq]
        light = lighten_color(color)

        # Hamming
        hamming = {}  # reuse your original function if needed

        # pLDDT
        plddt = {
            mu: parse_candidate_df(plddt_dfs[mu], wt_n, "pLDDT")
            if mu in plddt_dfs else {}
            for mu in mu_values
        }

        gt_plddt = None
        if plddt_dfs:
            df = list(plddt_dfs.values())[0].set_index("Sequence_ID")
            if f"GT{wt_n}" in df.index:
                gt_plddt = df.loc[f"GT{wt_n}", "pLDDT"]

        # RMSD
        rmsd = parse_rmsd_wt(rmsd_dfs[wt_n], wt_n) if wt_n in rmsd_dfs else {}

        gt_rmsd = None
        if wt_n in rmsd_dfs:
            df = rmsd_dfs[wt_n].set_index("Sequence_ID")
            key = f"GT{wt_n}_mu1.0"
            if key in df.index:
                gt_rmsd = df.loc[key, "rmsd"]

        panels = [
            (hamming, [], None, (0,1), "hamming"),
            (plddt, extant_plddt, gt_plddt, (0.6,1.0), "plddt"),
            (rmsd, extant_rmsd.get(wt_n, []), gt_rmsd, (0,20), "rmsd"),
        ]

        for c, (data, ext, gt, ylim, mtype) in enumerate(panels):
            ax = axes[r][c]

            draw_panel(ax, data, color, light, ext,
                       gt_val=gt, y_lim=ylim, metric_type=mtype)

            if r == 0:
                ax.set_title(["Hamming", "pLDDT", "RMSD"][c],
                             fontsize=16, fontweight="bold")

        # row label
        lab_ax = fig.add_subplot(gs[r, 3])
        lab_ax.axis("off")
        lab_ax.text(0.5, 0.5, wt_label,
                    ha="center", va="center",
                    fontsize=18, fontweight="bold")

    # legend
    axes[0][0].legend([
        Line2D([0],[0],color="gray",linestyle="--"),
        Line2D([0],[0],color="red",linestyle=":"),
        Line2D([0],[0],marker="s",color="w",
               markerfacecolor="#aaaaaa", markersize=8)
    ], ["MAP", "Ground Truth", "Extant"], loc="upper right")

    plt.tight_layout()
    return fig

# ============================================================
# --- FIGURE 5 BIS TRANSPOSED: 3×3 GRID (d_H, pLDDT, RMSD) × (GT1, GT3, GT5) ---
# ============================================================

def create_figure5_bis_transposed(
    sequences: list,
    mu_values: list,
    M: int,
    T: float,
    msa_folder: str,
    posterior_folder: str,
    GT_sequences: dict,
    fields_: np.ndarray,
    couplings: np.ndarray,
    seq_to_label: dict,
    plddt_result_files: dict = None,
    rmsd_result_files: dict = None,
    extant_rmsd_files: dict = None,
    extant_plddt_file: str = None,
    colors_dict: dict = None,
    energy_keep_pct: float = None,
    natural_alignment: str = None,
    include_yang_likelihood_row: bool = False,
    figsize: tuple = (20, 14),
    data_prefix: str = "DBD",
    include_figure_legend: bool = True,
):
    """
    Transposed Figure 5 bis: a 3-row x 3-column grid.
 
    Rows   : Hamming distance to GT | pLDDT | RMSD
    Columns: GT1, GT3, GT5

    If include_yang_likelihood_row is True, append a 4th row with
    Yang likelihood under the ancestral posterior.
 
    Each panel shows grouped boxplots per mu_value with two groups:
        - "10 best DCA"  (DCA{n}_1..10)  - dark WT color
        - "10 best s-i"  (Yang{n}_1..10) - light WT color
    Plus a dashed horizontal line for MAP (the ML/MAP reference).
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    from matplotlib import cm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
 
    # Helpers (same as create_figure5_bis)
    def lighten_color(color, factor=0.5):
        r, g, b = color[:3]
        return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)
 
    def parse_mu_df(df, wt_n, metric_col):
        indexed = df.set_index("Sequence_ID")
        out = {"DCA": [], "Yang": [], "MAP": None}
        for k in range(1, 11):
            sid = f"DCA{wt_n}_{k}"
            if sid in indexed.index:
                out["DCA"].append(float(indexed.loc[sid, metric_col]))
        for k in range(1, 11):
            sid = f"Yang{wt_n}_{k}"
            if sid in indexed.index:
                out["Yang"].append(float(indexed.loc[sid, metric_col]))
        map_id = f"MAP{wt_n}"
        if map_id in indexed.index:
            out["MAP"] = float(indexed.loc[map_id, metric_col])
        return out
    
    def parse_rmsd_data_from_wt_file(df, wt_n, mu_values_list):
        indexed = df.set_index("Sequence_ID")
        result = {}
        for mu in mu_values_list:
            out = {"DCA": [], "Yang": [], "MAP": None}
            for k in range(1, 11):
                sid = f"DCA{wt_n}_{k}_mu{mu}"
                if sid in indexed.index:
                    out["DCA"].append(float(indexed.loc[sid, "rmsd"]))
            for k in range(1, 11):
                sid = f"Yang{wt_n}_{k}_mu{mu}"
                if sid in indexed.index:
                    out["Yang"].append(float(indexed.loc[sid, "rmsd"]))
            map_id = f"MAP{wt_n}_mu{mu}"
            if map_id in indexed.index:
                out["MAP"] = float(indexed.loc[map_id, "rmsd"])
            result[mu] = out
        return result
 
    def load_hamming_data(wt_seq, wt_n):
        GT = GT_sequences[wt_seq]
        result = {}
        for mu in mu_values:
            try:
                reshuffled_path = (
                    f"{msa_folder}/{wt_seq}_mu={mu}_depth=None"
                    f"_shuffled_M={M}_T={T}"
                )
                if not os.path.exists(reshuffled_path):
                    continue
                dca_reshuffled = np.loadtxt(reshuffled_path, dtype=int)
                posterior_msa_path = (
                    f"{msa_folder}/{wt_seq}_mu={mu}_depth=None_M={M}"
                )
                dca_posterior_msa = None
                if os.path.exists(posterior_msa_path):
                    dca_posterior_msa = np.loadtxt(posterior_msa_path, dtype=int)
                energies_reshuffled = energy_of_msa(dca_reshuffled, fields_, couplings)
                posterior = None
                ppath = os.path.join(
                    posterior_folder,
                    f"{data_prefix}_{wt_seq}_mu{mu}_ancestral_probability"
                )
                if os.path.exists(ppath):
                    posterior = np.loadtxt(ppath)
                dca_vals, yang_vals, map_val = [], [], None
                if posterior is not None:
                    try:
                        sel_idx = select_topN(
                            dca_reshuffled, energies_reshuffled,
                            posterior=posterior, scoring="yang",
                            percentage=energy_keep_pct, topN=10
                        )
                        for idx in sel_idx:
                            seq = dca_reshuffled[int(idx)]
                            dca_vals.append(float(np.mean(seq != GT)))
                    except Exception:
                        pass
                    if dca_posterior_msa is not None:
                        try:
                            energies_post = energy_of_msa(
                                dca_posterior_msa, fields_, couplings
                            )
                            sel_idx2 = select_topN(
                                dca_posterior_msa, energies_post,
                                posterior=posterior, scoring="yang",
                                percentage=None, topN=10
                            )
                            for idx in sel_idx2:
                                seq = dca_posterior_msa[int(idx)]
                                yang_vals.append(float(np.mean(seq != GT)))
                        except Exception:
                            pass
                    ML_seq = np.argmax(posterior, axis=1)
                    map_val = float(np.mean(ML_seq != GT))
                result[mu] = {"DCA": dca_vals, "Yang": yang_vals, "MAP": map_val}
            except Exception as e:
                print(f"    Hamming error at WT{wt_n} mu={mu}: {e}")
        return result

    def load_yang_likelihood_data(wt_seq, wt_n):
        """Return {mu: {"DCA": [...], "Yang": [...], "MAP": float|None}} for Yang likelihood."""
        result = {}
        for mu in mu_values:
            try:
                reshuffled_path = (
                    f"{msa_folder}/{wt_seq}_mu={mu}_depth=None"
                    f"_shuffled_M={M}_T={T}"
                )
                if not os.path.exists(reshuffled_path):
                    continue
                dca_reshuffled = np.loadtxt(reshuffled_path, dtype=int)

                posterior_msa_path = (
                    f"{msa_folder}/{wt_seq}_mu={mu}_depth=None_M={M}"
                )
                dca_posterior_msa = None
                if os.path.exists(posterior_msa_path):
                    dca_posterior_msa = np.loadtxt(posterior_msa_path, dtype=int)

                energies_reshuffled = energy_of_msa(dca_reshuffled, fields_, couplings)
                posterior = None
                ppath = os.path.join(
                    posterior_folder,
                    f"{data_prefix}_{wt_seq}_mu{mu}_ancestral_probability"
                )
                if os.path.exists(ppath):
                    posterior = np.loadtxt(ppath)

                dca_vals, yang_vals, map_val = [], [], None
                if posterior is not None:
                    try:
                        sel_idx = select_topN(
                            dca_reshuffled,
                            energies_reshuffled,
                            posterior=posterior,
                            scoring="yang",
                            percentage=energy_keep_pct,
                            topN=10,
                        )
                        for idx in sel_idx:
                            seq = dca_reshuffled[int(idx)]
                            dca_vals.append(float(yang_score(seq, posterior)))
                    except Exception:
                        pass

                    if dca_posterior_msa is not None:
                        try:
                            energies_post = energy_of_msa(dca_posterior_msa, fields_, couplings)
                            sel_idx2 = select_topN(
                                dca_posterior_msa,
                                energies_post,
                                posterior=posterior,
                                scoring="yang",
                                percentage=None,
                                topN=10,
                            )
                            for idx in sel_idx2:
                                seq = dca_posterior_msa[int(idx)]
                                yang_vals.append(float(yang_score(seq, posterior)))
                        except Exception:
                            pass

                    ML_seq = np.argmax(posterior, axis=1)
                    map_val = float(yang_score(ML_seq, posterior))

                result[mu] = {"DCA": dca_vals, "Yang": yang_vals, "MAP": map_val}
            except Exception as e:
                print(f"    Yang-likelihood error at WT{wt_n} mu={mu}: {e}")
        return result
    
    def draw_panel(ax, mdata, dca_color, yang_color, extant_vals, y_fixed_max=None, gt_val=None, metric_type="hamming"):
        """
        Draw boxplots with single marker above D10 (diamond) and Y10 (circle).
        - One diamond marker centered above each D10 box
        - One circle marker centered above each Y10 box
        """
        dca_box_width = 0.32
        yang_box_width = 0.32
        pair_spacing = 0.5  # Gap between DCA box and Yang box
        
        # Build position mapping for all mu values
        mu_list = sorted([m for m in mu_values if m in mdata])
        if not mu_list:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13)
            return
        
        # Position each mu group: x = 0, 1, 2, 3, 4... for each mu value
        # Within each group: DCA at x - spacing/2, Yang at x + spacing/2
        mu_positions_dict = {mu: idx for idx, mu in enumerate(mu_list)}
        
        # --- BACKGROUND STYLING with alternating grey/white ---
        for idx, mu in enumerate(mu_list):
            x_base = idx
            # Background exactly between two vertical dividers
            left = x_base - 0.5
            right = x_base + 0.5
            
            # Alternate between light grey and white for better visual separation
            if idx % 2 == 0:
                bg_color = "#d6d6d6"  # Visible light grey
                alpha_val = 0.5
            else:
                bg_color = "#ffffff"  # White (no visible background)
                alpha_val = 0.0
            
            ax.axvspan(left, right, color=bg_color, alpha=alpha_val, zorder=0, edgecolor="none")
            
            # Divider between mu groups
            if idx < len(mu_list) - 1:
                next_x = idx + 1
                divider = (x_base + next_x) / 2
                ax.axvline(divider, color="#c0c0c0", linewidth=1.2, zorder=0, alpha=0.9)
        
        # Extant section (clearly separated)
        if len(extant_vals) > 0:
            extant_x = len(mu_list) + 1.0
            ax.axvspan(extant_x - 0.5, extant_x + 0.5, color="#f5f5f5", alpha=0.9, zorder=0, edgecolor="none")
            ax.axvline(extant_x - 0.55, color="#cccccc", linewidth=1.3, zorder=0, alpha=0.9)
        
        # --- DRAW BOXPLOTS WITH SINGLE MARKER ---
        for mu in mu_list:
            x_base = mu_positions_dict[mu]
            
            # DCA (10 best)
            dca_vals = mdata[mu].get("DCA", [])
            if len(dca_vals) > 0:
                dca_x = x_base - pair_spacing / 2
                
                # Boxplot
                bp_dca = ax.boxplot([dca_vals], positions=[dca_x], widths=dca_box_width,
                                   patch_artist=True, showfliers=False,
                                   boxprops=dict(facecolor=dca_color, edgecolor="black", linewidth=1.3, alpha=0.8),
                                   medianprops=dict(color="darkred", linewidth=2.1),
                                   whiskerprops=dict(color="black", linewidth=1.1),
                                   capprops=dict(color="black", linewidth=1.1))
                
                # Text label 'D' in orange above the box
                box_top = np.max(dca_vals)
                marker_y = box_top * 1.05  # Slightly above the max value
                ax.text(dca_x, marker_y, "D", fontsize=18, fontweight="bold", 
                       color="#FF8C00", ha="center", va="bottom", zorder=5)
            
            # YANG (10 best s-i)
            yang_vals = mdata[mu].get("Yang", [])
            if len(yang_vals) > 0:
                yang_x = x_base + pair_spacing / 2
                
                # Boxplot
                bp_yang = ax.boxplot([yang_vals], positions=[yang_x], widths=yang_box_width,
                                    patch_artist=True, showfliers=False,
                                    boxprops=dict(facecolor=yang_color, edgecolor="black", linewidth=1.3, alpha=0.8),
                                    medianprops=dict(color="darkred", linewidth=2.1),
                                    whiskerprops=dict(color="black", linewidth=1.1),
                                    capprops=dict(color="black", linewidth=1.1))
                
                # Text label 'Y' in blue above the box
                box_top = np.max(yang_vals)
                marker_y = box_top * 1.05  # Slightly above the max value
                ax.text(yang_x, marker_y, "Y", fontsize=18, fontweight="bold", 
                       color=yang_color, ha="center", va="bottom", zorder=5)
        
        # --- EXTANT DATA (if present) ---
        if len(extant_vals) > 0:
            extant_x = len(mu_list) + 1.0
            
            # Boxplot
            bp_ext = ax.boxplot([extant_vals], positions=[extant_x], widths=0.35,
                               patch_artist=True, showfliers=False,
                               boxprops=dict(facecolor="#999999", edgecolor="black", linewidth=1.3, alpha=0.75),
                               medianprops=dict(color="darkred", linewidth=2.1),
                               whiskerprops=dict(color="black", linewidth=1.1),
                               capprops=dict(color="black", linewidth=1.1))
        
        # --- MAP REFERENCE LINES ---
        for mu in mu_list:
            map_val = mdata.get(mu, {}).get("MAP", None)
            if map_val is not None:
                x_base = mu_positions_dict[mu]
                x_left = x_base - pair_spacing/2 - 0.3
                x_right = x_base + pair_spacing/2 + 0.3
                ax.hlines(map_val, x_left, x_right, colors="black", linestyles="--", 
                         linewidth=2.0, zorder=2, alpha=1.0)
        
        # --- GT REFERENCE LINE ---
        if gt_val is not None and metric_type in ["plddt", "rmsd"]:
            if len(mu_list) > 0:
                x_min = -0.7
                x_max = len(mu_list) + 1.7 if len(extant_vals) > 0 else len(mu_list) + 0.7
                ax.hlines(gt_val, x_min, x_max, colors="red", linestyles=":", 
                         linewidth=2.3, zorder=2, alpha=0.9)
        
        if y_fixed_max is not None:
            ax.set_ylim(0, y_fixed_max)
        
        # --- X-AXIS TICKS (shown on ALL rows) ---
        tick_positions = [mu_positions_dict[mu] for mu in mu_list]
        tick_labels = [f"{mu:g}" for mu in mu_list]
        
        if len(extant_vals) > 0:
            tick_positions.append(len(mu_list) + 1.0)
            tick_labels.append(r"$\mathbf{E}$")
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=19)
        ax.set_xlim(-0.8, len(mu_list) + 1.5 if len(extant_vals) > 0 else len(mu_list) + 0.8)
        
        # --- GRID & STYLING ---
        ax.grid(alpha=0.2, linestyle="-", axis="y", linewidth=0.5, which="major")
        ax.set_axisbelow(True)
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.1)
        ax.spines["bottom"].set_linewidth(1.1)
    
    # Load data
    plddt_dfs = {}
    if plddt_result_files:
        for mu, fpath in plddt_result_files.items():
            if os.path.exists(fpath):
                plddt_dfs[mu] = pd.read_csv(fpath)
                if mu == 5.5:
                    print(f"DEBUG: Successfully loaded plddt data for mu=5.5 from {fpath}")
            else:
                if mu == 5.5:
                    print(f"DEBUG: plddt file NOT found for mu=5.5: {fpath}")
    
    rmsd_dfs_per_wt = {}
    # Use provided rmsd_result_files parameter instead of hardcoded paths
    if rmsd_result_files:
        for wt_n, fpath in rmsd_result_files.items():
            if os.path.exists(fpath):
                rmsd_dfs_per_wt[wt_n] = pd.read_csv(fpath)
            else:
                print(f"    WARNING: RMSD file not found for WT{wt_n}: {fpath}")
    
    extant_plddt_vals = []
    if extant_plddt_file and os.path.exists(extant_plddt_file):
        df_ext_p = pd.read_csv(extant_plddt_file)
        extant_plddt_vals = df_ext_p["pLDDT"].dropna().tolist()
    
    extant_rmsd_vals = {}
    if extant_rmsd_files:
        for wt_n, fpath in extant_rmsd_files.items():
            if os.path.exists(fpath):
                df_ext_r = pd.read_csv(fpath)
                extant_rmsd_vals[wt_n] = df_ext_r["rmsd"].dropna().tolist()
    
    # Load natural sequences from FASTA file
    natural_sequences = []
    if natural_alignment and os.path.exists(natural_alignment):
        natural_sequences = read_fasta1(natural_alignment)
    
    if colors_dict is None:
        cmap_obj = cm.get_cmap("plasma")
        color_values = np.linspace(0.1, 0.9, len(sequences))
        colors_dict = {seq: cmap_obj(v) for seq, v in zip(sequences, color_values)}
    
    # Setup figure: rows (metrics) x 3 columns (GTs)
    metric_types = ["hamming", "plddt", "rmsd"]
    row_titles = [r"$d_\text{H}(\bm{s}_R,\bm{s}^\text{GT})$", "pLDDT", r"RMSD ($\AA$)"]
    if include_yang_likelihood_row:
        metric_types.append("yang_likelihood")
        row_titles.append("Yang likelihood")
    col_titles = ["GT1", "GT3", "GT5"]

    n_rows = len(metric_types)
    local_figsize = figsize
    if include_yang_likelihood_row and figsize == (20, 14):
        local_figsize = (20, 17)

    fig = plt.figure(figsize=local_figsize)
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.20, wspace=0.08)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(n_rows)]
    dca_color = "#D55E00"
    yang_color = "#0072B2"
    
    # Share y-axis within each row so scale is consistent across GTs
    for r in range(n_rows):
        for c in range(1, 3):
            axes[r][c].sharey(axes[r][0])
    
    # Process each metric (row) and WT (column)
    for row_idx, metric_type in enumerate(metric_types):
        for col_idx, wt_n in enumerate([1, 3, 5]):
            wt_seq_idx = wt_n - 1
            if wt_seq_idx >= len(sequences):
                continue
            
            wt_seq = sequences[wt_seq_idx]
            ax = axes[row_idx][col_idx]
            
            # Load appropriate data
            if metric_type == "hamming":
                mdata = load_hamming_data(wt_seq, wt_n)
                # For extant hamming distance: compute distance from natural sequences to GT
                ext_vals = []
                if GT_sequences and natural_sequences:
                    # Try to find GT sequence by wt_seq first, then by wt_n
                    gt_seq = GT_sequences.get(wt_seq)
                    if gt_seq is None:
                        gt_seq = GT_sequences.get(wt_n)
                    if gt_seq is not None:
                        gt_len = len(gt_seq)
                        ext_vals = [calculate_hamming_distance(nat_seq, gt_seq) / gt_len for nat_seq in natural_sequences]
                y_fixed = 1.0
                gt_val = None
            elif metric_type == "plddt":
                mdata = {
                    mu: (parse_mu_df(plddt_dfs[mu], wt_n, "pLDDT") if mu in plddt_dfs
                         else {"DCA": [], "Yang": [], "MAP": None})
                    for mu in mu_values
                }
                ext_vals = extant_plddt_vals
                y_fixed = 1.0
                gt_val = None
                if plddt_dfs and 1 in plddt_dfs:
                    indexed = plddt_dfs[1].set_index("Sequence_ID")
                    gt_id = f"GT{wt_n}"
                    if gt_id in indexed.index:
                        gt_val = float(indexed.loc[gt_id, "pLDDT"])
            elif metric_type == "rmsd":
                mdata = {}
                if wt_n in rmsd_dfs_per_wt:
                    mdata = parse_rmsd_data_from_wt_file(
                        rmsd_dfs_per_wt[wt_n], wt_n, mu_values
                    )
                ext_vals = extant_rmsd_vals.get(wt_n, [])
                y_fixed = 20.0
                gt_val = None
                if wt_n in rmsd_dfs_per_wt:
                    indexed = rmsd_dfs_per_wt[wt_n].set_index("Sequence_ID")
                    gt_id = f"GT{wt_n}_mu1.0"
                    if gt_id in indexed.index:
                        gt_val = float(indexed.loc[gt_id, "rmsd"])
            elif metric_type == "yang_likelihood":
                mdata = load_yang_likelihood_data(wt_seq, wt_n)
                ext_vals = []
                y_fixed = 1.0
                gt_val = None
            else:
                continue
            
            draw_panel(ax, mdata, dca_color, yang_color, extant_vals=ext_vals,
                      y_fixed_max=y_fixed, gt_val=gt_val, metric_type=metric_type)
            
            # Y-axis label on first column only
            if col_idx == 0:
                ax.set_ylabel(row_titles[row_idx], fontsize=25, fontweight="bold", labelpad=8)
                ax.tick_params(axis="y", labelsize=21, length=5, width=1.0)
            else:
                # Hide y-ticks for non-first columns
                ax.tick_params(axis="y", labelsize=0, length=0)
            
            # Adjust y-axis limits for pLDDT
            if metric_type == "plddt":
                ax.set_ylim(0.6, 1.0)
            
            # Column titles on first row (bold)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=27, fontweight="bold", pad=14)
            
            # Show x-axis ticks on ALL rows
            ax.tick_params(axis="x", labelsize=20)
            if row_idx == len(metric_types) - 1:
                ax.set_xlabel(r"$\mu_{\mathrm{gen}}$", fontsize=24, fontweight="bold", labelpad=12)
    
    # Legend - updated to show D and Y text labels
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#FF8C00", 
               markersize=12, label=r"$D_{10}$ (orange)", linewidth=0),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=yang_color, 
               markersize=12, label=r"$Y_{10}$ (blue)", linewidth=0),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2.0,
               label=r"$\bm{s}^{\text{MAP}}$"),
        Line2D([0], [0], color="red", linestyle=":", linewidth=2.5,
               label="Ground Truth (GT)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#999999", 
               markeredgecolor="black", markersize=8, label=r"Extant ($\mathbf{E}$)"),
    ]
    if include_figure_legend:
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=5,
            fontsize=16,
            frameon=True,
            framealpha=0.98,
            edgecolor="black",
            fancybox=False,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.945])
    else:
        plt.tight_layout(rect=[0, 0, 1, 1])
    
    return fig


def save_figure5_bis_transposed_legend(out_path: str, use_latex: bool = True):
    """Save the cleaned Figure 5 bis legend as a standalone PDF."""
    if use_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    dca_color = "#D55E00"
    yang_color = "#0072B2"
    legend_elements = [
        Patch(facecolor=dca_color, edgecolor="black", label=r"$D_{10}$"),
        Patch(facecolor=yang_color, edgecolor="black", label=r"$Y_{10}$"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label=r"$\bm{s}^{\text{MAP}}$"),
        Line2D([0], [0], color="red", linestyle=":", linewidth=2.5, label="Ground Truth"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#aaaaaa", markersize=8, label="Extant (E)"),
    ]

    fig_leg = plt.figure(figsize=(16, 2.2))
    fig_leg.legend(
        handles=legend_elements,
        loc="center",
        ncol=5,
        fontsize=18,
        frameon=True,
        framealpha=0.98,
        edgecolor="black",
        columnspacing=1.6,
        handlelength=2.2,
    )
    plt.axis("off")
    fig_leg.savefig(out_path, bbox_inches="tight")
    plt.close(fig_leg)


def create_supplementary_figure5_bis_legacy(
    sequences: list,
    mu_values: list,
    M: int,
    T: float,
    msa_folder: str,
    posterior_folder: str,
    GT_sequences: dict,
    fields_: np.ndarray,
    couplings: np.ndarray,
    seq_to_label: dict,
    plddt_result_files: dict = None,
    rmsd_result_files: dict = None,
    extant_rmsd_files: dict = None,
    extant_plddt_file: str = None,
    colors_dict: dict = None,
    energy_keep_pct: float = None,
    natural_alignment: str = None,
    include_yang_likelihood_row: bool = False,
    figsize: tuple = (20, 14),
    data_prefix: str = "DBD",
):
    """
    Legacy Supplementary Figure 5 bis layout restored from the original version.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    from matplotlib import cm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    def lighten_color(color, factor=0.5):
        r, g, b = color[:3]
        return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)

    def parse_mu_df(df, wt_n, metric_col):
        indexed = df.set_index("Sequence_ID")
        out = {"DCA": [], "Yang": [], "MAP": None}
        for k in range(1, 11):
            sid = f"DCA{wt_n}_{k}"
            if sid in indexed.index:
                out["DCA"].append(float(indexed.loc[sid, metric_col]))
        for k in range(1, 11):
            sid = f"Yang{wt_n}_{k}"
            if sid in indexed.index:
                out["Yang"].append(float(indexed.loc[sid, metric_col]))
        map_id = f"MAP{wt_n}"
        if map_id in indexed.index:
            out["MAP"] = float(indexed.loc[map_id, metric_col])
        return out

    def parse_rmsd_data_from_wt_file(df, wt_n, mu_values_list):
        indexed = df.set_index("Sequence_ID")
        result = {}
        for mu in mu_values_list:
            out = {"DCA": [], "Yang": [], "MAP": None}
            for k in range(1, 11):
                sid = f"DCA{wt_n}_{k}_mu{mu}"
                if sid in indexed.index:
                    out["DCA"].append(float(indexed.loc[sid, "rmsd"]))
            for k in range(1, 11):
                sid = f"Yang{wt_n}_{k}_mu{mu}"
                if sid in indexed.index:
                    out["Yang"].append(float(indexed.loc[sid, "rmsd"]))
            map_id = f"MAP{wt_n}_mu{mu}"
            if map_id in indexed.index:
                out["MAP"] = float(indexed.loc[map_id, "rmsd"])
            result[mu] = out
        return result

    def load_hamming_data(wt_seq, wt_n):
        GT = GT_sequences[wt_seq]
        result = {}
        for mu in mu_values:
            try:
                reshuffled_path = (
                    f"{msa_folder}/{wt_seq}_mu={mu}_depth=None"
                    f"_shuffled_M={M}_T={T}"
                )
                if not os.path.exists(reshuffled_path):
                    continue
                dca_reshuffled = np.loadtxt(reshuffled_path, dtype=int)
                posterior_msa_path = (
                    f"{msa_folder}/{wt_seq}_mu={mu}_depth=None_M={M}"
                )
                dca_posterior_msa = None
                if os.path.exists(posterior_msa_path):
                    dca_posterior_msa = np.loadtxt(posterior_msa_path, dtype=int)
                energies_reshuffled = energy_of_msa(dca_reshuffled, fields_, couplings)
                posterior = None
                ppath = os.path.join(
                    posterior_folder,
                    f"{data_prefix}_{wt_seq}_mu{mu}_ancestral_probability"
                )
                if os.path.exists(ppath):
                    posterior = np.loadtxt(ppath)
                dca_vals, yang_vals, map_val = [], [], None
                if posterior is not None:
                    try:
                        sel_idx = select_topN(
                            dca_reshuffled, energies_reshuffled,
                            posterior=posterior, scoring="yang",
                            percentage=energy_keep_pct, topN=10
                        )
                        for idx in sel_idx:
                            seq = dca_reshuffled[int(idx)]
                            dca_vals.append(float(np.mean(seq != GT)))
                    except Exception:
                        pass
                    if dca_posterior_msa is not None:
                        try:
                            energies_post = energy_of_msa(
                                dca_posterior_msa, fields_, couplings
                            )
                            sel_idx2 = select_topN(
                                dca_posterior_msa, energies_post,
                                posterior=posterior, scoring="yang",
                                percentage=None, topN=10
                            )
                            for idx in sel_idx2:
                                seq = dca_posterior_msa[int(idx)]
                                yang_vals.append(float(np.mean(seq != GT)))
                        except Exception:
                            pass
                    ML_seq = np.argmax(posterior, axis=1)
                    map_val = float(np.mean(ML_seq != GT))
                result[mu] = {"DCA": dca_vals, "Yang": yang_vals, "MAP": map_val}
            except Exception as e:
                print(f"    Hamming error at WT{wt_n} mu={mu}: {e}")
        return result

    def load_yang_likelihood_data(wt_seq, wt_n):
        result = {}
        for mu in mu_values:
            try:
                reshuffled_path = (
                    f"{msa_folder}/{wt_seq}_mu={mu}_depth=None"
                    f"_shuffled_M={M}_T={T}"
                )
                if not os.path.exists(reshuffled_path):
                    continue
                dca_reshuffled = np.loadtxt(reshuffled_path, dtype=int)
                posterior_msa_path = (
                    f"{msa_folder}/{wt_seq}_mu={mu}_depth=None_M={M}"
                )
                dca_posterior_msa = None
                if os.path.exists(posterior_msa_path):
                    dca_posterior_msa = np.loadtxt(posterior_msa_path, dtype=int)
                energies_reshuffled = energy_of_msa(dca_reshuffled, fields_, couplings)
                posterior = None
                ppath = os.path.join(
                    posterior_folder,
                    f"{data_prefix}_{wt_seq}_mu{mu}_ancestral_probability"
                )
                if os.path.exists(ppath):
                    posterior = np.loadtxt(ppath)
                dca_vals, yang_vals, map_val = [], [], None
                if posterior is not None:
                    try:
                        sel_idx = select_topN(
                            dca_reshuffled, energies_reshuffled,
                            posterior=posterior, scoring="yang",
                            percentage=energy_keep_pct, topN=10,
                        )
                        for idx in sel_idx:
                            seq = dca_reshuffled[int(idx)]
                            dca_vals.append(float(yang_score(seq, posterior)))
                    except Exception:
                        pass
                    if dca_posterior_msa is not None:
                        try:
                            energies_post = energy_of_msa(dca_posterior_msa, fields_, couplings)
                            sel_idx2 = select_topN(
                                dca_posterior_msa, energies_post,
                                posterior=posterior, scoring="yang",
                                percentage=None, topN=10,
                            )
                            for idx in sel_idx2:
                                seq = dca_posterior_msa[int(idx)]
                                yang_vals.append(float(yang_score(seq, posterior)))
                        except Exception:
                            pass
                    ML_seq = np.argmax(posterior, axis=1)
                    map_val = float(yang_score(ML_seq, posterior))
                result[mu] = {"DCA": dca_vals, "Yang": yang_vals, "MAP": map_val}
            except Exception as e:
                print(f"    Yang-likelihood error at WT{wt_n} mu={mu}: {e}")
        return result

    def draw_panel(ax, mdata, wt_color, light_color, extant_vals, y_fixed_max=None, gt_val=None, metric_type="hamming"):
        extant_label = "Extant"
        has_extant = len(extant_vals) > 0
        rows = []
        x_pos = 0
        mu_positions = []
        for mu in mu_values:
            if mu not in mdata:
                continue
            mu_start_pos = x_pos
            for gkey, glabel in [("DCA", "10 best DCA"), ("Yang", "10 best s-i")]:
                for val in mdata[mu].get(gkey, []):
                    rows.append({"x_id": f"{mu}_{glabel}", "Group": glabel, "Value": val, "x_pos": x_pos})
                x_pos += 1
            #mu_center = (mu_start_pos + x_pos - 1) / 2\
            mu_center = (mu_start_pos + x_pos - 1) / 2 - 0.5
            mu_positions.append(mu_center)
            x_pos += 0.5

        if has_extant:
            extant_x_pos = x_pos
            for val in extant_vals:
                rows.append({"x_id": "Extant", "Group": extant_label, "Value": val, "x_pos": extant_x_pos})

        if not rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=13)
            return

        df = pd.DataFrame(rows)
        for _, group_name in enumerate(df["x_id"].unique()):
            group_data = df[df["x_id"] == group_name]
            pos = group_data["x_pos"].iloc[0]
            if "10 best DCA" in group_name:
                color = wt_color
            elif "10 best s-i" in group_name:
                color = light_color
            else:
                color = "#aaaaaa"
            ax.boxplot([group_data["Value"].values], positions=[pos], widths=0.6,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=color, edgecolor="black"),
                       medianprops=dict(color="black"),
                       whiskerprops=dict(color="black"),
                       capprops=dict(color="black"))

        for mu_idx, mu in enumerate(mu_values):
            map_val = mdata.get(mu, {}).get("MAP", None)
            if map_val is None:
                continue
            mu_center = mu_positions[mu_idx]
            ax.hlines(map_val, xmin=mu_center - 0.5, xmax=mu_center + 0.5,
                      colors="gray", linestyles="--", linewidth=1.2)

        if gt_val is not None and metric_type in ["plddt", "rmsd"]:
            x_min = mu_positions[0] - 0.7
            x_max = mu_positions[-1] + 0.7
            ax.hlines(gt_val, xmin=x_min, xmax=x_max, colors="red", linestyles=":", linewidth=2.0)

        if y_fixed_max is not None:
            ax.set_ylim(0, y_fixed_max)
        ylim_cur = ax.get_ylim()
        y_range = ylim_cur[1] - ylim_cur[0]

        symbol_map_local = {
            "10 best DCA": r"$\mathbf{D_{10}}$",
            "10 best s-i": r"$\mathbf{Y_{10}}$",
            "Extant": r"$\mathbf{E}$",
        }
        for x_id in df["x_id"].unique():
            group_data = df[df["x_id"] == x_id]
            group = (
                "10 best DCA" if "10 best DCA" in x_id
                else "10 best s-i" if "10 best s-i" in x_id
                else "Extant"
            )
            if len(group_data) > 0:
                max_val = group_data["Value"].max()
                xpos_local = group_data["x_pos"].iloc[0]
                label_y = max_val + 0.02 * y_range
                if label_y < ylim_cur[1] * 0.95:
                    ax.text(xpos_local, label_y, symbol_map_local[group],
                            ha="center", va="bottom", fontsize=13, fontweight="bold", clip_on=True)

        all_positions = mu_positions.copy()
        all_labels = [f"{mu:g}" for mu in mu_values]
        if has_extant and len(extant_vals) > 0:
            extant_x_pos = (mu_positions[-1] + 1.5) if mu_positions else 1.0
            all_positions.append(extant_x_pos)
            all_labels.append("E")
        ax.set_xticks(all_positions)
        ax.set_xticklabels(all_labels, fontsize=14)
        ax.grid(alpha=0.3, linestyle="--", axis="y")

    plddt_dfs = {}
    if plddt_result_files:
        for mu, fpath in plddt_result_files.items():
            if os.path.exists(fpath):
                plddt_dfs[mu] = pd.read_csv(fpath)

    rmsd_dfs_per_wt = {}
    # Use provided rmsd_result_files parameter instead of hardcoded paths
    if rmsd_result_files:
        for wt_n, fpath in rmsd_result_files.items():
            if os.path.exists(fpath):
                rmsd_dfs_per_wt[wt_n] = pd.read_csv(fpath)
            else:
                print(f"    WARNING: RMSD file not found for WT{wt_n}: {fpath}")

    extant_plddt_vals = []
    if extant_plddt_file and os.path.exists(extant_plddt_file):
        df_ext_p = pd.read_csv(extant_plddt_file)
        extant_plddt_vals = df_ext_p["pLDDT"].dropna().tolist()

    extant_rmsd_vals = {}
    if extant_rmsd_files:
        for wt_n, fpath in extant_rmsd_files.items():
            if os.path.exists(fpath):
                df_ext_r = pd.read_csv(fpath)
                extant_rmsd_vals[wt_n] = df_ext_r["rmsd"].dropna().tolist()

    natural_sequences = []
    if natural_alignment and os.path.exists(natural_alignment):
        natural_sequences = read_fasta1(natural_alignment)

    if colors_dict is None:
        cmap_obj = cm.get_cmap("plasma")
        color_values = np.linspace(0.1, 0.9, len(sequences))
        colors_dict = {seq: cmap_obj(v) for seq, v in zip(sequences, color_values)}

    metric_types = ["hamming", "plddt", "rmsd"]
    row_titles = [r"$d_\text{H}(\bm{s}_R,\bm{s}^\text{GT})$", "pLDDT", r"RMSD ($\AA$)"]
    if include_yang_likelihood_row:
        metric_types.append("yang_likelihood")
        row_titles.append("Yang likelihood")
    col_titles = ["GT1", "GT3", "GT5"]

    n_rows = len(metric_types)
    local_figsize = figsize
    if include_yang_likelihood_row and figsize == (20, 14):
        local_figsize = (20, 17)

    fig = plt.figure(figsize=local_figsize)
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.15, wspace=0.02)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(n_rows)]

    for r in range(n_rows):
        for c in range(1, 3):
            axes[r][c].sharey(axes[r][0])

    for row_idx, metric_type in enumerate(metric_types):
        for col_idx, wt_n in enumerate([1, 3, 5]):
            wt_seq_idx = wt_n - 1
            if wt_seq_idx >= len(sequences):
                continue
            wt_seq = sequences[wt_seq_idx]
            wt_color = colors_dict.get(wt_seq, "gray")
            light_color = lighten_color(wt_color, factor=0.5)
            ax = axes[row_idx][col_idx]

            if metric_type == "hamming":
                mdata = load_hamming_data(wt_seq, wt_n)
                ext_vals = []
                if GT_sequences and natural_sequences:
                    gt_seq = GT_sequences.get(wt_seq)
                    if gt_seq is None:
                        gt_seq = GT_sequences.get(wt_n)
                    if gt_seq is not None:
                        gt_len = len(gt_seq)
                        ext_vals = [calculate_hamming_distance(nat_seq, gt_seq) / gt_len for nat_seq in natural_sequences]
                y_fixed = 1.0
                gt_val = None
            elif metric_type == "plddt":
                mdata = {
                    mu: (parse_mu_df(plddt_dfs[mu], wt_n, "pLDDT") if mu in plddt_dfs
                         else {"DCA": [], "Yang": [], "MAP": None})
                    for mu in mu_values
                }
                ext_vals = extant_plddt_vals
                y_fixed = 1.0
                gt_val = None
                if plddt_dfs and 1 in plddt_dfs:
                    indexed = plddt_dfs[1].set_index("Sequence_ID")
                    gt_id = f"GT{wt_n}"
                    if gt_id in indexed.index:
                        gt_val = float(indexed.loc[gt_id, "pLDDT"])
            elif metric_type == "rmsd":
                mdata = {}
                if wt_n in rmsd_dfs_per_wt:
                    mdata = parse_rmsd_data_from_wt_file(rmsd_dfs_per_wt[wt_n], wt_n, mu_values)
                ext_vals = extant_rmsd_vals.get(wt_n, [])
                y_fixed = 20.0
                gt_val = None
                if wt_n in rmsd_dfs_per_wt:
                    indexed = rmsd_dfs_per_wt[wt_n].set_index("Sequence_ID")
                    gt_id = f"GT{wt_n}_mu1.0"
                    if gt_id in indexed.index:
                        gt_val = float(indexed.loc[gt_id, "rmsd"])
            elif metric_type == "yang_likelihood":
                mdata = load_yang_likelihood_data(wt_seq, wt_n)
                ext_vals = []
                y_fixed = 1.0
                gt_val = None
            else:
                continue

            draw_panel(ax, mdata, wt_color, light_color, ext_vals, y_fixed_max=y_fixed, gt_val=gt_val, metric_type=metric_type)

            if col_idx == 0:
                ax.set_ylabel(row_titles[row_idx], fontsize=18, fontweight="bold")
                ax.tick_params(axis="y", labelsize=16)
            else:
                ax.tick_params(axis="y", labelsize=0, length=0)

            if metric_type == "plddt":
                ax.set_ylim(0.6, 1.0)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=18, fontweight="bold")
            if row_idx == len(metric_types) - 1:
                ax.set_xlabel(r"$\mu_{\mathrm{gen}}$", fontsize=18, fontweight="bold", labelpad=10)

    legend_elements = [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label=r"$\bm{s}^{\text{MAP}}$"),
        Line2D([0], [0], color="red", linestyle=":", linewidth=2.5, label="Ground Truth"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#aaaaaa", markersize=8, label="Extant (E)"),
    ]
    axes[0][0].legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=14,
        frameon=True,
        framealpha=0.98,
        edgecolor="black",
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    return fig


def create_figure5_quat(
    sequences: list,
    mu_values: list,
    M: int,
    T: float,
    msa_folder: str,
    posterior_folder: str,
    GT_sequences: dict,
    fields_: np.ndarray,
    couplings: np.ndarray,
    seq_to_label: dict,
    plddt_result_files: dict = None,
    rmsd_result_files: dict = None,
    extant_rmsd_files: dict = None,
    extant_plddt_file: str = None,
    colors_dict: dict = None,
    energy_keep_pct: float = None,
    natural_alignment: str = None,
    figsize: tuple = (17, 12),
    data_prefix: str = "DBD",
):
    """
    Cleaner 3x3 summary of Supplementary Figure 5 bis.

    Rows   : Hamming distance to GT | pLDDT | RMSD
    Columns: GT1 | GT3 | GT5

    Each panel shows D10 and Y10 as median lines across mu_gen with IQR error bars.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.lines import Line2D

    def parse_mu_df(df, wt_n, metric_col):
        indexed = df.set_index("Sequence_ID")
        out = {"DCA": [], "Yang": [], "MAP": None}
        for k in range(1, 11):
            sid = f"DCA{wt_n}_{k}"
            if sid in indexed.index:
                out["DCA"].append(float(indexed.loc[sid, metric_col]))
        for k in range(1, 11):
            sid = f"Yang{wt_n}_{k}"
            if sid in indexed.index:
                out["Yang"].append(float(indexed.loc[sid, metric_col]))
        map_id = f"MAP{wt_n}"
        if map_id in indexed.index:
            out["MAP"] = float(indexed.loc[map_id, metric_col])
        return out

    def parse_rmsd_data_from_wt_file(df, wt_n, mu_values_list):
        indexed = df.set_index("Sequence_ID")
        result = {}
        for mu in mu_values_list:
            out = {"DCA": [], "Yang": [], "MAP": None}
            for k in range(1, 11):
                sid = f"DCA{wt_n}_{k}_mu{mu}"
                if sid in indexed.index:
                    out["DCA"].append(float(indexed.loc[sid, "rmsd"]))
            for k in range(1, 11):
                sid = f"Yang{wt_n}_{k}_mu{mu}"
                if sid in indexed.index:
                    out["Yang"].append(float(indexed.loc[sid, "rmsd"]))
            map_id = f"MAP{wt_n}_mu{mu}"
            if map_id in indexed.index:
                out["MAP"] = float(indexed.loc[map_id, "rmsd"])
            result[mu] = out
        return result

    def load_hamming_data(wt_seq, wt_n):
        GT = GT_sequences[wt_seq]
        result = {}
        for mu in mu_values:
            try:
                reshuffled_path = (
                    f"{msa_folder}/{wt_seq}_mu={mu}_depth=None"
                    f"_shuffled_M={M}_T={T}"
                )
                if not os.path.exists(reshuffled_path):
                    continue
                dca_reshuffled = np.loadtxt(reshuffled_path, dtype=int)
                posterior_msa_path = f"{msa_folder}/{wt_seq}_mu={mu}_depth=None_M={M}"
                dca_posterior_msa = None
                if os.path.exists(posterior_msa_path):
                    dca_posterior_msa = np.loadtxt(posterior_msa_path, dtype=int)
                energies_reshuffled = energy_of_msa(dca_reshuffled, fields_, couplings)
                posterior = None
                ppath = os.path.join(
                    posterior_folder,
                    f"{data_prefix}_{wt_seq}_mu{mu}_ancestral_probability"
                )
                if os.path.exists(ppath):
                    posterior = np.loadtxt(ppath)

                dca_vals, yang_vals, map_val = [], [], None
                if posterior is not None:
                    try:
                        sel_idx = select_topN(
                            dca_reshuffled, energies_reshuffled,
                            posterior=posterior, scoring="yang",
                            percentage=energy_keep_pct, topN=10
                        )
                        for idx in sel_idx:
                            seq = dca_reshuffled[int(idx)]
                            dca_vals.append(float(np.mean(seq != GT)))
                    except Exception:
                        pass

                    if dca_posterior_msa is not None:
                        try:
                            energies_post = energy_of_msa(dca_posterior_msa, fields_, couplings)
                            sel_idx2 = select_topN(
                                dca_posterior_msa, energies_post,
                                posterior=posterior, scoring="yang",
                                percentage=None, topN=10
                            )
                            for idx in sel_idx2:
                                seq = dca_posterior_msa[int(idx)]
                                yang_vals.append(float(np.mean(seq != GT)))
                        except Exception:
                            pass

                    ML_seq = np.argmax(posterior, axis=1)
                    map_val = float(np.mean(ML_seq != GT))
                result[mu] = {"DCA": dca_vals, "Yang": yang_vals, "MAP": map_val}
            except Exception as e:
                print(f"    Hamming error at WT{wt_n} mu={mu}: {e}")
        return result

    def summarize(vals):
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            return None
        q25, med, q75 = np.percentile(arr, [25, 50, 75])
        return float(med), float(q25), float(q75)

    def plot_reference_band(ax, values, color, alpha, median_color=None):
        if not values:
            return
        arr = np.asarray(values, dtype=float)
        q25, med, q75 = np.percentile(arr, [25, 50, 75])
        ax.axhspan(q25, q75, color=color, alpha=alpha, zorder=0)
        ax.axhline(med, color=median_color or color, linewidth=1.8, alpha=0.95, zorder=1)

    def draw_summary_panel(ax, mdata, extant_vals, gt_val=None, y_limits=None):
        x_dca, y_dca, err_dca_low, err_dca_high = [], [], [], []
        x_yang, y_yang, err_yang_low, err_yang_high = [], [], [], []
        x_map, y_map = [], []

        for mu in mu_values:
            if mu not in mdata:
                continue
            if mdata[mu].get("MAP") is not None:
                x_map.append(float(mu))
                y_map.append(float(mdata[mu]["MAP"]))

            s_dca = summarize(mdata[mu].get("DCA", []))
            if s_dca is not None:
                med, q25, q75 = s_dca
                x_dca.append(float(mu))
                y_dca.append(med)
                err_dca_low.append(med - q25)
                err_dca_high.append(q75 - med)

            s_yang = summarize(mdata[mu].get("Yang", []))
            if s_yang is not None:
                med, q25, q75 = s_yang
                x_yang.append(float(mu))
                y_yang.append(med)
                err_yang_low.append(med - q25)
                err_yang_high.append(q75 - med)

        if y_limits is not None:
            ax.set_ylim(*y_limits)

        plot_reference_band(ax, extant_vals, color="#bdbdbd", alpha=0.28, median_color="#6e6e6e")
        if gt_val is not None:
            ax.axhline(gt_val, color="#c0392b", linestyle=":", linewidth=2.0, zorder=1)
        if x_map:
            ax.plot(x_map, y_map, linestyle="--", color="#666666", linewidth=1.5, zorder=2)

        if x_dca:
            ax.errorbar(
                x_dca, y_dca, yerr=[err_dca_low, err_dca_high],
                fmt="o-", color="#D55E00", ecolor="#D55E00",
                elinewidth=1.5, capsize=3, markersize=5.5, linewidth=2.0, zorder=4,
            )
        if x_yang:
            ax.errorbar(
                x_yang, y_yang, yerr=[err_yang_low, err_yang_high],
                fmt="o-", color="#0072B2", ecolor="#0072B2",
                elinewidth=1.5, capsize=3, markersize=5.5, linewidth=2.0, zorder=5,
            )

        ax.set_xscale("log")
        ax.set_xticks(mu_values)
        ax.get_xaxis().set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
        ax.grid(alpha=0.28, linestyle="--", axis="y")
        ax.grid(alpha=0.14, linestyle=":", axis="x")

    if colors_dict is None:
        cmap_obj = cm.get_cmap("plasma")
        color_values = np.linspace(0.1, 0.9, len(sequences))
        colors_dict = {seq: cmap_obj(v) for seq, v in zip(sequences, color_values)}

    natural_sequences = []
    if natural_alignment and os.path.exists(natural_alignment):
        natural_sequences = read_fasta1(natural_alignment)

    plddt_dfs = {}
    if plddt_result_files:
        for mu, fpath in plddt_result_files.items():
            if os.path.exists(fpath):
                plddt_dfs[mu] = pd.read_csv(fpath)

    rmsd_dfs_per_wt = {}
    # Use provided rmsd_result_files parameter instead of hardcoded paths
    if rmsd_result_files:
        for wt_n, fpath in rmsd_result_files.items():
            if os.path.exists(fpath):
                rmsd_dfs_per_wt[wt_n] = pd.read_csv(fpath)
            else:
                print(f"    WARNING: RMSD file not found for WT{wt_n}: {fpath}")

    extant_plddt_vals = []
    if extant_plddt_file and os.path.exists(extant_plddt_file):
        extant_plddt_vals = pd.read_csv(extant_plddt_file)["pLDDT"].dropna().tolist()

    extant_rmsd_vals = {}
    if extant_rmsd_files:
        for wt_n, fpath in extant_rmsd_files.items():
            if os.path.exists(fpath):
                extant_rmsd_vals[wt_n] = pd.read_csv(fpath)["rmsd"].dropna().tolist()

    fig, axes = plt.subplots(3, 3, figsize=figsize, sharex="col")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.11, wspace=0.16, hspace=0.18)

    metric_row_labels = [
        r"$d_\text{H}(\bm{s}_R,\bm{s}^\text{GT})$",
        "pLDDT",
        r"RMSD ($\AA$)",
    ]
    col_titles = ["GT1", "GT3", "GT5"]
    gt_numbers = [1, 3, 5]

    for col_idx, wt_n in enumerate(gt_numbers):
        wt_seq_idx = wt_n - 1
        if wt_seq_idx >= len(sequences):
            continue
        wt_seq = sequences[wt_seq_idx]

        hamming_data = load_hamming_data(wt_seq, wt_n)
        hamming_ext_vals = []
        if GT_sequences and natural_sequences:
            gt_seq = GT_sequences.get(wt_seq)
            if gt_seq is not None:
                gt_len = len(gt_seq)
                hamming_ext_vals = [calculate_hamming_distance(nat_seq, gt_seq) / gt_len for nat_seq in natural_sequences]

        plddt_data = {
            mu: (parse_mu_df(plddt_dfs[mu], wt_n, "pLDDT") if mu in plddt_dfs else {"DCA": [], "Yang": [], "MAP": None})
            for mu in mu_values
        }
        plddt_gt_val = None
        if plddt_dfs and 1 in plddt_dfs:
            indexed = plddt_dfs[1].set_index("Sequence_ID")
            gt_id = f"GT{wt_n}"
            if gt_id in indexed.index:
                plddt_gt_val = float(indexed.loc[gt_id, "pLDDT"])

        rmsd_data = parse_rmsd_data_from_wt_file(rmsd_dfs_per_wt[wt_n], wt_n, mu_values) if wt_n in rmsd_dfs_per_wt else {}
        rmsd_gt_val = None
        if wt_n in rmsd_dfs_per_wt:
            indexed = rmsd_dfs_per_wt[wt_n].set_index("Sequence_ID")
            gt_id = f"GT{wt_n}_mu1.0"
            if gt_id in indexed.index:
                rmsd_gt_val = float(indexed.loc[gt_id, "rmsd"])

        panel_specs = [
            (0, hamming_data, hamming_ext_vals, None, (0.0, 1.0)),
            (1, plddt_data, extant_plddt_vals, plddt_gt_val, (0.6, 1.0)),
            (2, rmsd_data, extant_rmsd_vals.get(wt_n, []), rmsd_gt_val, (0.0, 20.0)),
        ]

        for row_idx, mdata, ext_vals, gt_val, y_limits in panel_specs:
            ax = axes[row_idx, col_idx]
            draw_summary_panel(ax, mdata, ext_vals, gt_val=gt_val, y_limits=y_limits)
            ax.tick_params(axis="both", labelsize=15)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=20, fontweight="bold", pad=10)
            if col_idx == 0:
                ax.set_ylabel(metric_row_labels[row_idx], fontsize=19, fontweight="bold")
            if row_idx < 2:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel(r"$\mu_{\mathrm{gen}}$", fontsize=18, fontweight="bold")

    legend_handles = [
        Line2D([0], [0], color="#D55E00", marker="o", linewidth=2.0, markersize=6, label=r"$D_{10}$ median $\pm$ IQR"),
        Line2D([0], [0], color="#0072B2", marker="o", linewidth=2.0, markersize=6, label=r"$Y_{10}$ median $\pm$ IQR"),
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.5, label=r"$\bm{s}^{\mathrm{MAP}}$"),
        Line2D([0], [0], color="#c0392b", linestyle=":", linewidth=2.0, label="Ground Truth"),
        Line2D([0], [0], color="#9e9e9e", linewidth=7, alpha=0.35, label="Extant IQR"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5,
        fontsize=14,
        frameon=True,
        framealpha=0.98,
        edgecolor="black",
        bbox_to_anchor=(0.5, 0.985),
        columnspacing=1.4,
        handlelength=2.1,
    )

    return fig


# ============================================================
# --- SUPPLEMENTARY FIGURE: GT2/GT4 (INDICES 1,3) ---
# ============================================================

def create_figure5_bis_supplementary(
    sequences: list,
    mu_values: list,
    M: int,
    T: float,
    msa_folder: str,
    posterior_folder: str,
    GT_sequences: dict,
    fields_: np.ndarray,
    couplings: np.ndarray,
    seq_to_label: dict,
    plddt_result_files: dict = None,
    rmsd_result_files: dict = None,
    extant_rmsd_files: dict = None,
    extant_plddt_file: str = None,
    colors_dict: dict = None,
    energy_keep_pct: float = None,
    natural_alignment: str = None,
    figsize: tuple = (20, 10),
    data_prefix: str = "DBD",
):
    """
    Supplementary Figure: GT2 and GT4 boxplots.
    Same layout as Figure 5bis but for the two intermediate WTs.
    
    Rows: GT2, GT4
    Columns: Hamming distance to GT | pLDDT | RMSD
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    from matplotlib import cm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
 
    # Helpers (same as create_figure5_bis)
    def lighten_color(color, factor=0.5):
        r, g, b = color[:3]
        return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)
 
    def parse_mu_df(df, wt_n, metric_col):
        indexed = df.set_index("Sequence_ID")
        out = {"DCA": [], "Yang": [], "MAP": None}
        for k in range(1, 11):
            sid = f"DCA{wt_n}_{k}"
            if sid in indexed.index:
                out["DCA"].append(float(indexed.loc[sid, metric_col]))
        for k in range(1, 11):
            sid = f"Yang{wt_n}_{k}"
            if sid in indexed.index:
                out["Yang"].append(float(indexed.loc[sid, metric_col]))
        map_id = f"MAP{wt_n}"
        if map_id in indexed.index:
            out["MAP"] = float(indexed.loc[map_id, metric_col])
        return out
    
    def parse_rmsd_data_from_wt_file(df, wt_n, mu_values_list):
        indexed = df.set_index("Sequence_ID")
        result = {}
        for mu in mu_values_list:
            out = {"DCA": [], "Yang": [], "MAP": None}
            for k in range(1, 11):
                sid = f"DCA{wt_n}_{k}_mu{mu}"
                if sid in indexed.index:
                    out["DCA"].append(float(indexed.loc[sid, "rmsd"]))
            for k in range(1, 11):
                sid = f"Yang{wt_n}_{k}_mu{mu}"
                if sid in indexed.index:
                    out["Yang"].append(float(indexed.loc[sid, "rmsd"]))
            map_id = f"MAP{wt_n}_mu{mu}"
            if map_id in indexed.index:
                out["MAP"] = float(indexed.loc[map_id, "rmsd"])
            result[mu] = out
        return result
 
    def load_hamming_data(wt_seq, wt_n):
        GT = GT_sequences[wt_seq]
        result = {}
        for mu in mu_values:
            try:
                reshuffled_path = (
                    f"{msa_folder}/{wt_seq}_mu={mu}_depth=None"
                    f"_shuffled_M={M}_T={T}"
                )
                if not os.path.exists(reshuffled_path):
                    continue
                dca_reshuffled = np.loadtxt(reshuffled_path, dtype=int)
                posterior_msa_path = (
                    f"{msa_folder}/{wt_seq}_mu={mu}_depth=None_M={M}"
                )
                dca_posterior_msa = None
                if os.path.exists(posterior_msa_path):
                    dca_posterior_msa = np.loadtxt(posterior_msa_path, dtype=int)
                energies_reshuffled = energy_of_msa(dca_reshuffled, fields_, couplings)
                posterior = None
                ppath = os.path.join(
                    posterior_folder,
                    f"{data_prefix}_{wt_seq}_mu{mu}_ancestral_probability"
                )
                if os.path.exists(ppath):
                    posterior = np.loadtxt(ppath)
                dca_vals, yang_vals, map_val = [], [], None
                if posterior is not None:
                    try:
                        sel_idx = select_topN(
                            dca_reshuffled, energies_reshuffled,
                            posterior=posterior, scoring="yang",
                            percentage=energy_keep_pct, topN=10
                        )
                        for idx in sel_idx:
                            seq = dca_reshuffled[int(idx)]
                            dca_vals.append(float(np.mean(seq != GT)))
                    except Exception:
                        pass
                    if dca_posterior_msa is not None:
                        try:
                            energies_post = energy_of_msa(
                                dca_posterior_msa, fields_, couplings
                            )
                            sel_idx2 = select_topN(
                                dca_posterior_msa, energies_post,
                                posterior=posterior, scoring="yang",
                                percentage=None, topN=10
                            )
                            for idx in sel_idx2:
                                seq = dca_posterior_msa[int(idx)]
                                yang_vals.append(float(np.mean(seq != GT)))
                        except Exception:
                            pass
                    ML_seq = np.argmax(posterior, axis=1)
                    map_val = float(np.mean(ML_seq != GT))
                result[mu] = {"DCA": dca_vals, "Yang": yang_vals, "MAP": map_val}
            except Exception as e:
                print(f"    Hamming error at WT{wt_n} mu={mu}: {e}")
        return result
    
    def draw_panel(ax, mdata, wt_color, light_color, extant_vals, y_fixed_max=None):
        group_labels = ["10 best DCA", "10 best s-i"]
        extant_label = "Extant"
        has_extant = len(extant_vals) > 0
        rows = []
        x_pos = 0
        mu_positions, mu_labels = [], []
        
        for mu in mu_values:
            if mu not in mdata:
                continue
            mu_start_pos = x_pos
            for gkey, glabel in [("DCA", "10 best DCA"), ("Yang", "10 best s-i")]:
                for val in mdata[mu].get(gkey, []):
                    rows.append({"x_id": f"{mu}_{glabel}", "Group": glabel, "Value": val, "x_pos": x_pos})
                x_pos += 1
            mu_center = (mu_start_pos + x_pos - 1) / 2
            mu_positions.append(mu_center)
            mu_labels.append(f"{mu:g}")
            x_pos += 0.5  # Gap between mu groups
        
        if has_extant:
            extant_x_pos = x_pos
            for val in extant_vals:
                rows.append({"x_id": "Extant", "Group": extant_label, "Value": val, "x_pos": extant_x_pos})
        
        if not rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13)
            return
        
        df = pd.DataFrame(rows)
 
        # Per-box color palette (by group)
        palette = []
        for idx, row in df.iterrows():
            if "10 best DCA" in row["x_id"]:
                palette.append(wt_color)
            elif "10 best s-i" in row["x_id"]:
                palette.append(light_color)
            else:
                palette.append("#aaaaaa")
 
        # Create boxplot with spacing - use x_pos to create gaps
        for group_name in df["x_id"].unique():
            group_data = df[df["x_id"] == group_name]
            pos = group_data["x_pos"].iloc[0]
            if "10 best DCA" in group_name:
                color = wt_color
            elif "10 best s-i" in group_name:
                color = light_color
            else:
                color = "#aaaaaa"
            ax.boxplot([group_data["Value"].values], positions=[pos], widths=0.6,
                      patch_artist=True, showfliers=False,
                      boxprops=dict(facecolor=color, edgecolor="black"),
                      medianprops=dict(color="black"),
                      whiskerprops=dict(color="black"),
                      capprops=dict(color="black"))
 
        # MAP dashed reference line, spanning each mu group
        for mu_idx, mu in enumerate(mu_values):
            map_val = mdata.get(mu, {}).get("MAP", None)
            if map_val is None:
                continue
            mu_center = mu_positions[mu_idx]
            ax.hlines(map_val,
                      xmin=mu_center - 0.5, xmax=mu_center + 0.5,
                      colors="gray", linestyles="--", linewidth=1.2)
        
        if y_fixed_max is not None:
            ax.set_ylim(0, y_fixed_max)
        ylim_cur = ax.get_ylim()
        y_range = ylim_cur[1] - ylim_cur[0]
        
        symbol_map_local = {
            "10 best DCA": r"$\mathbf{D_{10}}$",
            "10 best s-i": r"$\mathbf{Y_{10}}$",
            "Extant": r"$\mathbf{E}$",
        }
        # Add one label per group (not per data point)
        for x_id in df["x_id"].unique():
            group_data = df[df["x_id"] == x_id]
            group = (
                "10 best DCA" if "10 best DCA" in x_id
                else "10 best s-i" if "10 best s-i" in x_id
                else "Extant"
            )
            # Position label at the max value of the group
            if len(group_data) > 0:
                max_val = group_data["Value"].max()
                x_pos = group_data["x_pos"].iloc[0]
                label_y = max_val + 0.02 * y_range
                # Only add label if it fits within the y-axis bounds
                if label_y < ylim_cur[1] * 0.95:  # Leave 5% margin
                    ax.text(x_pos, label_y,
                            symbol_map_local[group],
                            ha="center", va="bottom",
                            fontsize=13, fontweight="bold", clip_on=True)
        
        # Set x-axis ticks and labels for mu values and extant if present
        all_positions = mu_positions.copy()
        all_labels = [f"{mu:g}" for mu in mu_values]
        if has_extant and len(extant_vals) > 0:
            extant_x_pos = (mu_positions[-1] + 1.5) if mu_positions else 1.0
            all_positions.append(extant_x_pos)
            all_labels.append("E")
        
        ax.set_xticks(all_positions)
        ax.set_xticklabels(all_labels, fontsize=14)
 
        ax.grid(alpha=0.3, linestyle="--", axis="y")
    
    # Load data
    plddt_dfs = {}
    if plddt_result_files:
        for mu, fpath in plddt_result_files.items():
            if os.path.exists(fpath):
                plddt_dfs[mu] = pd.read_csv(fpath)
    
    rmsd_dfs_per_wt = {}
    # Use provided rmsd_result_files parameter instead of hardcoded paths
    if rmsd_result_files:
        for wt_n, fpath in rmsd_result_files.items():
            if os.path.exists(fpath):
                rmsd_dfs_per_wt[wt_n] = pd.read_csv(fpath)
            else:
                print(f"    WARNING: RMSD file not found for WT{wt_n}: {fpath}")
    
    extant_plddt_vals = []
    if extant_plddt_file and os.path.exists(extant_plddt_file):
        df_ext_p = pd.read_csv(extant_plddt_file)
        extant_plddt_vals = df_ext_p["pLDDT"].dropna().tolist()
    
    extant_rmsd_vals = {}
    if extant_rmsd_files:
        for wt_n, fpath in extant_rmsd_files.items():
            if os.path.exists(fpath):
                df_ext_r = pd.read_csv(fpath)
                extant_rmsd_vals[wt_n] = df_ext_r["rmsd"].dropna().tolist()
    
    if colors_dict is None:
        cmap_obj = cm.get_cmap("plasma")
        color_values = np.linspace(0.1, 0.9, len(sequences))
        colors_dict = {seq: cmap_obj(v) for seq, v in zip(sequences, color_values)}
    
    # Load natural sequences from FASTA file
    natural_sequences = []
    if natural_alignment and os.path.exists(natural_alignment):
        natural_sequences = read_fasta1(natural_alignment)
    
    # Setup figure: 2 rows x 4 columns
    col_titles = ["Hamming distance to GT", "pLDDT", "RMSD"]
    col_ylabels = [
        r"$d_\text{H}(\bm{s}_R,\bm{s}^\text{GT})$",
        "pLDDT",
        r"RMSD ($\AA$)",
    ]
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.38, wspace=0.3, width_ratios=[1, 1, 1, 0.15])
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
    
    for c in range(3):
        axes[1][c].sharey(axes[0][c])
    
    # Process GT2 and GT4
    for row_idx, wt_n in enumerate([2, 4]):
        wt_seq_idx = wt_n - 1
        if wt_seq_idx >= len(sequences):
            continue
        wt_seq = sequences[wt_seq_idx]
        wt_label = seq_to_label.get(wt_seq, f"WT{wt_n}")
        wt_color = colors_dict.get(wt_seq, "gray")
        light_color = lighten_color(wt_color, factor=0.5)
        
        print(f"  Processing supplementary row for {wt_label} (WT{wt_n})...")
        
        hamming_data = load_hamming_data(wt_seq, wt_n)
        # For extant hamming distance: compute distance from natural sequences to GT
        hamming_ext_vals = []
        if GT_sequences and natural_sequences:
            # Try to find GT sequence by wt_seq first, then by wt_n
            gt_seq = GT_sequences.get(wt_seq)
            if gt_seq is None:
                gt_seq = GT_sequences.get(wt_n)
            if gt_seq is not None:
                gt_len = len(gt_seq)
                hamming_ext_vals = [calculate_hamming_distance(nat_seq, gt_seq) / gt_len for nat_seq in natural_sequences]
        
        plddt_data = {
            mu: (parse_mu_df(plddt_dfs[mu], wt_n, "pLDDT") if mu in plddt_dfs
                 else {"DCA": [], "Yang": [], "MAP": None})
            for mu in mu_values
        }
        rmsd_data = {}
        if wt_n in rmsd_dfs_per_wt:
            rmsd_data = parse_rmsd_data_from_wt_file(
                rmsd_dfs_per_wt[wt_n], wt_n, mu_values
            )
        else:
            rmsd_data = {mu: {"DCA": [], "Yang": [], "MAP": None} for mu in mu_values}
        
        panel_configs = [
            (0, hamming_data, hamming_ext_vals, 1.0),
            (1, plddt_data, extant_plddt_vals, 1.0),
            (2, rmsd_data, extant_rmsd_vals.get(wt_n, []), 20.0),
        ]
        
        for col_idx, mdata, ext_vals, y_fixed in panel_configs:
            ax = axes[row_idx][col_idx]
            draw_panel(ax, mdata, wt_color, light_color,
                       extant_vals=ext_vals, y_fixed_max=y_fixed)
            ax.set_ylabel(col_ylabels[col_idx], fontsize=14)
            ax.tick_params(axis="y", labelsize=13)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=14, fontweight="bold")
            if row_idx < 1:
                ax.set_xticklabels([])
        
        right_ax = fig.add_subplot(gs[row_idx, 3])
        right_ax.axis("off")
        right_ax.text(0.5, 0.5, wt_label, fontsize=16, fontweight="bold",
                     ha="center", va="center", transform=right_ax.transAxes)
    
    # X-axis labels
    for col_idx in range(3):
        bottom_ax = axes[1][col_idx]
        try:
            bottom_ax.set_xlabel(r"$\mu_{\mathrm{gen}}$", fontsize=16, labelpad=10)
            fig.canvas.draw()
        except:
            bottom_ax.set_xlabel("", fontsize=0)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5,
               label=r"$\bm{s}^{\text{MAP}}$"),
        Patch(facecolor="#aaaaaa", edgecolor="black",
              label="Extant sequences"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=13,
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        bbox_to_anchor=(0.99, 0.01),
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig



# ============================================================
# --- SUPPLEMENTARY: TOP-10 CANDIDATES OVERLAYS ---
# ============================================================

def create_supplementary_top10_on_gridplot(
    sequences: list,
    mu_values: list,
    M: int,
    T: float,
    msa_save_folder: str,
    ancestral_probabilities_folder: str,
    consensus_directory: str,
    GT_sequences: dict,
    couplings: np.ndarray,
    fields_: np.ndarray,
    seq_to_label: dict,
    energy_keep_pct: float | None = None,
    data_prefix: str = "DBD",
    wt_indices: list | tuple | None = None,
    include_wt24: bool = False,
    topN: int = 10,
):
    """
    Supplementary figure:
    Overlay Figure-5 top-N candidates (DCA and site-independent) on Figure-4 grid panels.
    """
    selected_wts = _resolve_candidate_wts(sequences, wt_indices=wt_indices, include_wt24=include_wt24)
    if len(selected_wts) == 0:
        raise ValueError("No valid WT indices selected for supplementary top-N grid figure.")

    mu_values_grid = [mu for mu in mu_values if 1.0 <= mu <= 100.0]
    if len(mu_values_grid) == 0:
        raise ValueError("No mu values in [1, 100] available for grid overlay figure.")

    fig, _, _ = create_figure4(
        sequences=selected_wts,
        mu_values=mu_values_grid,
        M=M,
        msa_save_folder=msa_save_folder,
        ancestral_probabilities_folder=ancestral_probabilities_folder,
        consensus_directory=consensus_directory,
        GT_sequences=GT_sequences,
        couplings=couplings,
        fields_=fields_,
        T=T,
        seq_to_label=seq_to_label,
        data_prefix=data_prefix,
    )

    candidate_data = _collect_top10_candidates_by_wt_mu(
        selected_wts=selected_wts,
        mu_values=mu_values_grid,
        M=M,
        T=T,
        msa_folder=msa_save_folder,
        posterior_folder=ancestral_probabilities_folder,
        GT_sequences=GT_sequences,
        fields_=fields_,
        couplings=couplings,
        energy_keep_pct=energy_keep_pct,
        data_prefix=data_prefix,
        topN=topN,
    )

    n_rows = len(selected_wts)
    n_cols = len(mu_values_grid)
    axes = np.asarray(fig.axes[:n_rows * n_cols], dtype=object).reshape(n_rows, n_cols)

    for i, wt in enumerate(selected_wts):
        for j, mu in enumerate(mu_values_grid):
            ax = axes[i, j]
            entry = candidate_data.get(wt, {}).get(mu)
            if entry is None:
                continue

            if entry["dca_energy"].size > 0:
                ax.scatter(
                    entry["dca_energy"],
                    entry["dca_dist"],
                    marker="D",
                    s=42,
                    facecolors="none",
                    edgecolors="#111111",
                    linewidths=1.1,
                    zorder=260,
                    alpha=0.95,
                )

            if entry["si_energy"].size > 0:
                ax.scatter(
                    entry["si_energy"],
                    entry["si_dist"],
                    marker="o",
                    s=42,
                    facecolors="none",
                    edgecolors="#1f77b4",
                    linewidths=1.2,
                    zorder=270,
                    alpha=0.95,
                )

    overlay_handles = [
        Line2D([0], [0], marker="D", linestyle="None", markersize=7,
               markerfacecolor="none", markeredgecolor="#111111", markeredgewidth=1.1,
               label=rf"$D_{{{topN}}}$ candidates"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=7,
               markerfacecolor="none", markeredgecolor="#1f77b4", markeredgewidth=1.2,
               label=rf"$Y_{{{topN}}}$ candidates"),
    ]
    leg = axes[0, 0].legend(handles=overlay_handles, loc="upper right", frameon=True)
    leg.get_frame().set_alpha(0.95)

    # Keep Figure 4 x-label but place it slightly lower to avoid overlap with tick labels.
    for txt in fig.texts:
        if txt.get_text() == r"$E_{\mathrm{DCA}}$":
            txt.set_position((0.5, 0.015))
            break

    # No supplementary title (requested).
    fig.subplots_adjust(bottom=0.12)
    return fig


def create_supplementary_top10_on_reweighted_pca(
    sequences: list,
    mu_values: list,
    M: int,
    T: float,
    msa_save_folder: str,
    ancestral_probabilities_folder: str,
    GT_sequences: dict,
    fields_: np.ndarray,
    couplings: np.ndarray,
    seq_to_label: dict,
    pca_alignment: str,
    energy_keep_pct: float | None = None,
    data_prefix: str = "DBD",
    wt_indices: list | tuple | None = None,
    include_wt24: bool = True,
    topN: int = 10,
    extant_limit: int = 12000,
):
    """
    Supplementary figure:
    Plot Figure-5 top-N candidates on PCA fitted/projected on reweighted extant sequences,
    including the starting (GT) point for each WT.
    """
    if pca_alignment is None:
        raise ValueError("pca_alignment is required for supplementary PCA candidates figure.")

    # For this supplementary figure, default to all provided WTs unless explicitly subset.
    if wt_indices is None:
        selected_wts = list(sequences)
    else:
        selected_wts = _resolve_candidate_wts(sequences, wt_indices=wt_indices, include_wt24=include_wt24)
    if len(selected_wts) == 0:
        raise ValueError("No valid WT indices selected for supplementary top-N PCA figure.")

    mu_values_grid = [mu for mu in mu_values if 1.0 <= mu <= 100.0]
    if len(mu_values_grid) == 0:
        raise ValueError("No mu values in [1, 100] available for supplementary PCA figure.")

    candidate_data = _collect_top10_candidates_by_wt_mu(
        selected_wts=selected_wts,
        mu_values=mu_values_grid,
        M=M,
        T=T,
        msa_folder=msa_save_folder,
        posterior_folder=ancestral_probabilities_folder,
        GT_sequences=GT_sequences,
        fields_=fields_,
        couplings=couplings,
        energy_keep_pct=energy_keep_pct,
        data_prefix=data_prefix,
        topN=topN,
    )

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    def _subsample_rows(arr: np.ndarray, limit: int) -> np.ndarray:
        if (limit is None) or (limit <= 0) or (arr.shape[0] <= limit):
            return arr
        idx = np.random.choice(arr.shape[0], size=int(limit), replace=False)
        return arr[idx]

    def _one_hot_flat(msa_int: np.ndarray, q_: int) -> np.ndarray:
        oh = np.eye(q_, dtype=np.float32)[msa_int]
        return oh.reshape(msa_int.shape[0], -1)

    def _plot_density(ax, x, y, cmap_local, levels=22):
        points = np.vstack([x, y])
        if points.shape[1] < 5:
            return
        kde = gaussian_kde(points)
        xx, yy = np.mgrid[x.min()-1.2:x.max()+1.2:260j, y.min()-1.2:y.max()+1.2:260j]
        z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.contourf(xx, yy, z, levels=levels, cmap=cmap_local, alpha=0.93, zorder=1)
        ax.contour(xx, yy, z, levels=10, colors=["0.35"], linewidths=0.35, alpha=0.35, zorder=2)

    extant_msa = np.asarray(read_fasta1(pca_alignment), dtype=int)
    extant_msa = _subsample_rows(extant_msa, extant_limit)

    q = max(21, int(np.max(extant_msa)) + 1)
    for wt in selected_wts:
        for mu in mu_values_grid:
            entry = candidate_data.get(wt, {}).get(mu)
            if entry is None:
                continue
            if entry["dca_seqs"].shape[0] > 0:
                q = max(q, int(np.max(entry["dca_seqs"])) + 1)
            if entry["si_seqs"].shape[0] > 0:
                q = max(q, int(np.max(entry["si_seqs"])) + 1)
            q = max(q, int(np.max(entry["starting_seq"])) + 1)

    extant_flat = _one_hot_flat(extant_msa, q)
    scaler = StandardScaler().fit(extant_flat)
    pca = PCA(n_components=2)
    extant_proj = pca.fit_transform(scaler.transform(extant_flat))

    title_fs = 40
    axis_label_fs = 40
    tick_fs = 32
    legend_fs = 34

    n_rows = len(selected_wts)
    n_cols = len(mu_values_grid)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.6 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    # Fixed colors for candidate classes across all mu panels.
    d10_color = "#D55E00"
    y10_color = "#0072B2"

    grey_cmap = LinearSegmentedColormap.from_list(
        "supp_rw_grey",
        [(1, 1, 1, 1), (0.15, 0.15, 0.15)],
        N=256,
    )

    for i, wt in enumerate(selected_wts):
        for j, mu in enumerate(mu_values_grid):
            ax = axes[i, j]
            _plot_density(ax, extant_proj[:, 0], extant_proj[:, 1], grey_cmap, levels=22)

            entry = candidate_data.get(wt, {}).get(mu)
            if entry is not None:
                if entry["dca_seqs"].shape[0] > 0:
                    dca_proj = pca.transform(scaler.transform(_one_hot_flat(entry["dca_seqs"], q)))
                    ax.scatter(
                        dca_proj[:, 0],
                        dca_proj[:, 1],
                        s=30,
                        marker="D",
                        facecolors="none",
                        edgecolors=[d10_color],
                        linewidths=1.4,
                        zorder=12,
                    )

                if entry["si_seqs"].shape[0] > 0:
                    si_proj = pca.transform(scaler.transform(_one_hot_flat(entry["si_seqs"], q)))
                    ax.scatter(
                        si_proj[:, 0],
                        si_proj[:, 1],
                        s=30,
                        marker="o",
                        facecolors="none",
                        edgecolors=[y10_color],
                        linewidths=1.4,
                        zorder=13,
                    )

                start_proj = pca.transform(scaler.transform(_one_hot_flat(entry["starting_seq"].reshape(1, -1), q)))
                ax.scatter(
                    start_proj[:, 0],
                    start_proj[:, 1],
                    s=95,
                    marker="D",
                    color="white",
                    edgecolor="black",
                    linewidth=1.1,
                    zorder=20,
                )

            # Column headers show mu values; no global figure title.
            if i == 0:
                ax.set_title(rf"$\mu_{{\mathrm{{gen}}}}={float(mu):g}$", fontsize=title_fs)

            # Row labels for WT identity on first column only.
            if j == 0:
                ax.set_ylabel(f"{seq_to_label.get(wt, wt)}\nPC2", fontsize=axis_label_fs)

            if i == n_rows - 1:
                ax.set_xlabel("PC1", fontsize=axis_label_fs)

            ax.tick_params(axis="both", labelsize=tick_fs)
            ax.grid(alpha=0.20, linestyle="--")

    # No supplementary title (requested).
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    return fig


def save_supplementary_top10_on_reweighted_pca_legend(
    out_path: str,
    topN: int = 10,
    use_latex: bool = True,
):
    """Save the Supplementary Figure 6 legend as a standalone PDF."""
    if use_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })

    d10_color = "#D55E00"
    y10_color = "#0072B2"
    legend_fs = 34
    marker_handles = [
        Line2D([0], [0], marker="D", linestyle="None", markersize=12,
               markerfacecolor="none", markeredgecolor=d10_color, markeredgewidth=1.8,
               label=rf"$D_{{{topN}}}$ candidates"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=12,
               markerfacecolor="none", markeredgecolor=y10_color, markeredgewidth=1.8,
               label=rf"$Y_{{{topN}}}$ candidates"),
        Line2D([0], [0], marker="D", linestyle="None", markersize=13,
               markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.2,
               label=r"$\bm{s}^\text{GT}$"),
    ]

    fig_leg = plt.figure(figsize=(13.5, 2.0))
    fig_leg.legend(
        handles=marker_handles,
        loc="center",
        ncol=len(marker_handles),
        frameon=True,
        fontsize=legend_fs,
        columnspacing=1.5,
        handlelength=2.4,
        borderpad=0.6,
        handletextpad=0.7,
    )
    fig_leg.savefig(out_path, bbox_inches="tight")
    plt.close(fig_leg)


def _extract_wt_ids_from_ancestral_folder(ancestral_folder: str, prefix: str) -> list:
    """Infer WT identifiers (e.g. wt2748) from ancestral posterior filenames."""
    wt_ids = set()
    if not os.path.isdir(ancestral_folder):
        return []

    pattern = re.compile(rf"^{re.escape(prefix)}_(wt\d+)_mu")
    for fname in os.listdir(ancestral_folder):
        match = pattern.match(fname)
        if match:
            wt_ids.add(match.group(1))

    def _wt_key(name: str):
        m = re.search(r"(\d+)$", name)
        return int(m.group(1)) if m else 10**9

    return sorted(wt_ids, key=_wt_key)


def _extract_wt_selection_from_notebook(notebook_path: str) -> tuple[list, list]:
    """
    Extract WT labels and 1-based WT indices from a notebook config cell.

    Expected label pattern: 'wt<index>' (e.g. wt17282).
    """
    if not os.path.isfile(notebook_path):
        return [], []

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception:
        return [], []

    cells = nb.get("cells", [])
    seq_labels = []

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        match = re.search(r"(?m)^\s*sequences\s*=\s*(\[[^\n\r]*\])", src)
        if not match:
            continue
        try:
            parsed = ast.literal_eval(match.group(1))
            if isinstance(parsed, list):
                seq_labels = [str(x).strip() for x in parsed]
                break
        except Exception:
            continue

    wt_indices = []
    for label in seq_labels:
        m = re.match(r"^wt(\d+)$", label, flags=re.IGNORECASE)
        if m:
            wt_indices.append(int(m.group(1)))

    return seq_labels, wt_indices


def _default_cd_entropy_family_config(project_root: str, family: str) -> dict:
    """Build default config for context-dependent entropy supplementary figure."""
    root = os.path.abspath(project_root)
    family_key = family.strip().lower()

    if family_key in {"dbd"}:
        # Use the exact GT WT definitions from notebook configs.
        # Requested order: WT1..WT5 by increasing CDE
        # (13202, 21394, 24786, 4722, 2748)
        sequences = ["wt13202", "wt21394", "wt24786", "wt4722", "wt2748"]
        wt_indices = [13202, 21394, 24786, 4722, 2748]
        return {
            "family_label": "DBD",
            "natural_alignment": os.path.join(root, "DBD", "DBD_alignment_cleaned_collapsed_noonlychild.fa"),
            "original_alignment": os.path.join(root, "DBD", "DBD_alignment.uniref90.cov80.a2m"),
            "potts_parameters": os.path.join(root, "DBD", "Parameters_conv_Thr-PCD40.dat"),
            "sequences": sequences,
            "wt_indices": wt_indices,
        }

    if family_key in {"betalac", "betalactase", "beta", "betalactamase"}:
        # Use the exact GT WT definitions from notebook configs.
        sequences = ["wt17282", "wt13063", "wt16682", "wt5628", "wt2192"]
        wt_indices = [17282, 13063, 16682, 5628, 2192]
        return {
            "family_label": "betaLactase",
            "natural_alignment": os.path.join(root, "final_betaLac", "betaLac_collapsed.fasta"),
            "original_alignment": os.path.join(root, "final_betaLac", "PF13354_noinsert_max19gaps_nodupl_noclose_BetaLact.faa"),
            "potts_parameters": os.path.join(root, "final_betaLac", "Parameters_conv_Matteo_pc_BetaLact.dat"),
            "sequences": sequences,
            "wt_indices": wt_indices,
        }

    raise ValueError(f"Unsupported family '{family}'. Choose from: DBD, betaLac, both")


def create_supplementary_context_dependent_entropy_figure(
    family_configs: list,
    use_latex: bool = True,
    max_msa_sequences: int | None = None,
    bins: int = 70,
    seed: int = 7,
):
    """
    Plot context-dependent entropy distributions for families, with WT entropy
    values highlighted as vertical lines using the same plasma colorscale.

    Expected per-family config keys:
        - family_label: str
        - natural_alignment: str
        - potts_parameters: str (optional if fields_/couplings provided)
        - fields_: np.ndarray (optional)
        - couplings: np.ndarray (optional)
        - sequences: list[str]   # WT names to highlight
    """
    axis_label_fs = 26
    title_fs = 28
    legend_fs = 20
    tick_fs = 22
    panel_letter_fs = 28

    if use_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 18,
            "axes.labelsize": axis_label_fs,
            "axes.titlesize": title_fs,
            "legend.fontsize": legend_fs,
            "xtick.labelsize": tick_fs,
            "ytick.labelsize": tick_fs,
        })

    n_panels = len(family_configs)
    # One family per row; reduce panel height and rely on larger fonts for readability.
    fig, axes = plt.subplots(n_panels, 1, figsize=(15.0, 7.8 * n_panels), squeeze=False)
    axes = axes[:, 0]

    display_title_map = {
        "dbd": r"DNA Binding Domain",
        "betalactase": r"$\beta$-lactamase",
        "betalac": r"$\beta$-lactamase",
        "beta-lactamase": r"$\beta$-lactamase",
    }

    rng = np.random.default_rng(seed)

    for panel_idx, (ax, cfg) in enumerate(zip(axes, family_configs)):
        family_label = cfg["family_label"]
        natural_alignment = cfg["natural_alignment"]
        sequences = list(cfg.get("sequences", []))
        seq_to_display = {seq: f"GT{i+1}" for i, seq in enumerate(sequences)}

        msa, name_to_index, _ = read_fasta2(natural_alignment)
        msa = np.asarray(msa, dtype=int)
        if msa.ndim != 2 or msa.shape[0] == 0:
            raise ValueError(f"Invalid MSA loaded from {natural_alignment}: shape={msa.shape}")

        fields_ = cfg.get("fields_")
        couplings = cfg.get("couplings")
        if fields_ is None or couplings is None:
            potts_parameters = cfg.get("potts_parameters")
            if potts_parameters is None:
                raise ValueError(f"Family '{family_label}' requires potts_parameters or fields_/couplings")
            fields_, couplings = read_potts_parameters_proteins(potts_parameters)

        if msa.shape[1] != int(fields_.shape[0]):
            raise ValueError(
                f"Length mismatch for {family_label}: MSA L={msa.shape[1]} vs fields L={fields_.shape[0]}"
            )

        if max_msa_sequences is not None and max_msa_sequences > 0 and msa.shape[0] > max_msa_sequences:
            idx = rng.choice(msa.shape[0], size=max_msa_sequences, replace=False)
            msa_eval = msa[idx]
        else:
            msa_eval = msa

        entropy_dist = context_dependent_entropy_msa_torch(msa_eval, fields_, couplings)

        ax.hist(
            entropy_dist,
            bins=bins,
            density=True,
            alpha=0.55,
            color="#c7c7c7",
            edgecolor="#666666",
            linewidth=0.8,
            label="Extant sequences",
        )

        cmap_seq = cm.get_cmap("plasma")
        color_values = np.linspace(0.1, 0.9, max(1, len(sequences)))
        colors_dict = {seq: cmap_seq(val) for seq, val in zip(sequences, color_values)}

        wt_indices = cfg.get("wt_indices", [])
        original_alignment = cfg.get("original_alignment")
        plotted_any_wt = False

        if len(wt_indices) > 0:
            source_msa = None
            source_name = None
            if original_alignment and os.path.isfile(original_alignment):
                source_msa = np.asarray(read_fasta1(original_alignment), dtype=int)
                source_name = "original"
            else:
                source_msa = msa
                source_name = "collapsed"

            if source_msa.ndim == 2 and source_msa.shape[1] == msa.shape[1]:
                valid_pairs = []
                for pos, idx_1based in enumerate(wt_indices):
                    idx = int(idx_1based) - 1
                    if 0 <= idx < source_msa.shape[0]:
                        if pos < len(sequences):
                            seq_name = sequences[pos]
                        else:
                            seq_name = f"wt{idx_1based}"
                        valid_pairs.append((seq_name, idx))

                if len(valid_pairs) == 0:
                    print(f"Warning: no valid WT indices in {source_name} alignment for family {family_label}")
                else:
                    wt_msa = np.asarray([source_msa[idx] for _, idx in valid_pairs], dtype=int)
                    wt_entropy = context_dependent_entropy_msa_torch(wt_msa, fields_, couplings)
                    for (seq_name, _), value in zip(valid_pairs, wt_entropy):
                        ax.axvline(
                            value,
                            color=colors_dict.get(seq_name, "black"),
                            linestyle="-",
                            linewidth=2.0,
                            alpha=0.95,
                            label=seq_to_display.get(seq_name, seq_name),
                            zorder=3,
                        )
                        plotted_any_wt = True
            else:
                print(f"Warning: invalid source alignment for WT indices in family {family_label}")

        if not plotted_any_wt:
            wt_names_present = [seq for seq in sequences if seq in name_to_index]
            if len(wt_names_present) == 0:
                print(f"Warning: no requested WT names found in alignment for family {family_label}")
            else:
                wt_msa = np.asarray([msa[name_to_index[seq]] for seq in wt_names_present], dtype=int)
                wt_entropy = context_dependent_entropy_msa_torch(wt_msa, fields_, couplings)
                for seq_name, value in zip(wt_names_present, wt_entropy):
                    ax.axvline(
                        value,
                        color=colors_dict.get(seq_name, "black"),
                        linestyle="-",
                        linewidth=2.0,
                        alpha=0.95,
                        label=seq_to_display.get(seq_name, seq_name).replace("WT", "GT"), #set label to GT1, GT2, etc. instead of WT1, WT2, etc.
                        zorder=3,
                    )

        ax.set_xlabel(r"CDE($\bm{s}$)", fontsize=axis_label_fs)
        ax.set_ylabel("Density", fontsize=axis_label_fs)
        title_key = str(family_label).strip().lower()
        ax.set_title(display_title_map.get(title_key, family_label), fontsize=title_fs)
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)
        ax.grid(True, alpha=0.25, linestyle="--")
        leg = ax.legend(frameon=True, ncol=2, fontsize=legend_fs)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.95)
        leg.set_zorder(1000)

        panel_letter = chr(ord("A") + panel_idx)
        panel_text = rf"\textbf{{{panel_letter}}}" if plt.rcParams.get("text.usetex", False) else panel_letter
        ax.text(
            -0.12,
            1.05,
            panel_text,
            transform=ax.transAxes,
            fontsize=panel_letter_fs,
            fontweight="bold",
            va="bottom",
            ha="left",
            zorder=1200,
        )
    fig.tight_layout()
    return fig


def run_supplementary_context_entropy_from_defaults(
    project_root: str,
    family: str = "both",
    out_path: str | None = None,
    use_latex: bool = True,
    max_msa_sequences: int | None = None,
):
    """Run only the supplementary context-dependent entropy figure from terminal."""
    fam_key = family.strip().lower()
    if fam_key == "both":
        family_configs = [
            _default_cd_entropy_family_config(project_root, "DBD"),
            _default_cd_entropy_family_config(project_root, "betaLac"),
        ]
    else:
        family_configs = [_default_cd_entropy_family_config(project_root, family)]

    fig = create_supplementary_context_dependent_entropy_figure(
        family_configs=family_configs,
        use_latex=use_latex,
        max_msa_sequences=max_msa_sequences,
    )

    if out_path is None:
        out_dir = os.path.join(project_root, "paper_figures_v3")
        os.makedirs(out_dir, exist_ok=True)
        fam_tag = "both" if fam_key == "both" else family
        out_path = os.path.join(out_dir, f"Supplementary_context_dependent_entropy_{fam_tag}.pdf")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved supplementary context-dependent entropy figure: {out_path}")
    return out_path


def create_figure14(
    sequences: list,
    mu_values: list,
    msa_folder: str,
    ancestral_save_folder: str,
    GT_sequences: dict,
    seq_to_label: dict,
    colors_dict: dict = None,
    n_samples: int = 100,
    data_prefix: str = "DBD",
    figsize: tuple = (16, 10),
):
    """
    Figure 14: Ancestral posterior scores vs distance to GT - one plot per GT sequence.
    
    Each of the 5 subplots shows one GT sequence with scatter points colored by mu value.
    Layout: 3 plots on top row (full width), 2 plots on bottom row (centered).
    
    Parameters:
        sequences: list of GT sequence identifiers (e.g., ["WT1", "WT2", ...])
        mu_values: list of mu_gen values to plot
        msa_folder: path to folder containing Felsenstein-sampled MSAs (M=1000)
        ancestral_save_folder: path to folder containing ancestral probability matrices
        GT_sequences: dict mapping sequence ID to GT sequence
        seq_to_label: dict mapping sequence ID to display label
        colors_dict: dict mapping sequence ID to color (optional, will create plasma if None)
        n_samples: number of random samples to select from each M=1000 MSA
        data_prefix: prefix for ancestral probability file names
        figsize: figure size tuple
    
    Returns: matplotlib figure object
    """
    from matplotlib.gridspec import GridSpec
    
    # Create colors_dict if not provided
    if colors_dict is None:
        cmap = cm.get_cmap('viridis')
        mu_colors = {mu: cmap(i / len(mu_values)) for i, mu in enumerate(mu_values)}
    else:
        mu_colors = {mu: cm.get_cmap('viridis')(i / len(mu_values)) for i, mu in enumerate(mu_values)}
    
    # Collect data: results_per_gt[seq] = [{"mu": ..., "yang_score": ..., "distance": ...}, ...]
    results_per_gt = {seq: [] for seq in sequences}
    
    for seq in sequences:
        GT = GT_sequences[seq]
        print(f"Loading data for {seq}...")
        
        for mu in mu_values:
            # Load Felsenstein-sampled MSA
            msa_filename = f"{seq}_mu={mu}_depth=None_M=1000"
            msa_path = os.path.join(msa_folder, msa_filename)
            
            if not os.path.exists(msa_path):
                print(f"Warning: MSA file not found: {msa_path}")
                continue
            
            msa = np.loadtxt(msa_path, dtype=int)
            
            # Randomly select n_samples if MSA has more sequences
            if len(msa) > n_samples:
                indices = np.random.choice(len(msa), n_samples, replace=False)
                msa = msa[indices]
            
            # Load ancestral probability matrix
            posterior_path = os.path.join(
                ancestral_save_folder,
                f"{data_prefix}_{seq}_mu{mu}_ancestral_probability"
            )
            
            if not os.path.exists(posterior_path):
                print(f"Warning: Posterior file not found: {posterior_path}")
                continue
            
            posterior = np.loadtxt(posterior_path)
            
            # Calculate yang score and hamming distance for each sampled sequence
            for sampled_seq in msa:
                yang_sc = yang_score(sampled_seq, posterior)
                hamming_dist = calculate_hamming_distance(sampled_seq, GT)
                
                results_per_gt[seq].append({
                    "mu": mu,
                    "yang_score": yang_sc,
                    "distance": hamming_dist
                })
    
    # Create figure with GridSpec layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 6, figure=fig, wspace=0.35, hspace=0.35)
    
    # Create axes: top row spans full width (3 plots), bottom row centered (2 plots with margins)
    axes = []
    axes.append(fig.add_subplot(gs[0, 0:2]))  # GT1
    axes.append(fig.add_subplot(gs[0, 2:4]))  # GT2
    axes.append(fig.add_subplot(gs[0, 4:6]))  # GT3
    axes.append(fig.add_subplot(gs[1, 1:3]))  # GT4 (centered)
    axes.append(fig.add_subplot(gs[1, 3:5]))  # GT5 (centered)
    
    # Plot each GT sequence
    for ax_idx, seq in enumerate(sequences):
        ax = axes[ax_idx]
        data_list = results_per_gt[seq]
        
        if not data_list:
            print(f"Warning: No data for {seq}")
            ax.text(0.5, 0.5, f"No data for {seq}", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            continue
        
        # Plot points for each mu value
        for mu in mu_values:
            mu_data = [d for d in data_list if d["mu"] == mu]
            if not mu_data:
                continue
            
            distances = np.array([d["distance"] for d in mu_data])
            yang_scores = np.array([d["yang_score"] for d in mu_data])
            
            ax.scatter(distances, yang_scores, color=mu_colors[mu], alpha=0.6, 
                      s=50, label=f"μ={mu:.1f}" if ax_idx == 0 else "")
        
        # Add y = -x + 1 reference line (normalized by sequence length)
        L = len(GT_sequences[seq])
        x_line = np.linspace(0, L, 100)
        y_line = 1 - (x_line / L)
        ax.plot(x_line, y_line, 'k--', alpha=0.3, linewidth=1.5, label="y=-x/L+1" if ax_idx == 0 else "")
        
        # Calculate and display Pearson correlation
        all_distances = np.array([d["distance"] for d in data_list])
        all_yang = np.array([d["yang_score"] for d in data_list])
        if len(all_distances) > 1:
            corr, _ = pearsonr(all_distances, all_yang)
            ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, 
                   fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5))
        
        # Labels and formatting
        if ax_idx in [0, 3]:  # Left subplots
            ax.set_ylabel("Ancestral Posterior Score", fontsize=16)
        ax.set_xlabel("Hamming Distance to GT", fontsize=16)
        
        seq_label = _format_gt_legend_label(seq_to_label.get(seq, seq))
        ax.set_title(seq_label, fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)
    
    # Create single legend for the figure
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=mu_colors[mu], 
                      markersize=8, label=f"μ={mu:.1f}") for mu in mu_values]
    handles.append(Line2D([0], [0], color='k', linestyle='--', linewidth=1.5, label="y=-x/L+1"))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
              ncol=len(mu_values) + 1, fontsize=14, frameon=True, fancybox=True)
    
    return fig


def create_figure15(
    sequences: list,
    mu_values: list,
    msa_folder: str,
    ancestral_save_folder: str,
    GT_sequences: dict,
    seq_to_label: dict,
    colors_dict: dict = None,
    n_samples: int = 100,
    data_prefix: str = "DBD",
    figsize: tuple = (20, 20),
):
    """
    Figure 15: Ancestral posterior scores vs distance to GT - one plot per mu AND per GT (5×5 grid = 25 panels).
    
    Rows: mu values (5), Columns: GT sequences (5)
    Each subplot shows ancestral posterior score vs hamming distance for a specific (mu, GT) combination.
    
    Parameters:
        sequences: list of GT sequence identifiers (e.g., ["WT1", "WT2", ...])
        mu_values: list of mu_gen values to plot
        msa_folder: path to folder containing Felsenstein-sampled MSAs (M=1000)
        ancestral_save_folder: path to folder containing ancestral probability matrices
        GT_sequences: dict mapping sequence ID to GT sequence
        seq_to_label: dict mapping sequence ID to display label
        colors_dict: dict mapping sequence ID to color (optional, will create plasma if None)
        n_samples: number of random samples to select from each M=1000 MSA
        data_prefix: prefix for ancestral probability file names
        figsize: figure size tuple (default 20x20)
    
    Returns: matplotlib figure object
    """
    from matplotlib.gridspec import GridSpec
    
    # Create colors_dict if not provided (for visualization if needed)
    if colors_dict is None:
        cmap = cm.get_cmap('plasma')
        color_values = np.linspace(0.1, 0.9, len(sequences))
        colors_dict = {seq: cmap(val) for seq, val in zip(sequences, color_values)}
    
    # Collect data keyed by (mu, seq): results[mu][seq] = list of {"yang_score": ..., "distance": ...}
    results = {mu: {seq: [] for seq in sequences} for mu in mu_values}
    
    for seq in sequences:
        GT = GT_sequences[seq]
        print(f"Loading data for {seq}...")
        
        for mu in mu_values:
            # Load Felsenstein-sampled MSA
            msa_filename = f"{seq}_mu={mu}_depth=None_M=1000"
            msa_path = os.path.join(msa_folder, msa_filename)
            
            if not os.path.exists(msa_path):
                print(f"Warning: MSA file not found: {msa_path}")
                continue
            
            msa = np.loadtxt(msa_path, dtype=int)
            
            # Randomly select n_samples if MSA has more sequences
            if len(msa) > n_samples:
                indices = np.random.choice(len(msa), n_samples, replace=False)
                msa = msa[indices]
            
            # Load ancestral probability matrix
            posterior_path = os.path.join(
                ancestral_save_folder,
                f"{data_prefix}_{seq}_mu{mu}_ancestral_probability"
            )
            
            if not os.path.exists(posterior_path):
                print(f"Warning: Posterior file not found: {posterior_path}")
                continue
            
            posterior = np.loadtxt(posterior_path)
            
            # Calculate yang score and hamming distance for each sampled sequence
            for sampled_seq in msa:
                yang_sc = yang_score(sampled_seq, posterior)
                hamming_dist = calculate_hamming_distance(sampled_seq, GT)
                
                results[mu][seq].append({
                    "yang_score": yang_sc,
                    "distance": hamming_dist
                })
    
    # Create figure with 5×5 grid
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(mu_values), len(sequences), figure=fig, wspace=0.25, hspace=0.40)
    
    # Get sequence length for reference line
    L = len(GT_sequences[sequences[0]]) if sequences else 1
    
    # Create axes and plot each (mu, seq) combination
    for mu_idx, mu in enumerate(mu_values):
        for seq_idx, seq in enumerate(sequences):
            ax = fig.add_subplot(gs[mu_idx, seq_idx])
            data_list = results[mu][seq]
            
            if not data_list or len(data_list) == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10)
                # Add minimal formatting
                ax.set_xlim(0, L)
                ax.set_ylim(0, 1)
                if seq_idx == 0:
                    ax.set_ylabel(rf'$\mu={mu}$', fontsize=12)
                if mu_idx == len(mu_values) - 1:
                    ax.set_xlabel(_format_gt_legend_label(seq_to_label.get(seq, seq)), fontsize=12)
                ax.tick_params(labelsize=9)
                continue
            
            # Plot scatter points
            distances = np.array([d["distance"] for d in data_list])
            yang_scores = np.array([d["yang_score"] for d in data_list])
            
            ax.scatter(distances, yang_scores, color=colors_dict[seq], alpha=0.6, s=40)
            
            # Add y = -x + 1 reference line (normalized by sequence length)
            x_line = np.linspace(0, L, 100)
            y_line = 1 - (x_line / L)
            ax.plot(x_line, y_line, 'k--', alpha=0.3, linewidth=1)
            
            # Set limits
            ax.set_xlim(0, L)
            ax.set_ylim(0, 1)
            
            # Calculate and display Pearson correlation
            if len(distances) > 1:
                corr, _ = pearsonr(distances, yang_scores)
                ax.text(0.05, 0.95, f"r={corr:.2f}", transform=ax.transAxes, 
                       fontsize=9, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
            
            # Row labels (mu values on left)
            if seq_idx == 0:
                ax.set_ylabel(rf'$\mu={mu}$', fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel("")
            
            # Column labels (GT sequences on bottom)
            if mu_idx == len(mu_values) - 1:
                seq_label = _format_gt_legend_label(seq_to_label.get(seq, seq))
                ax.set_xlabel(seq_label, fontsize=12, fontweight='bold')
            else:
                ax.set_xlabel("")
            
            # Tick labels only on edges
            if seq_idx > 0:
                ax.set_yticklabels([])
            if mu_idx < len(mu_values) - 1:
                ax.set_xticklabels([])
            
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.2)
    
    # Add overall axis labels
    fig.text(0.5, 0.02, 'Hamming Distance to GT', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.5, 'Ancestral Posterior Score', va='center', rotation='vertical', 
            fontsize=14, fontweight='bold')
    
    return fig


def create_figure16(
    sequences: list,
    mu_values: list,
    msa_folder: str,
    ancestral_save_folder: str,
    GT_sequences: dict,
    seq_to_label: dict,
    colors_dict: dict = None,
    n_samples: int = 100,
    data_prefix: str = "DBD",
    figsize: tuple = (12, 7),
):
    """
    Figure 16: Pearson correlation coefficient r vs mu_gen for each GT sequence.
    
    One colored curve per GT sequence, showing how the correlation between 
    ancestral posterior score and hamming distance changes with mu_gen.
    Correlations are computed using ALL 1000 sequences in each MSA.
    NaN correlation values are replaced with -1.
    
    Parameters:
        sequences: list of GT sequence identifiers (e.g., ["WT1", "WT2", ...])
        mu_values: list of mu_gen values to plot
        msa_folder: path to folder containing Felsenstein-sampled MSAs (M=1000)
        ancestral_save_folder: path to folder containing ancestral probability matrices
        GT_sequences: dict mapping sequence ID to GT sequence
        seq_to_label: dict mapping sequence ID to display label
        colors_dict: dict mapping sequence ID to color (optional, will create plasma if None)
        n_samples: ignored (for fig16, all 1000 sequences are used)
        data_prefix: prefix for ancestral probability file names
        figsize: figure size tuple
    
    Returns: matplotlib figure object
    """
    
    # Create colors_dict if not provided (for GT sequences)
    if colors_dict is None:
        cmap = cm.get_cmap('plasma')
        color_values = np.linspace(0.1, 0.9, len(sequences))
        colors_dict = {seq: cmap(val) for seq, val in zip(sequences, color_values)}
    
    # Collect correlation results: results[seq] = {"mu_values": [...], "r_values": [...]}
    results = {seq: {"mu_values": [], "r_values": []} for seq in sequences}
    
    for seq in sequences:
        GT = GT_sequences[seq]
        print(f"Computing correlations for {seq}...")
        
        # Need to sort mu_values for proper plotting
        sorted_mu_values = sorted(mu_values)
        
        for mu in sorted_mu_values:
            # Load Felsenstein-sampled MSA
            msa_filename = f"{seq}_mu={mu}_depth=None_M=1000"
            msa_path = os.path.join(msa_folder, msa_filename)
            
            if not os.path.exists(msa_path):
                print(f"Warning: MSA file not found: {msa_path}")
                continue
            
            msa = np.loadtxt(msa_path, dtype=int)
            
            # Load ancestral probability matrix
            posterior_path = os.path.join(
                ancestral_save_folder,
                f"{data_prefix}_{seq}_mu{mu}_ancestral_probability"
            )
            
            if not os.path.exists(posterior_path):
                print(f"Warning: Posterior file not found: {posterior_path}")
                continue
            
            posterior = np.loadtxt(posterior_path)
            
            # Calculate yang score and hamming distance for each sampled sequence
            distances = []
            yang_scores = []
            for sampled_seq in msa:
                yang_sc = yang_score(sampled_seq, posterior)
                hamming_dist = calculate_hamming_distance(sampled_seq, GT)
                distances.append(hamming_dist)
                yang_scores.append(yang_sc)
            
            distances = np.array(distances)
            yang_scores = np.array(yang_scores)
            
            # Calculate Pearson correlation
            if len(distances) > 1:
                corr, _ = pearsonr(distances, yang_scores)
                # Replace NaN with -1
                if np.isnan(corr):
                    corr = -1.0
            else:
                corr = -1.0
            
            results[seq]["mu_values"].append(mu)
            results[seq]["r_values"].append(corr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot one curve per sequence
    for seq in sequences:
        if not results[seq]["mu_values"]:
            print(f"Warning: No data for {seq}")
            continue
        
        mu_vals = np.array(results[seq]["mu_values"])
        r_vals = np.array(results[seq]["r_values"])
        
        # Sort by mu for proper line plotting
        sort_idx = np.argsort(mu_vals)
        mu_vals = mu_vals[sort_idx]
        r_vals = r_vals[sort_idx]
        
        seq_label = _format_gt_legend_label(seq_to_label.get(seq, seq))
        ax.plot(mu_vals, r_vals, 'o-', color=colors_dict[seq], linewidth=2.5, 
               markersize=8, alpha=0.8, label=seq_label)
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel(r'$\mu_{\mathrm{gen}}$', fontsize=16)
    ax.set_ylabel('Pearson Correlation Coefficient (r)', fontsize=16)
    ax.set_title('Correlation between Ancestral Posterior Score and Hamming Distance', 
                fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.axhline(y=-1, color='r', linestyle='--', linewidth=1.5, alpha=0.5, label='NaN → -1')
    
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=14)
    ax.tick_params(labelsize=14)
    
    # Set y-axis limits
    ax.set_ylim(-1.1, 1.0)
    
    plt.tight_layout()
    return fig


# ============================================================
# --- MAIN EXPORT FUNCTION ---
# ============================================================

def export_all_figures_to_pdf(
    output_pdf_path: str,
    # Figure 2 params
    natural_alignment: str,
    fasta_folder: str,
    mu_values: list,
    GT_sequences: dict,
    sequences: list,
    # Figure 3 & 4 params
    ancestral_save_folder: str,
    consensus_directory: str,
    # Figure 4 params
    M: int,
    T: float,
    msa_save_folder: str,
    couplings: np.ndarray,
    fields_: np.ndarray,
    # Figure 5 params
    mu_values_reduced: list = None,
    energy_keep_pct: float = None,
    # General
    use_latex: bool = True,
    data_prefix: str = "DBD",
    pca_alignment: str | None = None,
    figure2_cache_dir: str = "figure2_precomputed_data",
    figure2_use_cache: bool = True,
    figure2_refresh_cache: bool = False,
):
    """
    Export all paper figures (2, 3, 4, 5) to a single PDF with LaTeX captions.
    
    Parameters:
        output_pdf_path: Path to output PDF file
        [all other params correspond to the individual figure functions]
    """
    # Setup LaTeX if requested
    if use_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "legend.fontsize": 13,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14
        })
    
    # Create seq_to_label mapping
    seq_to_label = {seq: f"GT{i+1}" for i, seq in enumerate(sequences)}
    
    # Create colors_dict
    cmap_seq = cm.get_cmap('plasma')
    color_values = np.linspace(0.1, 0.9, len(sequences))
    colors_dict = {seq: cmap_seq(val) for seq, val in zip(sequences, color_values)}
    
    # Use reduced mu values for figures 4 and 5 if provided
    if mu_values_reduced is None:
        mu_values_reduced = [mu for mu in mu_values if 1.0 <= mu <= 100.0]
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_pdf_path) if os.path.dirname(output_pdf_path) else '.', exist_ok=True)
    
    with PdfPages(output_pdf_path) as pdf:
        # --- Figure 2 ---
        print("Generating Figure 2...")
        fig2 = create_figure2(
            natural_alignment=natural_alignment,
            fasta_folder=fasta_folder,
            mu_values=mu_values,
            GT_sequences=GT_sequences,
            sequences=sequences,
            seq_to_label=seq_to_label,
            data_prefix=data_prefix,
            num_bins=len(GT_sequences[sequences[0]]) + 1,
            pca_gt_in_legend=True,
            pca_alignment=pca_alignment,
            cache_dir=figure2_cache_dir,
            use_cache=figure2_use_cache,
            refresh_cache=figure2_refresh_cache,
        )
        
        # Add caption page
        fig_caption = plt.figure(figsize=(8.5, 11))
        caption_text = (
            r"\textbf{Figure 2: MSA diversity analysis.} "
            r"\textbf{(A)} Normalized Hamming distance (divided by sequence length) from generated sequences to the root sequence "
            r"as a function of $\mu_{\mathrm{gen}}$. "
            r"\textbf{(B)} Distribution of pairwise Hamming distances in generated MSAs "
            r"compared to the natural $\beta$-lactamase MSA (black line). "
            r"\textbf{(C--E)} PCA overlays for GT1, GT3, and GT5: natural density background and "
            r"$\mu_{\mathrm{gen}}\in\{1,10,50,3000\}$ contour outlines, with $\bm{s}^{\mathrm{GT}}$ marked by a diamond."
        )
        fig_caption.text(0.1, 0.9, caption_text, wrap=True, fontsize=12, va='top', ha='left',
                        transform=fig_caption.transFigure)
        pdf.savefig(fig_caption, bbox_inches='tight')
        plt.close(fig_caption)
        
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)
        
        # --- Figure 3 ---
        print("Generating Figure 3...")
        fig3 = create_figure3(
            mu_values=mu_values,
            ancestral_save_folder=ancestral_save_folder,
            GT_sequences=GT_sequences,
            consensus_directory=consensus_directory,
            sequences=sequences,
            seq_to_label=seq_to_label,
            colors_dict=colors_dict,
            data_prefix=data_prefix
        )
        
        # Add caption page
        fig_caption = plt.figure(figsize=(8.5, 11))
        caption_text = (
            r"\textbf{Figure 3: ML and Consensus reconstruction accuracy.} "
            r"\textbf{(A)} Normalized Hamming distance between ML reconstruction and ground truth (GT). "
            r"\textbf{(B)} Normalized Hamming distance between Consensus reconstruction and ground truth (GT). "
            r"\textbf{(C)} Normalized Hamming distance between ML and Consensus reconstructions. "
            r"\textbf{(D)} Difference $d_\text{H}(\bm{s}^{\mathrm{MAP}},\bm{s}^{\mathrm{GT}})-d_\text{H}(\bm{s}^{\mathrm{cons}},\bm{s}^{\mathrm{GT}})$: "
            r"negative values indicate ML is closer to GT than Consensus."
        )
        fig_caption.text(0.1, 0.9, caption_text, wrap=True, fontsize=12, va='top', ha='left',
                        transform=fig_caption.transFigure)
        pdf.savefig(fig_caption, bbox_inches='tight')
        plt.close(fig_caption)
        
        pdf.savefig(fig3, bbox_inches='tight')
        plt.close(fig3)
        
        # --- Figure 4 ---
        print("Generating Figure 4...")
        fig4, legend_handles, legend_labels = create_figure4(
            sequences=sequences,
            mu_values=mu_values_reduced,
            M=M,
            msa_save_folder=msa_save_folder,
            ancestral_probabilities_folder=ancestral_save_folder,
            consensus_directory=consensus_directory,
            GT_sequences=GT_sequences,
            couplings=couplings,
            fields_=fields_,
            T=T,
            seq_to_label=seq_to_label,
            data_prefix=data_prefix
        )
        
        # Add caption page
        fig_caption = plt.figure(figsize=(8.5, 11))
        caption_text = (
            r"\textbf{Figure 4: Hamming distance to GT vs Potts energy.} "
            r"Each panel shows the relationship between sequence Potts energy "
            r"and Hamming distance to the ground truth for a specific WT and mutation rate. "
            r"Bayesian samples (circles) and DCA-informed samples (diamonds) are shown, "
            r"along with the Consensus (triangle), ML (star), and GT (dashed line) positions."
        )
        fig_caption.text(0.1, 0.9, caption_text, wrap=True, fontsize=12, va='top', ha='left',
                        transform=fig_caption.transFigure)
        pdf.savefig(fig_caption, bbox_inches='tight')
        plt.close(fig_caption)
        
        pdf.savefig(fig4, bbox_inches='tight')
        plt.close(fig4)
        
        # Add legend page - wider and tighter vertically, WTs on first row, rest on second
        fig_leg = plt.figure(figsize=(22, 2.2))
        n_wt = len(sequences)
        # Keep Figure 4 legend in direct order: all WTs first, then all non-WT entries
        ordered_handles = legend_handles
        ordered_labels = legend_labels

        # Matplotlib fills multi-column legends column-wise; build an interleaved
        # handle/label list so that visual row 1 is all WTs, row 2 is the rest.
        wt_handles = ordered_handles[:n_wt]
        wt_labels = ordered_labels[:n_wt]
        other_handles = ordered_handles[n_wt:]
        other_labels = ordered_labels[n_wt:]

        # Pad second row to keep a strict 2-row layout aligned with WT columns
        while len(other_handles) < n_wt:
            other_handles.append(Line2D([], [], linestyle='', marker='', alpha=0))
            other_labels.append('')

        interleaved_handles = []
        interleaved_labels = []
        for h_wt, l_wt, h_ot, l_ot in zip(wt_handles, wt_labels, other_handles, other_labels):
            interleaved_handles.extend([h_wt, h_ot])
            interleaved_labels.extend([l_wt, l_ot])

        ncols = n_wt

        leg = fig_leg.legend(
            interleaved_handles,
            interleaved_labels,
            loc="center",
            ncol=ncols,
            fontsize=26,
            frameon=True,
            markerscale=1.3,
            handlelength=2.0,
            columnspacing=2.0,
            labelspacing=0.5,
            borderpad=1.0
        )
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_linewidth(1.3)
        leg.get_frame().set_facecolor("#f0f0f0")
        plt.axis("off")
        pdf.savefig(fig_leg, bbox_inches='tight')
        plt.close(fig_leg)
        
        # --- Figure 5 (Boxplots per WT) ---
        print("Generating Figure 5 boxplots...")
        boxplot_figures = create_figure5_boxplots(
            sequences=sequences,
            mu_values=mu_values_reduced,
            M=M,
            T=T,
            msa_folder=msa_save_folder,
            posterior_folder=ancestral_save_folder,
            GT_sequences=GT_sequences,
            fields_=fields_,
            couplings=couplings,
            seq_to_label=seq_to_label,
            colors_dict=colors_dict,
            energy_keep_pct=energy_keep_pct,
            data_prefix=data_prefix
        )
        
        # Add caption page for Figure 5
        fig_caption = plt.figure(figsize=(8.5, 11))
        caption_text = (
            r"\textbf{Figure 5: Boxplots of normalized Hamming distance to GT.} "
            r"For each root sequence, boxplots show the distribution of normalized Hamming distances to GT "
            r"for sequences selected by DCA scoring ($D_1$ = best, $D_{10}$ = 10 best) "
            r"and site-independent scoring ($S_{10}$ = 10 best). "
            r"The dashed gray line indicates the ML reconstruction distance."
        )
        fig_caption.text(0.1, 0.9, caption_text, wrap=True, fontsize=12, va='top', ha='left',
                        transform=fig_caption.transFigure)
        pdf.savefig(fig_caption, bbox_inches='tight')
        plt.close(fig_caption)
        
        for wt, fig in boxplot_figures:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    print(f"\n✓ All figures exported to: {output_pdf_path}")
    

def export_all_figures_to_individual_pdfs(config: dict, set_of_figures: list = [2,3,4,5]):
    """
    Export figures into separate PDF files.
    Filenames are automatically generated from output_pdf_path.
                Example of set_of_figures = [2,3,4,5,6,7,8,9,10,11]
            - 6: Supplementary Figure 2
            - 7: Supplementary Figure 3
                    - 8: Supplementary Figure 4
            - 9: Supplementary top-N candidates on Figure 4 grid
            - 10: Supplementary top-N candidates on reweighted PCA
            - 11: Supplementary MAP confidence vs MAP-to-GT distance
    """

    # ------------------------------------------------------------
    # Unpack config (same as quick_export)
    # ------------------------------------------------------------
    output_pdf_path = config["output_pdf_path"]
    natural_alignment = config["natural_alignment"]
    fasta_folder = config["fasta_folder"]
    mu_values = config["mu_values"]
    GT_sequences = config["GT_sequences"]
    sequences = config["sequences"]
    ancestral_save_folder = config["ancestral_save_folder"]
    consensus_directory = config["consensus_directory"]
    M = config["M"]
    T = config["T"]
    msa_save_folder = config["msa_save_folder"]
    couplings = config["couplings"]
    fields_ = config["fields_"]
    mu_values_reduced = config.get("mu_values_reduced")
    energy_keep_pct = config.get("energy_keep_pct")
    use_latex = config.get("use_latex", True)
    data_prefix = config.get("data_prefix", "DBD")
    pca_alignment = config.get("pca_alignment")
    figure2_cache_dir = config.get("figure2_cache_dir", "figure2_precomputed_data")
    figure2_use_cache = config.get("figure2_use_cache", True)
    figure2_refresh_cache = config.get("figure2_refresh_cache", False)
    supplementary_figure3_panels = tuple(config.get("supplementary_figure3_panels", ["A", "B", "C", "D"]))
    supplementary_candidate_wt_indices = config.get("supplementary_candidate_wt_indices", [1, 3, 5])
    supplementary_candidate_include_wt24 = config.get("supplementary_candidate_include_wt24", False)
    supplementary_candidate_include_wt24_fig6 = config.get("supplementary_candidate_include_wt24_fig6", True)
    supplementary_candidate_topN = int(config.get("supplementary_candidate_topN", 10))

    # ------------------------------------------------------------
    # Setup LaTeX
    # ------------------------------------------------------------
    if use_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "legend.fontsize": 13,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14
        })

    # ------------------------------------------------------------
    # Helper objects (same as original function)
    # ------------------------------------------------------------
    seq_to_label = {seq: f"GT{i+1}" for i, seq in enumerate(sequences)}

    cmap_seq = cm.get_cmap('plasma')
    color_values = np.linspace(0.1, 0.9, len(sequences))
    colors_dict = {seq: cmap_seq(val) for seq, val in zip(sequences, color_values)}

    if mu_values_reduced is None:
        mu_values_reduced = [mu for mu in mu_values if 1.0 <= mu <= 100.0]

    # ------------------------------------------------------------
    # Create output folder
    # ------------------------------------------------------------
    base_dir = os.path.splitext(output_pdf_path)[0]
    os.makedirs(base_dir, exist_ok=True)

    # ============================================================
    # ------------------- FIGURE 2 -------------------------------
    # ============================================================

    if 2 in set_of_figures:
        print("Generating Figure 2...")

        fig2 = create_figure2(
            natural_alignment=natural_alignment,
            fasta_folder=fasta_folder,
            mu_values=mu_values,
            GT_sequences=GT_sequences,
            sequences=sequences,
            seq_to_label=seq_to_label,
            data_prefix=data_prefix,
            num_bins=len(GT_sequences[sequences[0]]) + 1,
            pca_gt_in_legend=True,
            pca_alignment=pca_alignment,
            cache_dir=figure2_cache_dir,
            use_cache=figure2_use_cache,
            refresh_cache=figure2_refresh_cache,
        )

        fig2_name = "Figure_2.pdf"
        fig2.savefig(os.path.join(base_dir, fig2_name), bbox_inches="tight")
        plt.close(fig2)

    # ============================================================
    # ----------- SUPPLEMENTARY FIGURE 2 -------------------------
    # ============================================================
    if 6 in set_of_figures:
        print("Generating Supplementary Figure 2...")

        fig_s2 = create_supplementary_figure2(
            natural_alignment=natural_alignment,
            fasta_folder=fasta_folder,
            mu_values=mu_values,
            GT_sequences=GT_sequences,
            sequences=sequences,
            seq_to_label=seq_to_label,
            data_prefix=data_prefix,
            num_bins=len(GT_sequences[sequences[0]]) + 1
        )

        fig_s2.savefig(os.path.join(base_dir, "Supplementary_Figure_2.pdf"), bbox_inches="tight")
        plt.close(fig_s2)

    # ============================================================
    # ------------------- FIGURE 3 -------------------------------
    # ============================================================
    if 3 in set_of_figures:
        print("Generating Figure 3...")

        fig3 = create_figure3(
            mu_values=mu_values,
            ancestral_save_folder=ancestral_save_folder,
            GT_sequences=GT_sequences,
            consensus_directory=consensus_directory,
            sequences=sequences,
            seq_to_label=seq_to_label,
            colors_dict=colors_dict,
            data_prefix=data_prefix
        )

        fig3.savefig(os.path.join(base_dir, "Figure_3.pdf"), bbox_inches="tight")
        plt.close(fig3)

    # ============================================================
    # ----------- SUPPLEMENTARY FIGURE 3 -------------------------
    # ============================================================
    if 7 in set_of_figures:
        print("Generating Supplementary Figure 3...")

        fig_s3 = create_supplementary_figure3(
            mu_values=mu_values,
            ancestral_save_folder=ancestral_save_folder,
            GT_sequences=GT_sequences,
            consensus_directory=consensus_directory,
            sequences=sequences,
            seq_to_label=seq_to_label,
            colors_dict=colors_dict,
            fasta_folder=fasta_folder,
            data_prefix=data_prefix,
            panels_to_show=supplementary_figure3_panels,
        )

        fig_s3.savefig(os.path.join(base_dir, "Supplementary_Figure_3.pdf"), bbox_inches="tight")
        plt.close(fig_s3)

    # ============================================================
    # ----------- SUPPLEMENTARY FIGURE 3 BIS ----------------------
    # ============================================================
    if 7.5 in set_of_figures or "3bis" in str(set_of_figures):
        print("Generating Supplementary Figure 3 bis (Figure 2 font sizes)...")

        fig_s3bis = create_supplementary_figure3_bis(
            mu_values=mu_values,
            ancestral_save_folder=ancestral_save_folder,
            GT_sequences=GT_sequences,
            consensus_directory=consensus_directory,
            sequences=sequences,
            seq_to_label=seq_to_label,
            colors_dict=colors_dict,
            fasta_folder=fasta_folder,
            data_prefix=data_prefix,
            panels_to_show=("A", "B"),
        )

        fig_s3bis.savefig(os.path.join(base_dir, "Supplementary_Figure_3_bis.pdf"), bbox_inches="tight")
        plt.close(fig_s3bis)

    # ============================================================
    # ----------- SUPPLEMENTARY FIGURE 4 -------------------------
    # ============================================================
    if 8 in set_of_figures:
        print("Generating Supplementary Figure 4...")

        fig_s4 = create_supplementary_figure4(
            mu_values=mu_values,
            ancestral_save_folder=ancestral_save_folder,
            GT_sequences=GT_sequences,
            consensus_directory=consensus_directory,
            sequences=sequences,
            seq_to_label=seq_to_label,
            colors_dict=colors_dict,
            fasta_folder=fasta_folder,
            data_prefix=data_prefix
        )

        fig_s4.savefig(os.path.join(base_dir, "Supplementary_Figure_4.pdf"), bbox_inches="tight")
        plt.close(fig_s4)

    # ============================================================
    # ----------- SUPPLEMENTARY FIGURE 5 -------------------------
    # ============================================================
    if 9 in set_of_figures:
        print("Generating Supplementary Figure 5 (top-N on Figure 4 grid)...")

        fig_s5 = create_supplementary_top10_on_gridplot(
            sequences=sequences,
            mu_values=mu_values_reduced,
            M=M,
            T=T,
            msa_save_folder=msa_save_folder,
            ancestral_probabilities_folder=ancestral_save_folder,
            consensus_directory=consensus_directory,
            GT_sequences=GT_sequences,
            couplings=couplings,
            fields_=fields_,
            seq_to_label=seq_to_label,
            energy_keep_pct=energy_keep_pct,
            data_prefix=data_prefix,
            wt_indices=supplementary_candidate_wt_indices,
            include_wt24=True,
            topN=supplementary_candidate_topN,
        )

        fig_s5.savefig(os.path.join(base_dir, "Supplementary_Figure_5_topN_on_grid.pdf"), bbox_inches="tight")
        plt.close(fig_s5)

    # ============================================================
    # ----------- SUPPLEMENTARY FIGURE 6 -------------------------
    # ============================================================
    if 10 in set_of_figures:
        print("Generating Supplementary Figure 6 (top-N on reweighted PCA)...")

        fig_s6 = create_supplementary_top10_on_reweighted_pca(
            sequences=sequences,
            mu_values=mu_values_reduced,
            M=M,
            T=T,
            msa_save_folder=msa_save_folder,
            ancestral_probabilities_folder=ancestral_save_folder,
            GT_sequences=GT_sequences,
            fields_=fields_,
            couplings=couplings,
            seq_to_label=seq_to_label,
            pca_alignment=pca_alignment,
            energy_keep_pct=energy_keep_pct,
            data_prefix=data_prefix,
            wt_indices=list(range(1, len(sequences) + 1)),
            include_wt24=True,
            topN=supplementary_candidate_topN,
        )

        fig_s6.savefig(os.path.join(base_dir, "Supplementary_Figure_6_topN_on_reweighted_PCA.pdf"), bbox_inches="tight")
        plt.close(fig_s6)
        save_supplementary_top10_on_reweighted_pca_legend(
            os.path.join(base_dir, "Supplementary_Figure_6_topN_on_reweighted_PCA_legend.pdf"),
            topN=supplementary_candidate_topN,
            use_latex=use_latex,
        )

    # ============================================================
    # ----------- SUPPLEMENTARY FIGURE 7 -------------------------
    # ============================================================
    if 11 in set_of_figures:
        print("Generating Supplementary Figure 7 (MAP confidence vs MAP-to-GT distance)...")

        fig_s7 = create_supplementary_map_confidence_vs_gt_distance(
            mu_values=mu_values,
            ancestral_save_folder=ancestral_save_folder,
            GT_sequences=GT_sequences,
            sequences=sequences,
            seq_to_label=seq_to_label,
            colors_dict=colors_dict,
            data_prefix=data_prefix,
            use_latex=use_latex,
        )

        fig_s7.savefig(
            os.path.join(base_dir, "Supplementary_Figure_7_map_confidence_vs_MAP_to_GT.pdf"),
            bbox_inches="tight",
        )
        plt.close(fig_s7)

    # ============================================================
    # ------------------- FIGURE 14 (Ancestral posteriors vs GT distance - 1 per GT) ---
    # ============================================================
    if 14 in set_of_figures:
        print("Generating Figure 14 (Ancestral posteriors vs GT distance - 1 per GT)...")
        
        figure14_msa_folder = config.get("figure14_msa_folder", msa_save_folder)
        figure14_n_samples = config.get("figure14_n_samples", 100)
        
        fig14 = create_figure14(
            sequences=sequences,
            mu_values=mu_values,
            msa_folder=figure14_msa_folder,
            ancestral_save_folder=ancestral_save_folder,
            GT_sequences=GT_sequences,
            seq_to_label=seq_to_label,
            colors_dict=colors_dict,
            n_samples=figure14_n_samples,
            data_prefix=data_prefix,
        )
        
        fig14.savefig(os.path.join(base_dir, "Figure_14_ancestral_posteriors_per_GT.pdf"), bbox_inches="tight")
        plt.close(fig14)

    # ============================================================
    # ------------------- FIGURE 15 (Ancestral posteriors vs GT distance - 1 per mu) ---
    # ============================================================
    if 15 in set_of_figures:
        print("Generating Figure 15 (Ancestral posteriors vs GT distance - 1 per mu)...")
        
        figure15_msa_folder = config.get("figure15_msa_folder", msa_save_folder)
        figure15_n_samples = config.get("figure15_n_samples", 100)
        
        fig15 = create_figure15(
            sequences=sequences,
            mu_values=mu_values,
            msa_folder=figure15_msa_folder,
            ancestral_save_folder=ancestral_save_folder,
            GT_sequences=GT_sequences,
            seq_to_label=seq_to_label,
            colors_dict=colors_dict,
            n_samples=figure15_n_samples,
            data_prefix=data_prefix,
        )
        
        fig15.savefig(os.path.join(base_dir, "Figure_15_ancestral_posteriors_per_mu.pdf"), bbox_inches="tight")
        plt.close(fig15)

    # ============================================================
    # ------------------- FIGURE 16 (Correlation r vs mu_gen) ---
    # ============================================================
    if 16 in set_of_figures:
        print("Generating Figure 16 (Correlation r vs mu_gen)...")
        
        figure16_msa_folder = config.get("figure16_msa_folder", msa_save_folder)
        figure16_n_samples = config.get("figure16_n_samples", 100)
        
        fig16 = create_figure16(
            sequences=sequences,
            mu_values=mu_values,
            msa_folder=figure16_msa_folder,
            ancestral_save_folder=ancestral_save_folder,
            GT_sequences=GT_sequences,
            seq_to_label=seq_to_label,
            colors_dict=colors_dict,
            n_samples=figure16_n_samples,
            data_prefix=data_prefix,
        )
        
        fig16.savefig(os.path.join(base_dir, "Figure_16_correlation_vs_mu.pdf"), bbox_inches="tight")
        plt.close(fig16)

    # ============================================================
    # ------------------- FIGURE 4 ---------------------------------------
    # ============================================================
    if 4 in set_of_figures:
        print("Generating Figure 4...")

        fig4, legend_handles, legend_labels = create_figure4(
            sequences=sequences,
            mu_values=mu_values_reduced,
            M=M,
            msa_save_folder=msa_save_folder,
            ancestral_probabilities_folder=ancestral_save_folder,
            consensus_directory=consensus_directory,
            GT_sequences=GT_sequences,
            couplings=couplings,
            fields_=fields_,
            T=T,
            seq_to_label=seq_to_label,
            data_prefix=data_prefix
        )

        fig4.savefig(os.path.join(base_dir, "Figure_4.pdf"), bbox_inches="tight")
        plt.close(fig4)


        # Legend separately (wider and tighter vertically, WTs on first row, rest on second)
        fig_leg = plt.figure(figsize=(22, 2.2))
        n_wt = len(sequences)
        # Keep Figure 4 legend in direct order: all WTs first, then all non-WT entries
        ordered_handles = legend_handles
        ordered_labels = legend_labels

        # Matplotlib fills multi-column legends column-wise; build an interleaved
        # handle/label list so that visual row 1 is all WTs, row 2 is the rest.
        wt_handles = ordered_handles[:n_wt]
        wt_labels = ordered_labels[:n_wt]
        other_handles = ordered_handles[n_wt:]
        other_labels = ordered_labels[n_wt:]

        # Pad second row to keep a strict 2-row layout aligned with WT columns
        while len(other_handles) < n_wt:
            other_handles.append(Line2D([], [], linestyle='', marker='', alpha=0))
            other_labels.append('')

        interleaved_handles = []
        interleaved_labels = []
        for h_wt, l_wt, h_ot, l_ot in zip(wt_handles, wt_labels, other_handles, other_labels):
            interleaved_handles.extend([h_wt, h_ot])
            interleaved_labels.extend([l_wt, l_ot])

        ncols = n_wt

        leg = fig_leg.legend(
            interleaved_handles,
            interleaved_labels,
            loc="center",
            ncol=ncols,
            fontsize=26,
            frameon=True,
            markerscale=1.3,
            handlelength=2.0,
            columnspacing=2.0,
            labelspacing=0.5,
            borderpad=1.0
        )
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_linewidth(1.3)
        leg.get_frame().set_facecolor("#f0f0f0")
        plt.axis("off")

        fig_leg.savefig(os.path.join(base_dir, "Figure_4_Legend.pdf"), bbox_inches="tight")
        plt.close(fig_leg)

    # ============================================================
    # ------------------- FIGURE 5 -------------------------------
    # ============================================================
    if 5 in set_of_figures:
        print("Generating Figure 5 boxplots...")

        boxplot_figures = create_figure5_boxplots(
            sequences=sequences,
            mu_values=mu_values_reduced,
            M=M,
            T=T,
            msa_folder=msa_save_folder,
            posterior_folder=ancestral_save_folder,
            GT_sequences=GT_sequences,
            fields_=fields_,
            couplings=couplings,
            seq_to_label=seq_to_label,
            colors_dict=colors_dict,
            energy_keep_pct=energy_keep_pct,
            data_prefix=data_prefix
        )

        for wt, fig in boxplot_figures:
            filename = f"Figure_5_{wt}.pdf" if "Legend" not in wt else f"Figure_5_Legend.pdf"
            fig.savefig(
                os.path.join(base_dir, filename),
                bbox_inches="tight"
            )
            plt.close(fig)

    print(f"\n✓ All figures exported individually to folder: {base_dir}")

    if 5.5 in set_of_figures or "5bis" in str(set_of_figures):
        print("Generating Figure 5 bis (boxplots with Figure 2 font sizes)...")
        
        # Suppress figure display in Jupyter by turning off interactive mode
        import matplotlib
        plt.ioff()  # Turn off interactive mode
        
        # ================== Helper function to build result file paths dynamically ==================
        def build_figure5bis_result_files(dataset_prefix: str, mu_values_list: list):
            """
            Build result file paths for figure 5.5 based on dataset.
            
            Supports:
            - "final_betaLac": Beta-lactamase dataset (legacy format)
            - "DBD": DBD family dataset
            
            For DBD: Automatically discovers all available mu values from plddt_results folder
                    to ensure all pLDDT data is loaded, regardless of mu_values_reduced.
            
            Returns:
                plddt_files: dict mapping mu -> plddt CSV path
                rmsd_wt_files: dict mapping wt (1,3,5) -> rmsd CSV path (one file per WT)
                extant_rmsd: dict mapping wt_n (1,3,5) -> extant RMSD CSV path
                extant_plddt: str path to extant pLDDT CSV
            """
            if dataset_prefix == "DBD":
                # DBD folders: plddt_results, rmsd_results, extant_results
                # pLDDT: one file per mu value - discover all available mu values
                plddt_dir = "DBD/plddt_results"
                plddt_files = {}
                
                # Try to find all available pLDDT files in the results directory
                if os.path.exists(plddt_dir):
                    for fname in os.listdir(plddt_dir):
                        if fname.startswith("candidate_sequences_mu") and fname.endswith("_results.csv"):
                            # Extract mu value from filename (e.g., "candidate_sequences_mu5.5_results.csv" -> 5.5)
                            mu_str = fname.replace("candidate_sequences_mu", "").replace("_results.csv", "")
                            try:
                                mu_val = float(mu_str)
                                plddt_files[mu_val] = os.path.join(plddt_dir, fname)
                            except ValueError:
                                # Skip files that don't match expected format
                                pass
                
                # RMSD: one file per WT (contains all mu values in Sequence_ID format)
                rmsd_wt_files = {
                    1: "DBD/rmsd_results/candidate_pdbs_wt1_results.csv",
                    3: "DBD/rmsd_results/candidate_pdbs_wt3_results.csv",
                    5: "DBD/rmsd_results/candidate_pdbs_wt5_results.csv",
                }
                extant_rmsd = {
                    1: "DBD/extant_results/dbd_nat_spaced_pdbs_results_wt1.csv",
                    3: "DBD/extant_results/dbd_nat_spaced_pdbs_results_wt3.csv",
                    5: "DBD/extant_results/dbd_nat_spaced_pdbs_results_wt5.csv",
                }
                extant_plddt = "DBD/extant_results/dbd_nat_spaced_results.csv"
                
            else:  # final_betaLac (default)
                # Beta-lactamase folders with legacy naming (note: mu5.5 uses "5.5", mu5_5 uses "5_5" in RMSD)
                plddt_files = {
                    1:   "final_betaLac/plddt_results/candidate_sequences_mu1_results.csv",
                    5.5: "final_betaLac/plddt_results/candidate_sequences_mu5.5_results.csv",
                    20:  "final_betaLac/plddt_results/candidate_sequences_mu20_results.csv",
                    55:  "final_betaLac/plddt_results/candidate_sequences_mu55_results.csv",
                    100: "final_betaLac/plddt_results/candidate_sequences_mu100_results.csv"
                }
                rmsd_wt_files = {
                    1: "final_betaLac/rmsd_results/candidate_pdbs_wt1_results.csv",
                    3: "final_betaLac/rmsd_results/candidate_pdbs_wt3_results.csv",
                    5: "final_betaLac/rmsd_results/candidate_pdbs_wt5_results.csv",
                }
                extant_rmsd = {
                    1: "final_betaLac/extant_results/beta_nat_spaced_pdbs_results_wt1.csv",
                    3: "final_betaLac/extant_results/beta_nat_spaced_pdbs_results_wt3.csv",
                    5: "final_betaLac/extant_results/beta_nat_spaced_pdbs_results_wt5.csv",
                }
                extant_plddt = "final_betaLac/extant_results/beta_nat_spaced_results.csv"
            
            return plddt_files, rmsd_wt_files, extant_rmsd, extant_plddt
        
        # =======================================================================================
        # Determine which dataset to use based on data_prefix
        # =======================================================================================
        # Infer dataset from data_prefix: if data_prefix is "DBD", use DBD dataset; otherwise use betaLac
        dataset_to_use = "DBD" if data_prefix == "DBD" else "final_betaLac"
        
        plddt_files, rmsd_wt_files, extant_rmsd, extant_plddt = build_figure5bis_result_files(
            dataset_to_use, 
            mu_values_reduced
        )
        
        fig = create_figure5_bis_transposed(
            sequences=sequences,
            mu_values=mu_values_reduced,
            M=M, T=T,
            msa_folder=msa_save_folder,
            posterior_folder=ancestral_save_folder,
            GT_sequences=GT_sequences,
            fields_=fields_,
            couplings=couplings,
            seq_to_label=seq_to_label,
            plddt_result_files=plddt_files,
            rmsd_result_files=rmsd_wt_files,
            extant_rmsd_files=extant_rmsd,
            extant_plddt_file=extant_plddt,
            colors_dict=colors_dict,
            energy_keep_pct=energy_keep_pct,
            natural_alignment=natural_alignment,
            include_yang_likelihood_row=False,
            include_figure_legend=False,
            data_prefix=data_prefix,
        )
        fig.savefig(os.path.join(base_dir, "Figure_5_bis.pdf"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        save_figure5_bis_transposed_legend(
            os.path.join(base_dir, "Figure_5_bis_legend.pdf"),
            use_latex=use_latex,
        )
        print("✓ Figure 5 bis saved")

        # Same for the transposed and the supplementary
        # Use create_figure5_bis with a "transpose" flag to swap axes and adjust labels accordingly
        fig_s5bis = create_supplementary_figure5_bis_legacy(
            sequences=sequences,
            mu_values=mu_values_reduced,
            M=M, T=T,
            msa_folder=msa_save_folder,
            posterior_folder=ancestral_save_folder,
            GT_sequences=GT_sequences,
            fields_=fields_,
            couplings=couplings,
            seq_to_label=seq_to_label,
            plddt_result_files=plddt_files,
            rmsd_result_files=rmsd_wt_files,
            extant_rmsd_files=extant_rmsd,
            extant_plddt_file=extant_plddt,
            colors_dict=colors_dict,
            energy_keep_pct=energy_keep_pct,
            natural_alignment=natural_alignment,
            include_yang_likelihood_row=config.get("supplementary_figure5bis_include_yang_likelihood", False),
            data_prefix=data_prefix,
        )
        fig_s5bis.savefig(os.path.join(base_dir, "Supplementary_Figure_5_bis.pdf"), bbox_inches="tight", dpi=300)
        plt.close(fig_s5bis)
        print("✓ Supplementary Figure 5 bis saved")

        # Use the create_figure5_bis_supplementary function to create a version with only the top-N candidates and the GTs, and save as Supplementary Figure 5 ter
        fig_s5ter = create_figure5_bis_supplementary(
            sequences=sequences,
            mu_values=mu_values_reduced,
            M=M, T=T,
            msa_folder=msa_save_folder,
            posterior_folder=ancestral_save_folder,
            GT_sequences=GT_sequences,
            fields_=fields_,    
            couplings=couplings,
            seq_to_label=seq_to_label,
            plddt_result_files=plddt_files,
            rmsd_result_files=rmsd_wt_files,
            extant_rmsd_files=extant_rmsd,
            extant_plddt_file=extant_plddt,
            colors_dict=colors_dict,
            energy_keep_pct=energy_keep_pct,
            natural_alignment=natural_alignment,
            data_prefix=data_prefix,
        )
        fig_s5ter.savefig(os.path.join(base_dir, "Supplementary_Figure_5_ter.pdf"), bbox_inches="tight", dpi=300)
        plt.close(fig_s5ter)
        print("✓ Supplementary Figure 5 ter saved")

        fig_s5quat = create_figure5_quat(
            sequences=sequences,
            mu_values=mu_values_reduced,
            M=M, T=T,
            msa_folder=msa_save_folder,
            posterior_folder=ancestral_save_folder,
            GT_sequences=GT_sequences,
            fields_=fields_,
            couplings=couplings,
            seq_to_label=seq_to_label,
            plddt_result_files=plddt_files,
            rmsd_result_files=rmsd_wt_files,
            extant_rmsd_files=extant_rmsd,
            extant_plddt_file=extant_plddt,
            colors_dict=colors_dict,
            energy_keep_pct=energy_keep_pct,
            natural_alignment=natural_alignment,
            data_prefix=data_prefix,
        )
        fig_s5quat.savefig(os.path.join(base_dir, "Supplementary_Figure_5_quat.pdf"), bbox_inches="tight", dpi=300)
        plt.close(fig_s5quat)
        print("✓ Supplementary Figure 5 quat saved")
        print("\n✓ All Figure 5 variants generated successfully!")
        
        # Close all remaining figures to prevent display issues
        plt.close('all')
        
        # Re-enable interactive mode for subsequent notebook cells
        plt.ion()


# ============================================================
# --- CONVENIENCE FUNCTION ---
# ============================================================

def quick_export(config: dict, set_of_figures = [2,3,4,5,6,7,8,9,10,11] ,individual=False):
    """
    Quick export using a config dictionary.
    
    Expected config keys:
        - output_pdf_path: str
        - natural_alignment: str
        - fasta_folder: str
        - mu_values: list
        - GT_sequences: dict
        - sequences: list
        - ancestral_save_folder: str
        - consensus_directory: str
        - M: int
        - T: float
        - msa_save_folder: str
        - couplings: np.ndarray
        - fields_: np.ndarray
        - mu_values_reduced: list (optional)
        - energy_keep_pct: float (optional)
        - use_latex: bool (optional, default True)
        - data_prefix: str (optional, default "DBD") - prefix for ancestral probability files
        - pca_alignment: str (optional) - alignment to fit PCA on (if None, uses natural_alignment)
        - figure2_cache_dir: str (optional, default "figure2_precomputed_data")
        - figure2_use_cache: bool (optional, default True)
        - figure2_refresh_cache: bool (optional, default False)
        - supplementary_figure3_panels: list[str] (optional, default ["A","B","C","D"])
        - supplementary_candidate_wt_indices: list[int] (optional, default [1,3,5])
        - supplementary_candidate_include_wt24: bool (optional, default False)
        - supplementary_candidate_include_wt24_fig6: bool (optional, default True)
        - supplementary_candidate_topN: int (optional, default 10)
        - figure14_msa_folder: str (optional, default msa_save_folder) - folder for Figure 14 MSAs
        - figure14_n_samples: int (optional, default 100) - number of samples per MSA for Figure 14
        - figure15_msa_folder: str (optional, default msa_save_folder) - folder for Figure 15 MSAs
        - figure15_n_samples: int (optional, default 100) - number of samples per MSA for Figure 15
        - figure16_msa_folder: str (optional, default msa_save_folder) - folder for Figure 16 MSAs
        - figure16_n_samples: int (optional, default 100) - number of samples per MSA for Figure 16
    
    Figure selection (when individual=True):
        - 2, 3, 4, 5: Main figures
        - 6: Supplementary Figure 2
        - 7: Supplementary Figure 3
        - 8: Supplementary Figure 4
        - 9: Supplementary top-N candidates on Figure 4 grid
        - 10: Supplementary top-N candidates on reweighted PCA
        - 11: Supplementary MAP confidence vs MAP-to-GT distance
        - 14: Figure 14 - Ancestral posteriors vs GT distance (1 per GT)
        - 15: Figure 15 - Ancestral posteriors vs GT distance (5×5 grid per mu and GT)
        - 16: Figure 16 - Correlation coefficient r vs mu_gen (one curve per GT)
    """
    if individual:
        export_all_figures_to_individual_pdfs(config, set_of_figures)
    else:
        export_all_figures_to_pdf(
            output_pdf_path=config["output_pdf_path"],
            natural_alignment=config["natural_alignment"],
            fasta_folder=config["fasta_folder"],
            mu_values=config["mu_values"],
            GT_sequences=config["GT_sequences"],
            sequences=config["sequences"],
            ancestral_save_folder=config["ancestral_save_folder"],
            consensus_directory=config["consensus_directory"],
            M=config["M"],
            T=config["T"],
            msa_save_folder=config["msa_save_folder"],
            couplings=config["couplings"],
            fields_=config["fields_"],
            mu_values_reduced=config.get("mu_values_reduced"),
            energy_keep_pct=config.get("energy_keep_pct"),
            use_latex=config.get("use_latex", True),
            data_prefix=config.get("data_prefix", "DBD"),
            pca_alignment=config.get("pca_alignment"),
            figure2_cache_dir=config.get("figure2_cache_dir", "figure2_precomputed_data"),
            figure2_use_cache=config.get("figure2_use_cache", True),
            figure2_refresh_cache=config.get("figure2_refresh_cache", False),
        )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper-figure helper CLI")
    parser.add_argument(
        "--supp-context-entropy",
        action="store_true",
        help="Run only the supplementary context-dependent entropy figure.",
    )
    parser.add_argument(
        "--family",
        type=str,
        default="both",
        choices=["DBD", "betaLac", "both"],
        help="Family to process for supplementary context-entropy figure.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        help="Project root (defaults to repository root).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PDF path for supplementary context-entropy figure.",
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Disable LaTeX rendering (useful when TeX is unavailable).",
    )
    parser.add_argument(
        "--max-msa-sequences",
        type=int,
        default=0,
        help="Maximum number of natural MSA sequences to evaluate per family (<=0 means all).",
    )
    return parser


def _main_cli() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.supp_context_entropy:
        run_supplementary_context_entropy_from_defaults(
            project_root=args.project_root,
            family=args.family,
            out_path=args.out,
            use_latex=not args.no_latex,
            max_msa_sequences=(None if args.max_msa_sequences <= 0 else args.max_msa_sequences),
        )
        return

    parser.print_help()


if __name__ == "__main__":
    _main_cli()
