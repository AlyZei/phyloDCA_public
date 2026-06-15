"""
Create paper-ready figures of mu inference curves.

This module provides utilities for plotting mu inference curves across multiple mu values.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ete3 import Tree
from utils.inferringMu import fit_mu

# Configure LaTeX rendering for publication-quality figures
rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
})


def fit_mu_and_get_plot_data(tree_path, fasta_path):
    """
    Fit mu inference model and extract data for plotting.
    
    Returns:
        tuple: (x_vals, y_vals, x_fit, y_fit, mu_fit, a_fit, unique_distances, mean_hamming)
    """
    from scipy.optimize import curve_fit
    from scipy.spatial.distance import pdist, squareform
    from utils.toolsForTreesAndMSAs import read_fasta2
    import random

    tree = Tree(tree_path)
    sequences, name_to_seq, _ = read_fasta2(fasta_path)

    # Get leaves
    leaves = tree.get_leaves()
    max_leaves = 1000
    if len(leaves) > max_leaves:
        leaves = random.sample(leaves, max_leaves)
    n = len(leaves)

    # Precompute tree distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = round(leaves[i].get_distance(leaves[j]), 4)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Create sequence array
    seq_array = np.array([sequences[name_to_seq[leaf.name]] for leaf in leaves])

    # Vectorized Hamming distance ignoring gaps
    gap_code = 0
    def hamming_no_gaps(u, v):
        mask = (u != gap_code) & (v != gap_code)
        if mask.sum() == 0:
            return 0.0
        return np.sum(u[mask] != v[mask]) / mask.sum()

    # Compute all pairwise Hamming distances at once
    y_vals = pdist(seq_array, metric=hamming_no_gaps)

    # Flatten upper triangle of tree distances
    x_vals = squareform(dist_matrix)

    # Fully vectorized aggregation by unique tree distances
    unique_distances, inverse = np.unique(x_vals, return_inverse=True)
    sum_hamming = np.bincount(inverse, weights=y_vals)
    count_hamming = np.bincount(inverse)
    mean_hamming = sum_hamming / count_hamming

    # Define the model function
    def model_func(x_t, a, mu):
        return a * (1 - np.exp(-mu * x_t))

    # Fit the curve
    try:
        popt, _ = curve_fit(model_func, unique_distances, mean_hamming, p0=(1, 5), bounds=(0, np.inf))
        a_fit, mu_fit = popt
    except RuntimeError:
        print("Warning: Optimal parameters not found.")
        return None

    # Generate smooth fit curve
    x_fit = np.linspace(min(unique_distances), max(unique_distances), 100)
    y_fit = model_func(x_fit, *popt)

    return x_vals, y_vals, x_fit, y_fit, mu_fit, a_fit, unique_distances, mean_hamming


def plot_mu_inference_curves_4panel(
    tree_path,
    fasta_paths_dict,
    mu_values=[1, 10, 55, 100],
    wt_name="wt16682",
    figsize=(12, 10),
    save_path=None,
):
    """
    Create a 4-panel figure showing mu inference curves for different mu values.
    
    Parameters:
    -----------
    tree_path : str
        Path to the phylogenetic tree file
    fasta_paths_dict : dict
        Dictionary mapping mu values to fasta file paths
        Example: {1: 'path/to/file_mu1.0.fa', 10: 'path/to/file_mu10.0.fa', ...}
    mu_values : list
        List of 4 mu values to plot
    wt_name : str
        Name of the wild-type sequence
    figsize : tuple
        Figure size (width, height)
    save_path : str or None
        Path to save the figure
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    from scipy.optimize import curve_fit
    from scipy.spatial.distance import pdist, squareform
    from utils.toolsForTreesAndMSAs import read_fasta2
    import random

    if len(mu_values) != 4:
        raise ValueError("Please provide exactly 4 mu values for 4 panels")

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes_flat = axes.flatten()

    tree = Tree(tree_path)
    
    # Define the model function
    def model_func(x_t, a, mu):
        return a * (1 - np.exp(-mu * x_t))

    # Panel labels
    panel_labels = ['A', 'B', 'C', 'D']

    for idx, mu_val in enumerate(mu_values):
        ax = axes_flat[idx]
        
        if mu_val not in fasta_paths_dict:
            ax.text(0.5, 0.5, f'No data for $\\mu$ = {mu_val}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue
        
        fasta_path = fasta_paths_dict[mu_val]
        
        try:
            sequences, name_to_seq, _ = read_fasta2(fasta_path)

            # Get leaves
            leaves = tree.get_leaves()
            max_leaves = 1000
            if len(leaves) > max_leaves:
                leaves = random.sample(leaves, max_leaves)
            n = len(leaves)

            # Precompute tree distance matrix
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    d = round(leaves[i].get_distance(leaves[j]), 4)
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d

            # Create sequence array
            seq_array = np.array([sequences[name_to_seq[leaf.name]] for leaf in leaves])

            # Vectorized Hamming distance ignoring gaps
            gap_code = 0
            def hamming_no_gaps(u, v):
                mask = (u != gap_code) & (v != gap_code)
                if mask.sum() == 0:
                    return 0.0
                return np.sum(u[mask] != v[mask]) / mask.sum()

            # Compute all pairwise Hamming distances
            y_vals = pdist(seq_array, metric=hamming_no_gaps)

            # Flatten upper triangle of tree distances
            x_vals = squareform(dist_matrix)

            # Vectorized aggregation by unique tree distances
            unique_distances, inverse = np.unique(x_vals, return_inverse=True)
            sum_hamming = np.bincount(inverse, weights=y_vals)
            count_hamming = np.bincount(inverse)
            mean_hamming = sum_hamming / count_hamming

            # Fit the curve
            popt, _ = curve_fit(model_func, unique_distances, mean_hamming, 
                               p0=(1, 5), bounds=(0, np.inf))
            a_fit, mu_fit = popt

            # Generate smooth fit curve
            x_fit = np.linspace(min(unique_distances), max(unique_distances), 100)
            y_fit = model_func(x_fit, *popt)

            # Plot
            ax.scatter(unique_distances, mean_hamming, 
                      color='#1f77b4', s=60, alpha=0.6, 
                      edgecolors='black', linewidth=0.8, 
                      label='Data', zorder=5)
            ax.plot(x_fit, y_fit, 'r-', linewidth=3.0, 
                   label=f'Fit: $\\mu_{{\\mathrm{{fit}}}}$ = {mu_fit:.3f}', zorder=10)

            ax.set_xlabel(r'Tree distance $t$')
            ax.set_ylabel(r'Hamming distance $d_{\text{H}}$')
            ax.set_title(r'$\mu_{\text{gen}}$ = ' + f'{mu_val}', fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black')
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            
            # Add panel label outside the panel (top-left)
            ax.text(-0.12, 1.08, f'$\\mathbf{{{panel_labels[idx]}}}$', transform=ax.transAxes,
                    fontsize=22, fontweight='bold', va='top', ha='left', clip_on=False)
            
        except Exception as e:
            print(f"Error processing mu={mu_val}: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig, axes


if __name__ == "__main__":
    # Example usage
    import glob
    
    # Configuration
    tree_path = 'final_betaLac/betaLactree_collapsed_noonlychild_midpointrooted_normalized.nwk'
    beta_fasta_folder = 'Beta_replicates/'
    wt_name = 'wt16682'
    mu_values = [1, 10, 55, 100]
    
    # Find the first replicate for each mu value
    fasta_paths_dict = {}
    for mu in mu_values:
        pattern = f"{beta_fasta_folder}Beta_{wt_name}_mu{float(mu)}_*_repl1.fa"
        matches = glob.glob(pattern)
        if matches:
            fasta_paths_dict[mu] = matches[0]
            print(f"Found file for mu={mu}: {matches[0]}")
        else:
            print(f"No file found for mu={mu} with pattern {pattern}")
    
    # Create the figure
    fig, axes = plot_mu_inference_curves_4panel(
        tree_path=tree_path,
        fasta_paths_dict=fasta_paths_dict,
        mu_values=mu_values,
        wt_name=wt_name,
        figsize=(12, 10),
        save_path='mu_inference_curves_4panel.png'
    )
    
    plt.show()
