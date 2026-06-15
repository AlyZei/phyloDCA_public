import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from utils.toolsForTreesAndMSAs import read_fasta2

def fit_mu(tree, file_path, max_leaves=1000, gap_code=0):
    # Read sequences
    sequences, name_to_seq, _ = read_fasta2(file_path)

    # Get leaves
    leaves = tree.get_leaves()
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
        print("Warning: Optimal parameters not found. Setting mu to 1.")
        return 1

    # Plot results
    x_fit = np.linspace(min(unique_distances), max(unique_distances), 100)
    y_fit = model_func(x_fit, *popt)

    plt.scatter(unique_distances, mean_hamming, label='Data Points MSA', color='blue')
    plt.plot(x_fit, y_fit, label='Fitted Curve', color='red')
    plt.xlabel('Distance on the tree')
    plt.ylabel('Hamming distance')
    plt.legend()
    plt.show()

    print(f"Fitted parameters: a = {a_fit}, mu = {mu_fit}. Fitted on {n} leaves. Fitted curve is d_hamming = {a_fit} * (1 - exp(-{mu_fit} * d_tree))")

    return mu_fit
