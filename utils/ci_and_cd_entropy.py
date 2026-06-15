import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from typing import List, Union, Optional, Tuple

from utils.toolsForTreesAndMSAs import getFrequencyDistribution


def context_independent_entropy(MSA: np.array)->np.array:
  """  Returns an array of length L, containing the entropy for each site
  in the MSA, calculated using the frequency distribution of amino acids at each site."""
  freqs=getFrequencyDistribution(MSA)
  L=freqs.shape[0]
  entropies=np.arange(L, dtype='float64')
  for site in range(L):
    entropies[site]=entropy(freqs[site,], base=2)
  return entropies

def context_dependent_entropy(sequence: np.array, site: int, fields, couplings, q: int = 21):
    """
    Calculate the context-dependent entropy at a specific site in a sequence using Potts model parameters.
    Args:
        sequence (np.array): The sequence in which to calculate the entropy.
        site (int): The index of the site for which to calculate the entropy.
        potts_parameters_lorenzo (str): The path to the Potts model parameters file.
        q (int): The number of states (default is 21 for amino acids).
    Returns:
        float: The context-dependent entropy at the specified site.
    """
    def context_dependent_probability(sequence, site, value, fields, couplings):
        """
        Calculate the context-dependent probability of a specific value at a given site in a sequence.
        """
        couplings_array = np.array([couplings[site, k, value, sequence[k]] for k in range(len(sequence)) if k != site])

        prob = np.exp(fields[site, value] + np.sum(couplings_array))

        return prob


    # Ensure sequence is a numpy array
    sequence = np.array(sequence)
    probabilites = np.array([context_dependent_probability(sequence, site, value, fields, couplings) for value in range(q)])
    # Normalize the probabilities
    probabilites /= np.sum(probabilites)
    return entropy(probabilites, base=2)

def context_dependent_entropy_sequence(sequence: np.array, fields, couplings, q: int = 21):
    """
    Calculate the context-dependent entropy for a specific site in a sequence using Potts model parameters.
    Args:
        sequence (np.array): The sequence in which to calculate the entropy.
        fields: The fields from the Potts model.
        couplings: The couplings from the Potts model. THEY SHOULD BE SYMMETRIZED.
        q (int): The number of states (default is 21 for amino acids).
    Returns:
        float: The mean context-dependent entropy over sites.
    """
    return np.mean(np.array([context_dependent_entropy(sequence, site, fields, couplings, q) for site in range(len(sequence))]))

# def context_dependent_entropy_msa(sequences: np.array, fields, couplings, q: int = 21):
#     """
#     Calculate the context-dependent entropy for a multiple sequence alignment (MSA) using Potts model parameters.
#     Args:
#         sequences (np.array): The MSA in which to calculate the entropy.
#         fields: The fields from the Potts model.
#         couplings: The couplings from the Potts model.
#         q (int): The number of states (default is 21 for amino acids).
#     """
#     #cde = lambda sequence: context_dependent_entropy_sequence(sequence, fields, couplings, q)
#     #return [cde(sequence) for sequence in sequences]
#     sequences_array = np.array(sequences)
#     return context_dependent_entropy_sequence(sequences_array, fields, couplings, q)

import torch

def context_dependent_entropy_msa_torch(sequences: np.array, fields, couplings, q: int = 21, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Calculate the context-dependent entropy for a multiple sequence alignment (MSA) using Potts model parameters.
    Parallelized using PyTorch.
    Args:
        sequences (np.array): The MSA in which to calculate the entropy (num_sequences, seq_length).
        fields: The fields from the Potts model (seq_length, q).
        couplings: The couplings from the Potts model (seq_length, seq_length, q, q).
        q (int): The number of states (default is 21 for amino acids).
        device (str): Device to use for computation ('cuda' or 'cpu').
    Returns:
        np.array: Context-dependent entropy for each sequence in the MSA.
    """
    # Convert to torch tensors
    sequences_tensor = torch.tensor(sequences, dtype=torch.long, device=device)  # (num_seqs, L)
    fields_tensor = torch.tensor(fields, dtype=torch.float32, device=device)  # (L, q)
    couplings_tensor = torch.tensor(couplings, dtype=torch.float32, device=device)  # (L, L, q, q)
    
    num_seqs, seq_length = sequences_tensor.shape
    
    # Initialize entropy array
    entropies = torch.zeros(num_seqs, device=device)
    
    # Loop over sites (this is harder to parallelize due to different coupling patterns)
    for site in range(seq_length):
        # For each site, calculate probabilities for all sequences and all possible values
        # Shape: (num_seqs, q)
        log_probs = torch.zeros(num_seqs, q, device=device)
        
        # Add field contribution
        log_probs += fields_tensor[site, :]  # Broadcasting: (q,) -> (num_seqs, q)
        
        # Add coupling contributions from all other sites
        for k in range(seq_length):
            if k != site:
                # Get the amino acid at position k for all sequences
                aa_at_k = sequences_tensor[:, k]  # (num_seqs,)
                
                # Get couplings for site-k interaction for all possible values at site
                # couplings_tensor[site, k, :, :] has shape (q, q)
                # We want couplings[site, k, value, sequence[k]] for all sequences and values
                
                # Extract relevant couplings: (q, q) -> index with aa_at_k
                coupling_contribution = couplings_tensor[site, k, :, aa_at_k]  # (q, num_seqs)
                log_probs += coupling_contribution.T  # (num_seqs, q)
        
        # Convert to probabilities
        probs = torch.softmax(log_probs, dim=1)  # (num_seqs, q)
        
        # Calculate entropy for this site across all sequences
        # Avoid log(0) by adding small epsilon
        site_entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=1)  # (num_seqs,)
        
        entropies += site_entropy
    
    # Return mean entropy over sites
    mean_entropies = entropies / seq_length
    
    return mean_entropies.cpu().numpy()

def plot_entropy_distributions(
    entropy_data: Union[np.ndarray, List[np.ndarray]], 
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    linewidth: float = 1.5,
    alpha: float = 0.9,
    title: str = 'Entropy per Site',
    xlabel: str = 'Site',
    ylabel: str = 'Entropy',
    save_path: Optional[str] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot entropy distributions with clean, modern styling.
    
    Parameters:
    -----------
    entropy_data : np.ndarray or list of np.ndarray
        Single entropy array or list of entropy arrays to plot
    labels : list of str, optional
        Labels for each entropy distribution
    colors : list of str, optional
        Colors for each line
    figsize : tuple, default (10, 6)
        Figure size (width, height)
    linewidth : float, default 1.5
        Line width for the plots
    alpha : float, default 0.9
        Line transparency
    title : str, default 'Entropy per Site'
        Plot title
    xlabel : str, default 'Site'
        X-axis label
    ylabel : str, default 'Entropy'
        Y-axis label
    save_path : str, optional
        Path to save the figure
    dpi : int, default 300
        DPI for saved figure
    
    Returns:
    --------
    plt.Figure
        The created figure object
    """
    
    # Convert single array to list
    if isinstance(entropy_data, np.ndarray):
        entropy_data = [entropy_data]
    
    # Set up the plot with clean styling
    plt.rcParams.update({
        'font.size': 11,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'legend.frameon': False,
        'figure.facecolor': 'white'
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Modern color palette
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Generate labels if none provided
    if labels is None:
        if len(entropy_data) > 1:
            labels = [f'Dataset {i+1}' for i in range(len(entropy_data))]
        else:
            labels = [None]
    
    # Plot each distribution
    for i, entropies in enumerate(entropy_data):
        sites = np.arange(len(entropies))
        color = colors[i % len(colors)]
        label = labels[i] if i < len(labels) else f'Dataset {i+1}'
        
        ax.plot(sites, entropies, 
               color=color, 
               linewidth=linewidth, 
               alpha=alpha,
               label=label)
    
    # Clean styling
    ax.set_title(title, fontsize=14, fontweight='600', pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Add legend only if multiple datasets
    if len(entropy_data) > 1:
        ax.legend(loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig

def compute_and_plot_entropy(
    prob_files: Union[str, List[str]],
    labels: Optional[List[str]] = None,
    **plot_kwargs
) -> plt.Figure:
    """
    Compute entropy from probability files and plot them.
    """
    if isinstance(prob_files, str):
        prob_files = [prob_files]
    
    entropy_distributions = []
    
    for file in prob_files:
        # Load probability data
        prob = np.loadtxt(file)
        L = prob.shape[0]
        entropies = np.zeros(L)
        
        # Compute entropy for each site
        for site in range(L):
            entropies[site] = entropy(prob[site, :], base=2)
        
        entropy_distributions.append(entropies)
    
    return plot_entropy_distributions(entropy_distributions, labels=labels, **plot_kwargs)

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Plot entropy distributions from probability files')
    parser.add_argument('files', nargs='*', help='Probability files to process')
    parser.add_argument('--labels', nargs='*', help='Labels for each distribution')
    parser.add_argument('--title', default='Entropy per Site', help='Plot title')
    parser.add_argument('--output', '-o', help='Output file path (e.g., entropy_plot.png)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[10, 6], help='Figure size (width height)')
    parser.add_argument('--no-show', action='store_true', help='Don\'t display plot')
    parser.add_argument('--examples', action='store_true', help='Run example plots')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saved figure')
    
    args = parser.parse_args()
    
    
    if args.files:
        try:
            fig = compute_and_plot_entropy(
                args.files,
                labels=args.labels,
                title=args.title,
                figsize=tuple(args.figsize),
                save_path=args.output,
                dpi=args.dpi
            )
            
            if not args.no_show:
                plt.show()
            
            if args.output:
                print(f"Plot saved to: {args.output}")
                
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        print("No files provided. Use --examples to see example plots or provide probability files.")
        parser.print_help()