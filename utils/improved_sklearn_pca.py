import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import os
from typing import Tuple, Dict, List, Optional, Union

from utils.utils import createFolder
from utils.pca_tools import *
from utils.toolsForTreesAndMSAs import read_fasta2, int_to_amino_acid_seq

# Define amino acid alphabet - standard 20 amino acids + gap
AMINO_ACID_ALPHABET_WITH_GAP = "-ACDEFGHIKLMNPQRSTVWY"  # Gap as first character
AMINO_ACID_ALPHABET_NO_GAP = "ACDEFGHIKLMNPQRSTVWY"    # Without gap

def read_fasta_lit(file_path: str) -> tuple[np.ndarray, dict, list]:
    """
    Reads from a FASTA file and returns:
    - M x L array of sequences (MSA) where M is the number of sequences and L is the length of the sequences
    - A dictionary mapping each sequence name to its index in the MSA array
    - A list of sequence names in order

    Any sequences with non-standard characters or different lengths will be printed and skipped.

    Args:
        file_path (str): File containing the sequences in FASTA format.

    Returns:
        tuple[np.ndarray, dict, list]: (MSA, name_to_index, names)
    """
    sequences = []
    name_to_index = {}
    names = []
    current_sequence = []
    current_sequence_name = None
    expected_length = None  # Expected sequence length

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if line.startswith('>'):
                # Store the previous sequence if it exists
                if current_sequence_name is not None:
                    sequence_str = "".join(current_sequence)  

                    # Validate sequence before storing
                    try:
                        
                        if expected_length is None:
                            expected_length = len(sequence_str)

                        if len(sequence_str) != expected_length:
                            print(f"⚠️ Skipping sequence '{current_sequence_name}' - Length {len(seq_encoded)}, expected {expected_length}")
                        else:
                            sequences.append(sequence_str)
                            name_to_index[current_sequence_name] = len(sequences) - 1
                            names.append(current_sequence_name)

                    except KeyError as e:
                        print(f"❌ Error: Unexpected character '{e}' in sequence '{current_sequence_name}'. Check the FASTA file.")
                        continue  # Skip this sequence

                # Start a new sequence (extracting only the name, ignoring extra metadata)
                current_sequence_name = line[1:].split()[0]  # Remove '>' and take only the first part (before spaces)
                current_sequence = []  # Reset sequence storage
            else:
                # Ignore malformed lines (those with '>' inside sequences)
                if '>' in line:
                    print(f"⚠️ Skipping malformed line in sequence '{current_sequence_name}': {line}")
                    continue

                # Accumulate sequence lines (valid characters only)
                current_sequence.append(line)

        # Store the last sequence in the file
        if current_sequence_name is not None:
            sequence_str = "".join(current_sequence)

            try:
                if expected_length is None:
                    expected_length = len(sequence_str)

                if len(sequence_str) != expected_length:
                    print(f"⚠️ Skipping sequence '{current_sequence_name}' - Length {len(sequence_str)}, expected {expected_length}")
                else:
                    sequences.append(sequence_str)
                    name_to_index[current_sequence_name] = len(sequences) - 1
                    names.append(current_sequence_name)

            except KeyError as e:
                print(f"❌ Error: Unexpected character '{e}' in sequence '{current_sequence_name}'. Check the FASTA file.")
                pass  # Skip this sequence

    return (np.array(sequences, dtype=str), name_to_index, names)

# def one_hot_encode_msa(fasta_path: str, alphabet: List[str] = None, ignore_gaps_in_encoding: bool = True, seq_number_limit = np.inf) -> np.ndarray:
#     """
#     One-hot encode amino acid sequences using sklearn's OneHotEncoder.
    
#     Args:
#         fasta_path (str): Path to FASTA file
#         alphabet (List[str]): Custom alphabet to use. If None, defaults based on ignore_gaps_in_encoding
#         ignore_gaps_in_encoding (bool): Whether to exclude gap character from encoding alphabet
#         seq_number_limit (int): Maximum number of sequences to include (for sampling)
#     Returns:
#         np.ndarray: One-hot encoded MSA of shape (n_sequences, seq_length, n_categories)
#     """
#     # Read sequences
#     sequences = read_fasta_lit(fasta_path)[0]

#     if len(sequences) > seq_number_limit: #10000:
#         print(f"⚠️ Warning: More than {seq_number_limit} sequences found in {fasta_path}.")
#         sequences = np.random.permutation(sequences)[:seq_number_limit]
#         print(f"Randomly sampled {seq_number_limit} sequences for encoding.")
#     else:
#         print(f"Total sequences in {fasta_path}: {len(sequences)}. Proceeding with all sequences.")
#         # Ask for user confirmation
#         # user_input = input("Do you want to sample 10,000 sequences randomly? (y/n): ")
#         # if user_input.lower() == 'y':
#         #     # Randomly sample 10,000 sequences
#         #     sequences=np.random.permutation(sequences)[:10000]
#         # elif user_input.lower() != 'n':
#         #     print("Invalid input. Proceeding with all sequences.")
    
#     # Set default alphabet if not provided
#     if alphabet is None:
#         alphabet = np.array(list(AMINO_ACID_ALPHABET_NO_GAP)) if ignore_gaps_in_encoding else np.array(list(AMINO_ACID_ALPHABET_WITH_GAP))  # ← Fixed!
#         #alphabet = np.array(list(AMINO_ACID_ALPHABET_NO_GAP)) if ignore_gaps_in_encoding else np.array(list(AMINO_ACID_ALPHABET))
    
#     # Convert sequences to 2D character array
#     seq_array = np.array([list(seq) for seq in sequences])
#     print(f"Shape of seq_array: {seq_array.shape}")
#     n_sequences, seq_length = seq_array.shape

#     # Create encoder with consistent categories for all positions
#     encoder = OneHotEncoder(
#         categories=[alphabet], 
#         sparse_output=False, 
#         handle_unknown='ignore'
#     )
    
#     # Reshape for sklearn: (n_sequences * seq_length, 1)
#     seq_flat = seq_array.reshape(-1, 1)
#     print(f"Shape of seq_flat: {seq_flat.shape}")
    
#     # Fit and transform in one step
#     encoded_flat = encoder.fit_transform(seq_flat)
    
#     # Reshape back to 3D: (n_sequences, seq_length, n_categories)
#     #n_sequences=len(sequences)
#     #seq_length = len(sequences[0])
#     n_categories = len(alphabet)
#     encoded_3d = encoded_flat.reshape(n_sequences, seq_length, n_categories)
    
#     return encoded_3d

from utils.pca_tools import one_hot_encode_msa

def compare_msas_pca(fasta_paths: List[str], labels: Optional[List[str]] = None,
                    n_components: int = 5, save_folder: Optional[str] = None,
                    ignore_gaps_in_encoding: bool = True, file_name=None) -> None:
    """
    Compare multiple MSAs using PCA with enhanced aesthetics.
    
    Args:
        fasta_paths (List[str]): List of paths to FASTA files
        labels (Optional[List[str]]): Labels for each MSA (defaults to filenames)
        n_components (int): Number of principal components
        save_folder (Optional[str]): Folder to save results
        ignore_gaps_in_encoding (bool): Whether to exclude gap character from one-hot encoding dictionary
        file_name: Optional custom filename for saving
    """
    if save_folder:
        createFolder(save_folder)
    
    if labels is None:
        labels = [os.path.basename(path).split('.')[0] for path in fasta_paths]
    
    # Process first MSA (reference)
    seqs, names_to_ids, names = read_fasta2(fasta_paths[0])

    # Make suure seqs is a torch tensor
    seqs_tensor = torch.tensor(np.array(seqs), dtype=torch.int64)

    weights = compute_weights(seqs_tensor, th=0.8)

    pca_seqs = resample_sequences(seqs_tensor, weights, nextract=len(seqs)*3)

    file_path = fasta_paths[0].strip('.fasta').strip('.faa').strip('.fa')+'_reweighted.fa'
    index = 1
    with open(file_path, 'w') as file:
        for seq in pca_seqs.numpy():
            file.write(f">N{index}\n{int_to_amino_acid_seq(seq)}\n") 
            index += 1

    ref_encoded = one_hot_encode_msa(file_path, ignore_gaps_in_encoding=ignore_gaps_in_encoding)


    #ref_encoded = one_hot_encode_msa(fasta_paths[0], ignore_gaps_in_encoding=ignore_gaps_in_encoding)
    
    # Get dimensions for later use
    n_sequences, seq_length, n_categories = ref_encoded.shape
    
    # Flatten for PCA
    ref_flat = ref_encoded.reshape(n_sequences, seq_length * n_categories)
    
    # Standardize
    scaler = StandardScaler()
    ref_scaled = scaler.fit_transform(ref_flat)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    ref_pca_result = pca.fit_transform(ref_scaled)
    
    # Enhanced color palette - more sophisticated colors
    colors = ['#2E3440', '#BF616A', '#5E81AC', '#A3BE8C', '#B48EAD', '#D08770', '#EBCB8B', '#88C0D0']
    
    # Create figure with better proportions
    fig = plt.figure(figsize=(16, 14))
    
    # Set style for better aesthetics
    plt.style.use('default')  # Reset to clean style
    
    # Create GridSpec with more space for text elements
    gs = gridspec.GridSpec(n_components, n_components, figure=fig,
                          hspace=0.25, wspace=0.25,
                          left=0.08, right=0.75, top=0.88, bottom=0.08)
    
    # Track axes for different PC pairs
    axes_dict = {}
    
    # Create scatter plots in lower triangle with enhanced styling
    for i in range(n_components):
        for j in range(n_components):
            if i > j:  # Lower triangle - scatter plots
                ax = fig.add_subplot(gs[i, j])
                
                # Enhanced scatter plot styling
                scatter = ax.scatter(ref_pca_result[:, j], ref_pca_result[:, i],
                                   c=colors[0], s=20, alpha=0.7, 
                                   edgecolors='white', linewidth=0.5,
                                   label=labels[0])
                
                # Store axis for later
                axes_dict[(i, j)] = ax
                
                # Enhanced axis styling
                ax.set_xlabel(f"PC{j+1}", fontsize=11, fontweight='bold')
                ax.set_ylabel(f"PC{i+1}", fontsize=11, fontweight='bold')
                
                # Add subtle grid
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                ax.set_facecolor('#FAFAFA')
                
                # Style the spines
                for spine in ax.spines.values():
                    spine.set_color('#CCCCCC')
                    spine.set_linewidth(1)
                
            elif i == j:  # Diagonal - enhanced density plots
                ax = fig.add_subplot(gs[i, i])
                
                # Enhanced histogram/density plot
                sns.histplot(ref_pca_result[:, i], kde=True, color=colors[0], 
                           alpha=0.6, label=labels[0], ax=ax,
                           stat='density', element='step', fill=True,
                           edgecolor='white', linewidth=1.5)
                
                # Store axis
                axes_dict[(i, i)] = ax
                ax.set_xlabel(f"PC{i+1}", fontsize=11, fontweight='bold')
                ax.set_ylabel("Density", fontsize=11, fontweight='bold')
                
                # Enhanced styling for density plots
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                ax.set_facecolor('#FAFAFA')
                
                # Style the spines
                for spine in ax.spines.values():
                    spine.set_color('#CCCCCC')
                    spine.set_linewidth(1)
    
    # Process other MSAs with enhanced styling
    for idx, (path, label) in enumerate(zip(fasta_paths[1:], labels[1:]), 1):
        color_idx = idx % len(colors)
        
        # Read and encode
        encoded = one_hot_encode_msa(path, ignore_gaps_in_encoding=ignore_gaps_in_encoding)
        
        # Skip if dimensions don't match
        if encoded.shape[1:] != ref_encoded.shape[1:]:
            print(f"Warning: MSA dimensions don't match! Reference: {ref_encoded.shape}, {label}: {encoded.shape}")
            continue
        
        # Flatten and transform using same scaler and PCA
        flat = encoded.reshape(encoded.shape[0], seq_length * n_categories)
        scaled = scaler.transform(flat)
        pca_result = pca.transform(scaled)
        
        # Add to existing plots with enhanced styling
        for i in range(n_components):
            for j in range(n_components):
                if i > j:  # Scatter plots
                    ax = axes_dict[(i, j)]
                    ax.scatter(pca_result[:, j], pca_result[:, i],
                              c=colors[color_idx], s=20, alpha=0.7,
                              edgecolors='white', linewidth=0.5,
                              label=label)
                elif i == j:  # Enhanced density plots
                    ax = axes_dict[(i, i)]
                    sns.histplot(pca_result[:, i], kde=True, color=colors[color_idx],
                               alpha=0.6, label=label, ax=ax,
                               stat='density', element='step', fill=True,
                               edgecolor='white', linewidth=1.5)
    
    # Create variance information box with better positioning
    variance_text = "Explained Variance:\n" + "\n".join([
        f"PC{i+1}: {var:.1%}" for i, var in enumerate(pca.explained_variance_ratio_)
    ]) + f"\nCumulative: {sum(pca.explained_variance_ratio_):.1%}"
    
    # Add variance info using fig.text for absolute positioning
    fig.text(0.78, 0.70, variance_text, 
            fontsize=10, fontweight='bold',
            bbox={'facecolor': '#E8E8E8', 'alpha': 0.9, 'pad': 8,
                  'boxstyle': 'round,pad=0.5', 'edgecolor': '#CCCCCC'},
            verticalalignment='top', horizontalalignment='left')
    
    # Create legend using fig.text area
    handles = []
    for i, label in enumerate(labels):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=colors[i % len(colors)], 
                                markersize=10, markeredgecolor='white',
                                markeredgewidth=1, label=label))
    
    # Create a temporary invisible axis for the legend
    legend_ax = fig.add_axes([0.78, 0.25, 0.2, 0.25])
    legend_ax.axis('off')
    legend_ax.legend(handles=handles, loc='upper left', frameon=True,
                    fancybox=True, shadow=True, framealpha=0.9,
                    facecolor='#F5F5F5', edgecolor='#CCCCCC')
    
    # Enhanced title
    title = f"PCA comparison of {labels[0]} and {labels[1]}"
    if ignore_gaps_in_encoding:
        title += "\n(gaps ignored in encoding)"
    
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.95,
                bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10,
                      'boxstyle': 'round,pad=0.5', 'edgecolor': '#CCCCCC'})
    
    # No need for tight_layout since we're using explicit positioning
    
    # Save or display
    if save_folder:
        if not file_name:
            filename = f"msa_pca_{labels[0]}_vs_{labels[1]}"
        else:
            filename = file_name
        if ignore_gaps_in_encoding:
            filename += "_ignore_gaps_encoding"
        save_path = os.path.join(save_folder, filename + ".png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Enhanced comparison plot saved to {save_path}")
    else:
        plt.show()