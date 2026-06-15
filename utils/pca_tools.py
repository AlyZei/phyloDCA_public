import numpy as np
from typing import Union, List
import torch

def one_hot_encode_msa(input: Union[str, np.ndarray, list], alphabet: List[str] = None, ignore_gaps_in_encoding: bool = True, seq_number_limit = np.inf) -> np.ndarray:
    """
    One-hot encode amino acid sequences using sklearn's OneHotEncoder.
    
    Args:
        fasta_path (str): Path to FASTA file
        alphabet (List[str]): Custom alphabet to use. If None, defaults based on ignore_gaps_in_encoding
        ignore_gaps_in_encoding (bool): Whether to exclude gap character from encoding alphabet
        seq_number_limit (int): Maximum number of sequences to include (for sampling)
    Returns:
        np.ndarray: One-hot encoded MSA of shape (n_sequences, seq_length, n_categories)
    """
    
    # shared alphabet used in PCA
    inv_mapping = {v: k for k, v in amino_acid_mapping_noX_noB.items()}  # 0->'-', 1->'A', ...

    def decode_numeric_to_str(msa_numeric: np.ndarray) -> list[str]:
        return ["".join(inv_mapping[i] for i in list(row)) for row in list(msa_numeric)]
    
    
    if isinstance(input, str):
        try:
            sequences = read_fasta_lit(input)[0]
        except Exception as e:
            print(f"Error reading FASTA file: {e}")
            try:
                msa_numeric = np.loadtxt(input, dtype=int)
                sequences = decode_numeric_to_str(msa_numeric)
            except Exception as e:
                print(f"Error loading array from path: {e}")
    elif isinstance(input, np.ndarray):
        if input.ndim == 1:
            sequences = decode_numeric_to_str([input])
        elif input.ndim == 2:
            sequences = decode_numeric_to_str(input)
    
    elif isinstance(input, list):
        # Check if list of integers (single sequence)
        if all(isinstance(x, int) for x in input):
            sequences = decode_numeric_to_str([np.array(input)])
        # Check if list of lists (multiple sequences)
        elif all(isinstance(x, list) and all(isinstance(i, int) for i in x) for x in input):
            sequences = decode_numeric_to_str(np.array(input))
        
    else:
        raise ValueError("Input array must be 1D or 2D.")
        
    # Check that sequences was defined
    if 'sequences' not in locals():
        raise ValueError("Either fasta_path or array_path must be provided.")
    if len(sequences) > seq_number_limit: #10000:
            print(f"⚠️ Warning: More than {seq_number_limit} sequences found.")
            sequences = np.random.permutation(sequences)[:seq_number_limit]
            print(f"Randomly sampled {seq_number_limit} sequences for encoding.")
    else:
        print(f"Total sequences found: {len(sequences)}. Proceeding with all sequences.")
        # Ask for user confirmation
        # user_input = input("Do you want to sample 10,000 sequences randomly? (y/n): ")
        # if user_input.lower() == 'y':
        #     # Randomly sample 10,000 sequences
        #     sequences=np.random.permutation(sequences)[:10000]
        # elif user_input.lower() != 'n':
        #     print("Invalid input. Proceeding with all sequences.")
    
    # Set default alphabet if not provided
    if alphabet is None:
        alphabet = np.array(list(AMINO_ACID_ALPHABET_NO_GAP)) if ignore_gaps_in_encoding else np.array(list(AMINO_ACID_ALPHABET_WITH_GAP))  # ← Fixed!
        #alphabet = np.array(list(AMINO_ACID_ALPHABET_NO_GAP)) if ignore_gaps_in_encoding else np.array(list(AMINO_ACID_ALPHABET))
    
        # Convert sequences to 2D character array
    seq_array = np.array([list(seq) for seq in sequences])
    print(f"Shape of seq_array: {seq_array.shape}")
    n_sequences, seq_length = seq_array.shape

    # Create encoder with consistent categories for all positions
    encoder = OneHotEncoder(
        categories=[alphabet], 
        sparse_output=False, 
        handle_unknown='ignore'
    )
    
    # Reshape for sklearn: (n_sequences * seq_length, 1)
    seq_flat = seq_array.reshape(-1, 1)
    print(f"Shape of seq_flat: {seq_flat.shape}")
    
    # Fit and transform in one step
    encoded_flat = encoder.fit_transform(seq_flat)
    
    # Reshape back to 3D: (n_sequences, seq_length, n_categories)
    #n_sequences=len(sequences)
    #seq_length = len(sequences[0])
    n_categories = len(alphabet)
    encoded_3d = encoded_flat.reshape(n_sequences, seq_length, n_categories)
    
    return encoded_3d

    # def encode_with_same_pipeline(msa_numeric: np.ndarray) -> np.ndarray:
    #     seqs = decode_numeric_to_str(msa_numeric)
    #     # Temporarily write to FASTA or adapt one_hot_encode_msa to accept lists
    #     encoded = one_hot_encode_msa_from_list(seqs, alphabet=alphabet,
    #                                         ignore_gaps_in_encoding=ignore_gaps_in_encoding)
    #     return encoded


def plot_pca_grid_msas_leo_trees(
    natural_fasta: str,
    sequences: list,
    mu_values: list,
    fasta_save_folder: str,
    consensus_directory: str,
    root_sequences: dict,
    save_folder: str,
    ignore_gaps_in_encoding: bool = True,
    plot_shuffled: bool = False,
    shuffled_folder: str = None
):
    """
    PCA plot of sequences in a grid for tree-generated data.
    Natural sequences in grey background. Fixed colors for MSA/consensus/reweighted consensus/root.
    
    Args:
        natural_fasta: Path to natural alignment FASTA
        sequences: List of WT sequences (e.g., ['wt13202', 'wt21394', ...])
        mu_values: List of mutation rates
        fasta_save_folder: Folder containing the tree MSA files
        consensus_directory: Folder containing consensus sequences
        root_sequences: Dict mapping sequence names to root sequences (as integer arrays)
        save_folder: Where to save output plot
        ignore_gaps_in_encoding: Whether to ignore gaps in one-hot encoding
        plot_shuffled: Whether to include shuffled MSAs (if available)
        shuffled_folder: Folder containing shuffled MSAs
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from utils.utils import createFolder, get_all_file_paths
    import os

    # --- Step 1: PCA on natural sequences ---
    print("Loading and encoding natural sequences...")
    natural_encoded = one_hot_encode_msa(natural_fasta, ignore_gaps_in_encoding=ignore_gaps_in_encoding)
    n_nat, L, q = natural_encoded.shape
    print(f"Natural MSA shape: {natural_encoded.shape}")
    
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    natural_flat = natural_encoded.reshape(n_nat, L*q)
    natural_scaled = scaler.fit_transform(natural_flat)
    natural_pca = pca.fit_transform(natural_scaled)
    explained_var = pca.explained_variance_ratio_
    print(f"PCA explained variance: PC1={explained_var[0]*100:.2f}%, PC2={explained_var[1]*100:.2f}%")

    # --- Step 2: Load consensus sequences ---
    print("\nLoading consensus sequences...")
    consensus_dict = {}
    consensus_dict_reweighted = {}
    
    for file in get_all_file_paths(consensus_directory):
        for seq in sequences:
            if seq in file:
                for mu in mu_values:
                    # Match the mu value in the filename
                    if f'mu{float(mu):.1f}' in file or f'mu{mu}' in file:
                        try:
                            if 'reweighted' in file:
                                consensus_dict_reweighted[(seq, mu)] = np.loadtxt(file, dtype=int)
                                print(f"  Loaded reweighted consensus: {seq} mu={mu}")
                            else:
                                consensus_dict[(seq, mu)] = np.loadtxt(file, dtype=int)
                                print(f"  Loaded consensus: {seq} mu={mu}")
                        except Exception as e:
                            print(f"  ⚠️ Error loading {file}: {e}")

    print(f"\nTotal consensus loaded: {len(consensus_dict)}")
    print(f"Total reweighted consensus loaded: {len(consensus_dict_reweighted)}")

    # --- Helper function to project sequences ---
    def project(encoded):
        """Project encoded sequences into PCA space"""
        if encoded.ndim == 2:  # Single sequence
            encoded = encoded[np.newaxis, :, :]
        flat = encoded.reshape(encoded.shape[0], L*q)
        scaled = scaler.transform(flat)
        return pca.transform(scaled)

    createFolder(save_folder)
    
    # Create figure
    fig, axes = plt.subplots(len(sequences), len(mu_values),
                             figsize=(4*len(mu_values), 4*len(sequences)),
                             squeeze=False)

    print("\nProcessing sequences...")
    for i, seq in enumerate(sequences):
        # Encode root sequence (already integer-encoded)
        root_seq = root_sequences[seq]
        root_enc = one_hot_encode_msa(root_seq, ignore_gaps_in_encoding=ignore_gaps_in_encoding)  # Pass as array/list
        root_proj = project(root_enc)
        
        for j, mu in enumerate(mu_values):
            ax = axes[i, j]
            
            # Construct filename for tree-generated MSA
            msa_file = os.path.join(
                fasta_save_folder,
                f"DBD_{seq}_mu{mu}_amino_DBDtree_collapsed_noonlychild_midpointrooted_mean1.fa"
            )
            
            # Check if file exists
            if not os.path.exists(msa_file):
                print(f"  ⚠️ Missing: {seq} mu={mu}")
                ax.set_title(f"{seq.replace('wt', 'WT ')} $\\mu$={mu}\n(File missing)")
                ax.axis("off")
                continue
            
            try:
                # Encode MSA
                print(f"  Processing: {seq} mu={mu}")
                msa_enc = one_hot_encode_msa(msa_file, seq_number_limit=2000, 
                                            ignore_gaps_in_encoding=ignore_gaps_in_encoding)
                msa_proj = project(msa_enc)
                
                # --- Plot natural background ---
                ax.scatter(natural_pca[:,0], natural_pca[:,1], 
                          color='lightgrey', alpha=0.3, s=10, 
                          label='Natural' if (i==0 and j==0) else '')
                
                # --- Plot MSA sequences ---
                ax.scatter(msa_proj[:,0], msa_proj[:,1], 
                          color='blue', marker='o', s=20, alpha=0.5, 
                          label='MSA leaves' if (i==0 and j==0) else '')
                
                # --- Plot consensus (if available) ---
                if (seq, mu) in consensus_dict:
                    try:
                        cons_enc = one_hot_encode_msa(consensus_dict[(seq, mu)], 
                                                     ignore_gaps_in_encoding=ignore_gaps_in_encoding)
                        cons_proj = project(cons_enc)
                        ax.scatter(cons_proj[:,0], cons_proj[:,1], 
                                  color='green', marker='s', s=80, 
                                  edgecolors='black', linewidths=1.5,
                                  label='Consensus' if (i==0 and j==0) else '',
                                  zorder=6)
                    except Exception as e:
                        print(f"    ⚠️ Error encoding consensus for {seq} mu={mu}: {e}")
                
                # --- Plot reweighted consensus (if available) ---
                if (seq, mu) in consensus_dict_reweighted:
                    try:
                        cons_rw_enc = one_hot_encode_msa(consensus_dict_reweighted[(seq, mu)],
                                                        ignore_gaps_in_encoding=ignore_gaps_in_encoding)
                        cons_rw_proj = project(cons_rw_enc)
                        ax.scatter(cons_rw_proj[:,0], cons_rw_proj[:,1], 
                                  color='purple', marker='D', s=80, 
                                  edgecolors='black', linewidths=1.5,
                                  label='Rew. Cons.' if (i==0 and j==0) else '',
                                  zorder=7)
                    except Exception as e:
                        print(f"    ⚠️ Error encoding reweighted consensus for {seq} mu={mu}: {e}")
                
                # --- Plot root/GT (WT) ---
                ax.scatter(root_proj[:,0], root_proj[:,1], 
                          color='red', marker='*', s=200, 
                          edgecolors='black', linewidths=1.5,
                          label='Root (WT)' if (i==0 and j==0) else '',
                          zorder=8)
                
                # --- Optional: Plot shuffled if available ---
                if plot_shuffled and shuffled_folder:
                    shuffled_file = os.path.join(
                        shuffled_folder,
                        f"{seq}_mu={mu}_shuffled_M=1000_T=0.2"
                    )
                    if os.path.exists(shuffled_file):
                        try:
                            shuffled_msa = np.loadtxt(shuffled_file, dtype=int)
                            shuffled_enc = one_hot_encode_msa(shuffled_msa,
                                                             ignore_gaps_in_encoding=ignore_gaps_in_encoding)
                            shuffled_proj = project(shuffled_enc)
                            ax.scatter(shuffled_proj[:,0], shuffled_proj[:,1], 
                                      color='orange', marker='s', s=20, alpha=0.5,
                                      label='Shuffled' if (i==0 and j==0) else '')
                        except Exception as e:
                            print(f"    ⚠️ Error with shuffled data: {e}")
                
                # Format axes
                # ax.set_title(f"{seq.replace('wt', 'WT ')} $\\mu$={mu}", fontsize=10)
                # if i == len(sequences)-1:
                #     ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)", fontsize=9)
                # if j == 0:
                #     ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)", fontsize=9)
                # ax.grid(True, linestyle='--', alpha=0.3)
                # ax.tick_params(labelsize=8)
                # Replace the axis formatting section with this:

                # Format axes
                ax.set_title(f"{seq.replace('wt', 'WT ')} $\\mu$={mu}", fontsize=10)
                if i == len(sequences)-1:
                    ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}\\%)", fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}\\%)", fontsize=9)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.tick_params(labelsize=6)
                                
            except Exception as e:
                print(f"  ❌ Error processing {seq} mu={mu}: {e}")
                import traceback
                traceback.print_exc()
                ax.set_title(f"{seq.replace('wt', 'WT ')} $\\mu$={mu}\n(Error)")
                ax.axis("off")

    # Add legend
    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
    
    fig.legend(handles, labels, loc='center right', fontsize=11, 
              title="Legend", frameon=True, fancybox=True, shadow=True)

    #plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.tight_layout(rect=(0, 0.05, 1, 0.98), pad=1.0, h_pad=1.2, w_pad=1.2)
    out_file = os.path.join(save_folder, "PCA_grid_tree_msas.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {out_file}")
    
    # Also save as PDF
    out_file_pdf = os.path.join(save_folder, "PCA_grid_tree_msas.pdf")
    plt.savefig(out_file_pdf, bbox_inches='tight')
    print(f"✓ Saved: {out_file_pdf}")
    
    plt.close()


@torch.jit.script
def _get_sequence_weight(s: torch.Tensor, data: torch.Tensor, L: int, th: float):
    seq_id = torch.sum(s == data, dim=1) / L
    n_clust = torch.sum(seq_id > th)
    
    return 1.0 / n_clust

#Or is it tracking this?

def compute_weights(
    data: np.ndarray, # or torch.Tensor,
    th: float = 0.8,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Computes the weight to be assigned to each sequence 's' in 'data' as 1 / n_clust, where 'n_clust' is the number of sequences
    that have a sequence identity with 's' >= th.

    Args:
        data (np.ndarray | torch.Tensor): Input dataset. Must be either a 2D or a 3D (one-hot encoded) array.
        th (float, optional): Sequence identity threshold for the clustering. Defaults to 0.8.
        device (torch.device, optional): Device. Defaults to "cpu".
        dtype (torch.dtype, optional): Data type. Defaults to torch.float32.

    Returns:
        torch.Tensor: Array with the weights of the sequences.
    """
    assert len(data.shape) == 2 or len(data.shape) == 3, "'data' must be either a 2D or a 3D array"
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, device=device)
    if len(data.shape) == 3:
        data_encoded = data.argmax(dim=2)
    else:
        data_encoded = data
    _, L = data_encoded.shape
    weights = torch.vstack([_get_sequence_weight(s, data_encoded, L, th) for s in data_encoded])

    return weights.to(dtype)

def resample_sequences(
    data: torch.Tensor,
    weights: torch.Tensor,
    nextract: int,
) -> torch.Tensor:
    """Extracts nextract sequences from data with replacement according to the weights.
    
    Args:
        data (torch.Tensor): Data array.
        weights (torch.Tensor): Weights of the sequences.
        nextract (int): Number of sequences to be extracted.

    Returns:
        torch.Tensor: Extracted sequences.
    """
    weights = weights.view(-1)
    indices = torch.multinomial(weights, nextract, replacement=True)
    
    return data[indices]   

if __name__=='__main__':
    # --- Usage ---
    natural_fasta = "DBD/DBD_alignment.uniref90.cov80.a2m"
    sequences = ['wt13202', 'wt21394', 'wt24786', 'wt4722', 'wt2748']
    mu_values = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.5, 10.0, 20.0, 
                30.0, 55.0, 100.0, 200.0, 300.0, 550.0, 1000.0, 2000.0, 3000.0]

    # Load root sequences
    seqs, names_to_ids, names = read_fasta2("DBD/DBD_alignment.uniref90.cov80.a2m")

    # Make suure seqs is a torch tensor
    seqs_tensor = torch.tensor(np.array(seqs), dtype=torch.int64)

    weights = compute_weights(seqs_tensor, th=0.8)

    pca_seqs = resample_sequences(seqs_tensor, weights, nextract=len(seqs)*3)

    file_path = natural_fasta+'_reweighted.fa'
    index = 1
    with open(file_path, 'w') as file:
        for seq in pca_seqs.numpy():
            file.write(f">N{index}\n{int_to_amino_acid_seq(seq)}\n") 
            index += 1

    GT_sequences = {
        'wt2748': seqs[2748-1],
        'wt13202': seqs[13202-1],
        'wt21394': seqs[21394-1],
        'wt24786': seqs[24786-1],
        'wt4722': seqs[4722-1]
    }

    # Run the plot
    plot_pca_grid_msas_leo_trees(
        natural_fasta=file_path,
        sequences=sequences,
        mu_values=mu_values,
        fasta_save_folder='saverioprova_DBDTree/',
        consensus_directory='consensus_sequences_full_DBD/',  # ← Add this
        root_sequences=GT_sequences,
        save_folder='figures/pca_full_dbd_tree_sampled_with_gaps/',
        ignore_gaps_in_encoding=False,
        plot_shuffled=False  # Set to True if you have shuffled MSAs
    )




