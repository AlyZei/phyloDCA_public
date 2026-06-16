# Done for the paper
# Works perfectly
# November 2025
# import necessary functions from files that should be deleted

import torch
import numpy as np
from typing import Dict

#from utils.shuffleFromFelsensteinDistrib import sample_site_probability
from utils.toolsForTreesAndMSAs import create_MSA_profile, get_maximum_likelihood_sequence
from utils.ci_and_cd_entropy import context_independent_entropy

def sample_site_probability(MSA: np.array, type: str = 'entropy') -> np.array:
    """
    Not normalized - matches your existing implementation.
    
    type: 'uniform', 'entropy' or '2^entropy'
    
    'uniform': all sites equally likely
    'entropy': site probabilities proportional to entropy(site) (default)
    '2^entropy': site probabilities proportional to 2^entropy(site) 
    """
    entropies = context_independent_entropy(MSA)

    if type == 'uniform':
        print('Site probabilities are uniform')
        return np.ones_like(entropies) / entropies.size
    elif type == '2^entropy':
        print('Site probabilities are computed as 2^entropy(site), non normalized')
        return 2**(entropies)
    else:
        print('Site probabilities are computed as entropy(site), non normalized')
        return entropies
    

def ensure_tensor(x, dtype=None):
    """Convert numpy array to torch tensor if needed."""
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=dtype)
    return x.to(dtype) if dtype is not None else x

def get_deltaE(
    idx: int,
    chain: torch.Tensor,
    residue_old: torch.Tensor,
    residue_new: torch.Tensor,
    params: Dict[str, torch.Tensor],
    L: int,
    q: int,
) -> torch.Tensor:
    """
    Compute ΔE for swapping a residue at site idx.
    chain: [N, L, q] one-hot
    residue_old: [N, q]
    residue_new: [N, q]
    """
    couplings = params["couplings"]  # [L, L, q, q]
    bias = params["bias"]            # [L, q]

    # Interaction energy of all other sites with site idx
    interaction = torch.einsum(
        "n l r, l r s -> n s",
        chain,              # [N, L, q]
        couplings[:, idx],  # [L, q, q]
    )  # -> [N, q]

    E_old = -(residue_old * (bias[idx] + interaction)).sum(dim=1)
    E_new = -(residue_new * (bias[idx] + interaction)).sum(dim=1)

    return E_new - E_old


def integer_to_onehot(sequences: torch.Tensor, q: int) -> torch.Tensor:
    """Convert integer-encoded sequences to one-hot encoding."""
    return torch.nn.functional.one_hot(sequences, num_classes=q).float()



def MCMC_columns_pytorch_efficient(
    MSA: np.array, 
    couplings: np.array, 
    fields_: np.array, 
    T: float, 
    scale=1, 
    device=None,
    q: int = 21,  # Number of amino acid states
    verbose: bool = True  # Print acceptance statistics
) -> np.array:
    """
    MCMC algorithm using the efficient get_deltaE function for pairwise swaps.
    
    Args:
        MSA: Multiple sequence alignment numpy array [M, L]
        couplings: Coupling parameters [L, L, q, q]
        fields_: Field parameters [L, q]
        T: Temperature parameter
        scale: Scaling factor for number of steps
        device: PyTorch device
        q: Number of amino acid states
        verbose: Print acceptance rate statistics
        
    Returns:
        Shuffled MSA as numpy array
    """
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
        
    # Print device info
    if verbose:
        print(f"Using device: {device}")
    
    # Convert MSA to tensor and then to one-hot
    MSA_tensor = torch.tensor(MSA, dtype=torch.int64, device=device)
    M, L = MSA_tensor.shape
    
    # Convert to one-hot encoding [M, L, q]
    MSA_onehot = integer_to_onehot(MSA_tensor, q)
    
    # Reshape parameters for efficient computation
    params = {
        "couplings": torch.tensor(couplings, dtype=torch.float32, device=device).view(L, L, q, q),
        "bias": torch.tensor(fields_, dtype=torch.float32, device=device).view(L, q)
    }
    for key in params:
        params[key] = params[key].to(device)
    
    # Compute site entropy and probabilities (using original MSA)
    site_entropy_array = sample_site_probability(MSA)
    total_entropy = np.sum(site_entropy_array)
    site_probability_array = site_entropy_array / total_entropy
    
    nb_steps = round(total_entropy * scale * M / 2)
    
    # Initialize counters for acceptance statistics
    total_proposed_swaps = 0
    total_accepted_swaps = 0
    
    if verbose:
        print(f"Starting MCMC with {nb_steps} steps, T={T}")
        print(f"MSA shape: {MSA.shape}")
        print(f"Coupling matrix shape: {params['couplings'].shape}")
        print(f"Bias shape: {params['bias'].shape}")
    
    for k in range(nb_steps):
        # Pick a random site to swap
        site = np.random.choice(np.arange(L), p=site_probability_array)
        
        # Generate permutation indices for pairwise swaps
        indices = torch.tensor(np.random.permutation(M), device=device)
        n_pairs = M // 2
        
        if n_pairs == 0:
            continue
            
        # Get pairs
        idx1 = indices[0:n_pairs*2:2]  # Even indices: 0, 2, 4...
        idx2 = indices[1:n_pairs*2:2]  # Odd indices: 1, 3, 5...
        
        # Get current amino acids at the site (integer form for comparison)
        aa1_int = MSA_tensor[idx1, site]
        aa2_int = MSA_tensor[idx2, site] 
        
        # Only process pairs with different amino acids
        diff_mask = (aa1_int != aa2_int)
        if not diff_mask.any():
            continue
            
        # Filter to valid pairs
        valid_idx1 = idx1[diff_mask]
        valid_idx2 = idx2[diff_mask]
        n_valid = len(valid_idx1)
        
        # Update proposed swaps counter
        total_proposed_swaps += n_valid
        
        # Get one-hot representations for current residues
        residue_old_1 = MSA_onehot[valid_idx1, site, :]  # [n_valid, q]
        residue_old_2 = MSA_onehot[valid_idx2, site, :]  # [n_valid, q]
        
        # Calculate energy differences using the efficient function
        # Energy change for sequence 1 when swapping to residue from sequence 2
        deltaE_1 = get_deltaE(
            site,
            MSA_onehot[valid_idx1],  # Original chain 1
            residue_old_1,           # Old residue (current)
            residue_old_2,           # New residue (from chain 2)
            params,
            L,
            q
        )
        
        # Energy change for sequence 2 when swapping to residue from sequence 1
        deltaE_2 = get_deltaE(
            site,
            MSA_onehot[valid_idx2],  # Original chain 2  
            residue_old_2,           # Old residue (current)
            residue_old_1,           # New residue (from chain 1)
            params,
            L,
            q
        )
        
        # Total energy change for the pairwise swap
        total_deltaE = deltaE_1 + deltaE_2
        
        # Metropolis acceptance criterion
        acceptance_probs = torch.exp(-total_deltaE / T)
        random_values = torch.rand(n_valid, device=device)
        accepted = random_values < acceptance_probs
        
        # Apply accepted swaps
        swap_idx1 = valid_idx1[accepted]
        swap_idx2 = valid_idx2[accepted]
        
        n_accepted = len(swap_idx1)
        total_accepted_swaps += n_accepted
        
        if n_accepted > 0:
            # Swap in both integer and one-hot representations
            # Integer representation
            temp_int = MSA_tensor[swap_idx1, site].clone()
            MSA_tensor[swap_idx1, site] = MSA_tensor[swap_idx2, site]
            MSA_tensor[swap_idx2, site] = temp_int
            
            # One-hot representation
            temp_onehot = MSA_onehot[swap_idx1, site, :].clone()
            MSA_onehot[swap_idx1, site, :] = MSA_onehot[swap_idx2, site, :]
            MSA_onehot[swap_idx2, site, :] = temp_onehot
    
    # Print final statistics
    if verbose:
        acceptance_rate = (total_accepted_swaps / total_proposed_swaps * 100) if total_proposed_swaps > 0 else 0.0
        print(f"\nMCMC Completed:")
        print(f"  Total steps: {nb_steps}")
        print(f"  Proposed swaps: {total_proposed_swaps}")
        print(f"  Accepted swaps: {total_accepted_swaps}")
        print(f"  Acceptance rate: {acceptance_rate:.2f}%")
    
    return MSA_tensor.cpu().numpy()


def shuffle_and_save_pytorch_optimized(scale: float, T: float, M: int, base_directory: str, consensus_directory:str,
                                      couplings: np.array, fields_:np.array, 
                                      msa_save_folder: str, sequences:list =None, mu_values:list = None, 
                                      depth:int = None, GT_sequences: dict = None):
    """
    Modified version of your shuffle_and_plot_pytorch function using optimized MCMC.
    Set use_optimization=False to use your original implementation.
    sequences = ['seq1', 'seq2', 'seq3']
    mu_values = ['2.13', '9.61', '42.88', '191.25', '3798.64']
    """
    # Your existing code until the MCMC call
    from utils.PottsEnergies import energy_of_msa, energy
    from utils.utils import get_all_file_paths, createFolder
    
    createFolder(msa_save_folder)
    
    filenames = get_all_file_paths(base_directory)
    
    file_dict = {}
    consensus_dict = {}
    consensus_dict_reweighted = {}

        
    for file in filenames:
            for seq in sequences:
                if seq in file:
                    for mu in mu_values:
                        if f'mu{mu:.1f}' in file:
                            file_dict[(seq, mu)] = file
        
    for file in get_all_file_paths(consensus_directory):
        for seq in sequences:
            if seq in file:
                for mu in mu_values:
                    if f'mu{mu:.1f}' in file:
                        if not 'reweighted' in file:
                            consensus_dict[(seq, mu)] = np.loadtxt(file, dtype=int)
                            #print(consensus_dict[(seq, mu)].dtype)

    for file in get_all_file_paths(consensus_directory):
        for seq in sequences:
            if seq in file:
                for mu in mu_values:
                    if f'mu{mu:.1f}' in file:
                        if 'reweighted' in file:
                            consensus_dict_reweighted[(seq, mu)] = np.loadtxt(file, dtype=int)
                            #print(consensus_dict[(seq, mu)].dtype)
 

    from tqdm import tqdm

    for i, seq in enumerate(sequences):
        for j, mu in enumerate(tqdm(mu_values, desc=f"mu for seq {i}", leave=False)):
            file = file_dict.get((seq, mu))
            if file:
                name = seq + '_mu=' + str(mu)
                print('Processing:', name)
                anc_prob = np.loadtxt(file, dtype=np.float64)
                ML_seq = get_maximum_likelihood_sequence(anc_prob)
                GT = GT_sequences[seq]
                MSA = create_MSA_profile(anc_prob, cardinal=M)

                # Load reshuffled MSA (from your MCMC step)
                reshuffled_MSA = MCMC_columns_pytorch_efficient(MSA, couplings, fields_, T=T, scale=scale)
                
                # Save MSA
                fname = name + f'_depth={depth}_M={M}'
                np.savetxt(msa_save_folder + fname, np.array(MSA).astype(int), fmt='%d')
                
                # Save reshuffled MSA
                fname = name + f'_depth={depth}_shuffled_M={M}_T={T}'
                np.savetxt(msa_save_folder + fname, np.array(reshuffled_MSA).astype(int), fmt='%d')