
import numpy as np
import torch

from matplotlib import pyplot as plt
from typing import Tuple, List, Union

from utils.energy_plotting_mahaut import reshape_couplings
from utils.toolsForTreesAndMSAs import read_fasta2
from utils.utils import generate_colors
from utils.utils_lore import load_params

def ensure_tensor(x, dtype=None):
    """
    Safely convert numpy or tensor to torch.Tensor without warnings.
    Keeps tensors intact, only converts numpy/array-like.
    """
    # Look for available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.is_tensor(x):
        return x.to(dtype=dtype or x.dtype, device=device or x.device)
    return torch.tensor(x, dtype=dtype, device=device)



def symmetrize_couplings(couplings: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Symmetrize coupling matrix by filling zeros with corresponding symmetric values."""
    is_torch = torch.is_tensor(couplings)
    if is_torch:
        device = couplings.device
        L, _, q, _ = couplings.shape
        
        # Create indices for all positions
        k_idx, site_idx, val_idx, val2_idx = torch.meshgrid(
            torch.arange(L, device=device),
            torch.arange(L, device=device),
            torch.arange(q, device=device),
            torch.arange(q, device=device),
            indexing='ij'
        )
        
        # Get symmetric positions
        symmetric_vals = couplings[site_idx, k_idx, val2_idx, val_idx]
        
        # Only update where current value is 0
        zero_mask = (couplings == 0.0)
        couplings = torch.where(zero_mask, symmetric_vals, couplings)
        
        return couplings
    else:
        # Original numpy implementation for compatibility
        L, _, q, _ = couplings.shape
        for k in range(L):
            for site in range(L):
                for value in range(q):
                    for value2 in range(q):
                        c1 = couplings[k, site, value, value2]
                        c2 = couplings[site, k, value2, value]
                        if c1 == 0.0:
                            couplings[k, site, value, value2] = c2
                        else:
                            couplings[site, k, value2, value] = c1
        return couplings

def verify_symmetry(couplings: np.ndarray, tol: float = 1e-6):
    """
    Verify that couplings are symmetric: J[i,j,a,b] == J[j,i,b,a]
    """
    symmetric_diff = np.abs(couplings - couplings.transpose(1,0,3,2))
    if not np.all(symmetric_diff < tol):
        raise ValueError("Couplings are not symmetric!")
    print("Couplings verified as symmetric.")


def readParametersFraZ(file_path: str, q: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read Potts parameters from Francesco format file.
    
    Args:
        file_path: Path to parameter file
        q: Number of states (default 21 for amino acids)
    
    Returns:
        Tuple of (couplings, fields) arrays
    """
    couplings = {}
    fields_ = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            fields = line.strip().split()
            if fields[0] == 'J':
                position_i, position_j = int(fields[1]), int(fields[2])
                amino_i, amino_j = int(fields[3]), int(fields[4])
                interaction_values = float(fields[5])        
                couplings[(position_i, position_j, amino_i, amino_j)] = interaction_values
            elif fields[0] == 'h':
                position = int(fields[1])
                amino = int(fields[2])
                field_val = float(fields[3])
                fields_[(position, amino)] = field_val

    # Get sequence length
    L = len(fields_) // q
    print('L:', L)

    # Initialize arrays
    couplings_arr = np.zeros((int(L), int(L), q, q), dtype=np.float32)
    fields_arr = np.zeros((int(L), q), dtype=np.float32)

    # Fill arrays
    for key, value in couplings.items():
        couplings_arr[key[0], key[1], key[2], key[3]] = value
    for key, value in fields_.items():
        fields_arr[key[0], key[1]] = value
    
    # Symmetrize couplings
    couplings_arr = symmetrize_couplings(couplings_arr)
    return couplings_arr, fields_arr


def get_fields_and_couplings_from_params(
    potts_parameters_lorenzo: str,
    device: Union[str, torch.device, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load fields and couplings from Lorenzo's format parameter file.
    
    Args:
        potts_parameters_lorenzo: Path to parameter file
    
    Returns:
        Tuple of (fields, couplings) arrays
    """
    if device is None:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    params = load_params(
        potts_parameters_lorenzo,
        tokens='protein',
        device=resolved_device,
    )

    fields = params.get("bias")
    couplings = params.get("coupling_matrix")

    if fields is None or couplings is None:
        raise ValueError(f"Could not load parameters from {potts_parameters_lorenzo}")

    # Convert to numpy
    if torch.is_tensor(couplings):
        couplings = couplings.detach().cpu().numpy()
    if torch.is_tensor(fields):
        fields = fields.detach().cpu().numpy()

    # Reshape and symmetrize
    couplings = reshape_couplings(couplings)
    print("Coupling matrix shape:", couplings.shape)
    print("Fields shape:", fields.shape)
    couplings = symmetrize_couplings(couplings)

    return fields, couplings


def read_potts_parameters_proteins(
    potts_parameters_path: str,
    q: int = 21,
    device: Union[str, torch.device, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read Potts model parameters with fallback between formats.
    
    Args:
        potts_parameters_path: Path to parameter file
        q: Number of states (default 21)
    
    Returns:
        Tuple of (fields, couplings) arrays
    """
    try:
        fields, couplings = get_fields_and_couplings_from_params(
            potts_parameters_path,
            device=device,
        )
        if fields is None or couplings is None:
            raise ValueError("Fields or couplings are None")
    except Exception as e:
        print(f"Error loading parameters as bmDCA format: {e}")
        print("Trying FraZ format...")
        couplings, fields = readParametersFraZ(potts_parameters_path, q=q)
        if fields is None or couplings is None:
            raise ValueError("Could not load parameters in either format")

    verify_symmetry(couplings)

    return fields, couplings


def energy_site_gibbs(sequence: torch.Tensor, site: int, 
                     couplings: torch.Tensor, fields: torch.Tensor, 
                     q: int = 21) -> torch.Tensor:
    """
    Compute energy contributions for a site using triangular coupling matrix.
    
    Args:
        sequence: (L,) tensor of site states
        site: Site index
        couplings: (L, L, q, q) triangular tensor (only i < j defined)
        fields: (L, q) tensor of field energies
        q: Number of states
    
    Returns:
        (q,) tensor of energy values for each state at site
    """
    sequence = ensure_tensor(sequence, dtype=torch.long)
    device = sequence.device
    L = len(sequence)
    
    # Start with field contribution
    ener = -fields[site].clone()  # (q,)
    
    # Process all other sites
    other_sites = torch.arange(L, device=device)
    other_sites = other_sites[other_sites != site]
    
    for j in other_sites:
        if site < j:
            ener -= couplings[site, j, :, sequence[j]]
        else:
            ener -= couplings[j, site, sequence[j], :]
    
    return ener


def energy(sequence: Union[list, np.ndarray, torch.Tensor], 
          couplings: Union[np.ndarray, torch.Tensor], 
          fields: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Compute energy of a single sequence using Francesco's method.
    
    Args:
        sequence: (L,) array of site states
        couplings: (L, L, q, q) array of coupling energies
        fields: (L, q) array of field energies
    
    Returns:
        Energy value as float
    """
    if isinstance(sequence, list):
        sequence = torch.tensor(sequence, dtype=torch.long)
    elif isinstance(sequence, np.ndarray):
        sequence = torch.from_numpy(sequence).long()
    
    # Use energy_of_sequence which handles conversion and computation

    return float(energy_of_sequence(sequence, fields, couplings).item())

def energy_site_MCMC(sequence: torch.Tensor, site: int, amino_acid: int,
                    couplings: torch.Tensor, fields: torch.Tensor) -> float:
    """
    Compute energy difference for mutation at a site (vectorized implementation).
    
    Args:
        sequence: (L,) tensor of site states
        site: Site being mutated
        amino_acid: New amino acid state
        couplings: (L, L, q, q) triangular tensor
        fields: (L, q) tensor of field energies
    
    Returns:
        Energy difference as float
    """
    sequence = ensure_tensor(sequence, dtype=torch.long)
    device = sequence.device
    L = len(sequence)
    
    # Field energy difference
    diff = -fields[site, amino_acid] + fields[site, sequence[site]]
    
    # Vectorized coupling differences
    # Create masks for sites before and after the mutation site
    sites_before = torch.arange(site, device=device)  # [0, 1, ..., site-1]
    sites_after = torch.arange(site + 1, L, device=device)  # [site+1, ..., L-1]
    
    if len(sites_before) > 0:
        # For sites j < site: couplings[j, site, sequence[j], :]
        states_before = sequence[sites_before]  # States at sites before
        coupling_diffs_before = (
            couplings[sites_before, site, states_before, amino_acid] - 
            couplings[sites_before, site, states_before, sequence[site]]
        )
        diff -= coupling_diffs_before.sum()
    
    if len(sites_after) > 0:
        # For sites j > site: couplings[site, j, :, sequence[j]]
        states_after = sequence[sites_after]  # States at sites after
        coupling_diffs_after = (
            couplings[site, sites_after, amino_acid, states_after] - 
            couplings[site, sites_after, sequence[site], states_after]
        )
        diff -= coupling_diffs_after.sum()
    
    return float(-diff.item())


def energy_site_MCMC_batch(sequences: torch.Tensor, sites: torch.Tensor, 
                          amino_acids: torch.Tensor, couplings: torch.Tensor, 
                          fields: torch.Tensor) -> torch.Tensor:
    """
    Compute energy differences for multiple mutations simultaneously.
    
    Args:
        sequences: (batch_size, L) tensor of sequences
        sites: (batch_size,) tensor of sites being mutated
        amino_acids: (batch_size,) tensor of new amino acid states
        couplings: (L, L, q, q) triangular tensor
        fields: (L, q) tensor of field energies
    
    Returns:
        (batch_size,) tensor of energy differences
    """
    batch_size, L = sequences.shape
    sequences = ensure_tensor(sequences, dtype=torch.long)
    device = sequences.device
    
    # Field energy differences
    batch_indices = torch.arange(batch_size, device=device)
    current_states = sequences[batch_indices, sites]  # Current amino acids at mutation sites
    
    field_diffs = -fields[sites, amino_acids] + fields[sites, current_states]
    
    # Initialize coupling differences
    coupling_diffs = torch.zeros(batch_size, device=device)
    
    # Vectorized coupling computation for all pairs
    for k in range(L):
        for j in range(k + 1, L):
            # Find mutations affecting this coupling pair
            site_k_mask = (sites == k)  # Mutations at site k
            site_j_mask = (sites == j)  # Mutations at site j
            
            if site_k_mask.any():
                # Mutations at site k affect coupling (k,j)
                affected_seqs = batch_indices[site_k_mask]
                old_states_k = sequences[affected_seqs, k]
                new_states_k = amino_acids[site_k_mask]
                states_j = sequences[affected_seqs, j]
                
                coupling_diff_k = (
                    couplings[k, j, new_states_k, states_j] - 
                    couplings[k, j, old_states_k, states_j]
                )
                coupling_diffs[affected_seqs] -= coupling_diff_k
            
            if site_j_mask.any():
                # Mutations at site j affect coupling (k,j)  
                affected_seqs = batch_indices[site_j_mask]
                states_k = sequences[affected_seqs, k]
                old_states_j = sequences[affected_seqs, j]
                new_states_j = amino_acids[site_j_mask]
                
                coupling_diff_j = (
                    couplings[k, j, states_k, new_states_j] - 
                    couplings[k, j, states_k, old_states_j]
                )
                coupling_diffs[affected_seqs] -= coupling_diff_j
    
    return -(field_diffs + coupling_diffs)


# PyTorch optimized versions of Francesco's functions
def energy_of_msa(msa: Union[torch.Tensor, np.ndarray], 
                  fields: Union[torch.Tensor, np.ndarray], 
                  couplings: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Francesco's energy function for MSA (optimized PyTorch implementation).
    """
    # Convert inputs to torch tensors if needed
    if isinstance(msa, np.ndarray):
        msa = torch.from_numpy(msa)
    if isinstance(fields, np.ndarray):
        fields = torch.from_numpy(fields)
    if isinstance(couplings, np.ndarray):
        couplings = torch.from_numpy(couplings)
    
    device = msa.device
    L, q = fields.shape
    
    # One-hot encode efficiently
    #inputs = torch.nn.functional.one_hot(msa, num_classes=q).float()  # (num_sequences, L, q)
    inputs = torch.nn.functional.one_hot(msa.long(), num_classes=q).float()

    
    # Symmetrize couplings correctly
    couplings_sym = 0.5 * (couplings + couplings.permute(1, 0, 3, 2))
    couplings_mask = 1 - torch.eye(L, device=device)
    couplings_mask = couplings_mask[:, :, None, None]  # (L,L,1,1)
    couplings_sym = couplings_sym * couplings_mask
    
    # Compute energies
    Jenergy = torch.einsum("nia,njb,ijab->n", inputs, inputs, couplings_sym)
    henergy = torch.einsum("nia,ia->n", inputs, fields)
    
    return -henergy - Jenergy / 2.0



def energy_of_sequence(sequence: Union[torch.Tensor, np.ndarray], 
                    fields: Union[torch.Tensor, np.ndarray], 
                    couplings: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Francesco's energy function for single sequence (optimized PyTorch implementation).
    
    Args:
        sequence: (L,) tensor/array of sequence states
        fields: (L, q) tensor/array of field energies
        couplings: (L, L, q, q) tensor/array of coupling energies
    
    Returns:
        (1,) tensor of energy (single value)
    """
    # Convert to tensors if needed
    if isinstance(sequence, np.ndarray):
        sequence = torch.from_numpy(sequence)
    if isinstance(fields, np.ndarray):
        fields = torch.from_numpy(fields)
    if isinstance(couplings, np.ndarray):
        couplings = torch.from_numpy(couplings)
    
    # Add batch dimension and use the MSA function
    sequence_batch = sequence.unsqueeze(0)  # (1, L)
    return energy_of_msa(sequence_batch, fields, couplings)


# Utility functions
def getEnergyProfile(MSA: Union[np.ndarray, torch.Tensor], 
                    couplings: Union[np.ndarray, torch.Tensor],
                    fields: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Get energy profile for an MSA using Francesco's method."""
    # Use energy_of_msa directly - it's already optimized for batch computation
    energies = energy_of_msa(MSA, fields, couplings)
    return energies.cpu().numpy()


def plotEnergyProfiles(MSAs: List[np.ndarray], names: List[str], 
                      ML_seq: np.ndarray, gibbs_seq: np.ndarray,
                      couplings: np.ndarray, fields: np.ndarray, 
                      save_path: str = ''):
    """Plot and compare energy profiles of MSAs."""
    colors = generate_colors(len(MSAs))
    energies = [getEnergyProfile(MSA, couplings, fields) for MSA in MSAs]
    
    # Determine global range
    all_data = np.concatenate(energies)
    data_min, data_max = np.min(all_data), np.max(all_data)
    bin_edges = np.linspace(data_min, data_max, 101)
    
    # Plot histograms
    for i, energy_profile in enumerate(energies):
        plt.hist(energy_profile, alpha=0.5, density=True, label=names[i],
                color=colors[i], histtype='barstacked', bins=bin_edges)
    
    # Add reference lines
    y_max = plt.gca().get_ylim()[1]
    ml_energy = energy(ML_seq, couplings, fields)
    gibbs_energy = energy(gibbs_seq, couplings, fields)
    
    plt.plot([ml_energy]*2, [0, y_max], "--", color=colors[0], 
             label="Energy of the ML seq")
    plt.plot([gibbs_energy]*2, [0, y_max], "--", color=colors[1], 
             label="Energy of the GT seq")
    
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plotEnergyProfilesGenerated(MSAs: List[np.ndarray], names: List[str],
                              root_seq: np.ndarray, couplings: np.ndarray,
                              fields: np.ndarray, title: str, save_path: str = ''):
    """Plot energy profiles for generated sequences."""
    colors = generate_colors(len(MSAs))
    energies = [getEnergyProfile(MSA, couplings, fields) for MSA in MSAs]
    
    # Determine global range
    all_data = np.concatenate(energies)
    data_min, data_max = np.min(all_data), np.max(all_data)
    bin_edges = np.linspace(data_min, data_max, 101)
    
    # Plot histograms
    for i, energy_profile in enumerate(energies):
        plt.hist(energy_profile, alpha=0.5, density=True, label=names[i],
                color=colors[i], histtype='barstacked', bins=bin_edges)
    
    # Add reference line
    y_max = plt.gca().get_ylim()[1]
    root_energy = energy(root_seq, couplings, fields)
    plt.plot([root_energy]*2, [0, y_max], "--", color=colors[0], 
             label="Energy of the root sequence")
    
    plt.xlabel('Energy from Potts model')
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def energy_site_gibbs(sequence: torch.Tensor, site: int, 
                     couplings: torch.Tensor, fields: torch.Tensor, 
                     q: int = 21, device = None) -> torch.Tensor:
    """
    Compute energy contributions for a site using triangular coupling matrix.
    
    Args:
        sequence: (L,) tensor of site states
        site: Site index
        couplings: (L, L, q, q) triangular tensor (only i < j defined)
        fields: (L, q) tensor of field energies
        q: Number of states
    
    Returns:
        (q,) tensor of energy values for each state at site
    """
    # Check available device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence = ensure_tensor(sequence, dtype=torch.long)
    fields = ensure_tensor(fields, dtype=torch.float32)
    couplings = ensure_tensor(couplings, dtype=torch.float32)
    
    sequence = sequence.to(device)
    fields = fields.to(device)
    couplings = couplings.to(device)
    #device = sequence.device
    L = len(sequence)
    
    # Start with field contribution
    ener = -fields[site].clone()  # (q,)
    
    # Process all other sites
    other_sites = torch.arange(L, device=device)
    other_sites = other_sites[other_sites != site]
    
    for j in other_sites:
        if site < j:
            ener -= couplings[site, j, :, sequence[j]]
        else:
            ener -= couplings[j, site, sequence[j], :]
    
    return ener

def mutate_gibbs_steps(sequence: np.array, nb_steps: int, fields_: torch.Tensor, couplings: torch.Tensor, q=21) -> np.array:
    """
    Samples sequences using Gibbs MCMC.
    Args:
        sequence: Initial sequence as a numpy array of shape (L,)
        nb_steps: Number of Gibbs sampling steps
        fields_: (L, q) tensor of field energies
        couplings: (L, L, q, q) tensor of coupling energies
        q: Number of states (default 21)
    Example usage:
        mutate_gibbs_steps(torch.tensor(seqs[0]), 10, torch.tensor(fields), torch.tensor(couplings), q=21)
    """
    L = len(sequence)
    seq = ensure_tensor(sequence, dtype=torch.long)
    
    for _ in range(nb_steps):
        ra = int(np.random.randint(L))
        seq_copy = seq.clone()
        
        ener_ = energy_site_gibbs(seq_copy.numpy(), ra, couplings, fields_)  # assuming energy_site_gibbs expects numpy array
        ener_ = ensure_tensor(ener_, dtype=torch.float32)

        # Softmax in a numerically stable way
        max_e = torch.max(-ener_)
        probs = torch.exp(-ener_ - max_e)  # subtract max to avoid overflow
        probs = probs / probs.sum()
        
        # Convert to numpy for np.random.choice
        probs_np = probs.cpu().numpy()
        probs_np /= probs_np.sum()  # make sure it sums exactly to 1
        
        selected_mutation = np.random.choice(q, p=probs_np)
        seq[ra] = selected_mutation

    return seq.cpu().numpy()

if __name__ == "__main__":
    # --- Load parameters and sequences ---
    fields, couplings = read_potts_parameters_proteins("DBD/Parameters_conv_Thr-PCD40.dat")
    seqs, names_to_ids, names = read_fasta2("DBD/DBD_alignment.uniref90.cov80.a2m")

    # --- Verify symmetry of couplings ---
    verify_symmetry(couplings)

    # --- Pick one sequence for testing ---
    seq = seqs[0]  # first sequence (numpy array of ints)
    print("Testing with sequence:", seq)

    # --- Test single sequence energy ---
    e_seq = energy(seq, couplings, fields)
    print("Energy of first sequence:", e_seq)

    # --- Test site energy contributions ---
    site = 3
    site_energy = energy_site_gibbs(
        torch.tensor(seq, dtype=torch.long),
        site,
        torch.tensor(couplings),
        torch.tensor(fields)
    )
    print(f"Site energy profile at site {site}: shape {site_energy.shape}")

    # --- Test batch MSA energy ---
    energies = getEnergyProfile(seqs[:10], couplings, fields)  # first 10 seqs
    print("Energies of first 10 sequences:", energies)

    # --- Test Gibbs mutation sampling ---
    mutated_seq = mutate_gibbs_steps(
        seq,
        nb_steps=20,
        fields_=torch.tensor(fields),
        couplings=torch.tensor(couplings),
        q=fields.shape[1]
    )
    print("Original sequence:", seq)
    print("Mutated sequence :", mutated_seq)

    # --- Test batch energy differences ---
    sequences = torch.tensor(np.vstack([seq, mutated_seq]), dtype=torch.long)
    sites = torch.tensor([1, 4])
    amino_acids = torch.tensor([2, 5])
    energy_diffs = energy_site_MCMC_batch(
        sequences,
        sites,
        amino_acids,
        torch.tensor(couplings),
        torch.tensor(fields)
    )
    print("Energy differences for batch mutations:", energy_diffs)
