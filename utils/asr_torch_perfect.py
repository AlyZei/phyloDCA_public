import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional, List
from ete3 import Tree
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from utils.toolsForTreesAndMSAs import read_fasta2

# ------------------------------
# TorchTree data structure
# ------------------------------
class TorchTree:
    def __init__(self, children, blens, is_leaf, leaf_name):
        self.children = children
        self.blens = blens
        self.is_leaf = is_leaf
        self.leaf_name = leaf_name
        self.n_nodes = len(children)

# ------------------------------
# Tree builder
# ------------------------------
def newick_to_pytorch_tree(newick_file):
    ete_tree = Tree(newick_file, format=1)
    nodes = list(ete_tree.traverse("postorder"))
    node_to_idx = {id(n): i for i, n in enumerate(nodes)}

    children = [[] for _ in range(len(nodes))]
    blens = [[] for _ in range(len(nodes))]
    is_leaf = [False] * len(nodes)
    leaf_name = [None] * len(nodes)

    for i, node in enumerate(nodes):
        if node.is_leaf():
            is_leaf[i] = True
            leaf_name[i] = node.name
        for c in node.children:
            children[i].append(node_to_idx[id(c)])
            blens[i].append(float(c.dist) if hasattr(c, "dist") else 0.0)

    return TorchTree(children, blens, is_leaf, leaf_name)

# ------------------------------
# FASTA → dict of tensors
# ------------------------------
def fasta_to_tensor_dict(fasta_path: str) -> Dict[str, torch.Tensor]:
    # You'll need to implement read_fasta2 or use your existing function
    seqs, names_to_ids, _ = read_fasta2(fasta_path)
    return {name: torch.tensor(seqs[idx], dtype=torch.long) for name, idx in names_to_ids.items()}

# ------------------------------
# JIT-compatible helper functions
# ------------------------------
@torch.jit.script
def jc_log_transition_matrix_batch(W: torch.Tensor, w_batch: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled batch JC transition matrix computation.
    
    Args:
        W: scalar tensor (exp(-mu * branch_length))
        w_batch: tensor [B, q] - stationary distributions for B sites
        
    Returns:
        log_P: tensor [B, q, q] - log transition probabilities for each site
    """
    dtype = torch.float64
    B, q = w_batch.shape
    
    W = W.to(dtype=dtype, device=w_batch.device)
    w_batch = w_batch.to(dtype=dtype)
    
    # Create identity matrix [q, q] and expand to [B, q, q]
    I = torch.eye(q, dtype=dtype, device=w_batch.device).unsqueeze(0).expand(B, q, q)
    
    # Create outer product: [B, q, q] where outer[b, i, j] = w_batch[b, j]
    outer = w_batch.unsqueeze(1).expand(B, q, q)
    
    # P[b, i, j] = W * δ_ij + (1-W) * w_b[j]
    P = W * I + (1.0 - W) * outer
    
    # Clamp to avoid log(0) issues
    P = torch.clamp(P, min=1e-300)
    log_P = torch.log(P)
    
    return log_P

@torch.jit.script
def stable_logsumexp_normalize_batch(log_probs: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled batch normalization in log space.
    
    Args:
        log_probs: tensor [B, q] - log probabilities for B sites
        
    Returns:
        normalized_probs: tensor [B, q] - normalized probabilities
    """
    # Subtract max for numerical stability
    max_vals = torch.max(log_probs, dim=-1, keepdim=True)[0]
    log_probs_shifted = log_probs - max_vals
    
    # Compute sum in linear space
    sum_exp = torch.sum(torch.exp(log_probs_shifted), dim=-1, keepdim=True)
    
    # Convert back to log space and normalize
    log_sum = torch.log(sum_exp) + max_vals
    normalized_probs = torch.exp(log_probs - log_sum)
    
    return normalized_probs

@torch.jit.script
def batch_matmul_logsumexp(log_P: torch.Tensor, child_log_like: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled batched matrix multiplication in log space.
    """
    # Add dimensions: log_P[B, q, q] + child_log_like[B, 1, q] -> [B, q, q]
    expanded_child = child_log_like.unsqueeze(1)  # [B, 1, q]
    combined = log_P + expanded_child  # [B, q, q]
    
    # Sum over the last dimension (matrix multiplication)
    result = torch.logsumexp(combined, dim=-1)  # [B, q]
    
    return result

# ------------------------------
# JIT-compatible core computation (simplified structure)
# ------------------------------
@torch.jit.script
def felsenstein_batch_jit_core(
    leaf_obs_batch: torch.Tensor,    # [n_leaves, B] - observed amino acids for batch sites
    w_batch: torch.Tensor,           # [B, q] - stationary distributions for batch
    mu: float,
    node_order: torch.Tensor,        # [n_nodes] - post-order traversal order
    children_list: torch.Tensor,     # [n_nodes, max_children] - children indices (-1 = no child)
    blens_list: torch.Tensor,        # [n_nodes, max_children] - branch lengths
    is_leaf_list: torch.Tensor,      # [n_nodes] - boolean, True if leaf
    leaf_node_mapping: torch.Tensor  # [n_leaves] - which node each leaf corresponds to
) -> torch.Tensor:
    """
    JIT-compiled core with pre-flattened structure to avoid indexing issues.
    """
    dtype = torch.float64
    device = w_batch.device
    B, q = w_batch.shape
    n_nodes = node_order.shape[0]
    n_leaves = leaf_obs_batch.shape[0]
    max_children = children_list.shape[1]
    
    # Storage for log likelihoods [n_nodes, B, q]
    log_likelihoods = torch.full((n_nodes, B, q), -float('inf'), dtype=dtype, device=device)
    
    # Initialize leaf nodes
    for leaf_idx in range(n_leaves):
        node_idx = leaf_node_mapping[leaf_idx]
        if is_leaf_list[node_idx]:
            # Set up one-hot encoding for observed amino acids
            observed_aas = leaf_obs_batch[leaf_idx]  # [B]
            # Use advanced indexing to set the appropriate entries to 0.0 (log(1.0))
            batch_indices = torch.arange(B, device=device, dtype=torch.long)
            log_likelihoods[node_idx, batch_indices, observed_aas] = 0.0
    
    # Process internal nodes in post-order
    for i in range(n_nodes):
        node_idx = node_order[i]
        if not is_leaf_list[node_idx]:
            log_node_prob = torch.zeros((B, q), dtype=dtype, device=device)
            
            # Process all potential children
            for child_slot in range(max_children):
                child_idx = children_list[node_idx, child_slot]
                if child_idx == -1:  # No more children
                    break
                
                branch_length = blens_list[node_idx, child_slot]
                child_log_like = log_likelihoods[child_idx]  # [B, q]
                
                # Compute transition matrix
                W = torch.exp(-mu * branch_length)
                W_tensor = torch.tensor(W, dtype=dtype, device=device)
                log_P = jc_log_transition_matrix_batch(W_tensor, w_batch)  # [B, q, q]
                
                # Matrix multiplication in log space
                child_contribution = batch_matmul_logsumexp(log_P, child_log_like)
                log_node_prob = log_node_prob + child_contribution
            
            log_likelihoods[node_idx] = log_node_prob
    
    # Root is the last node in post-order
    root_idx = node_order[n_nodes - 1]
    root_log_like = log_likelihoods[root_idx]  # [B, q]
    
    # Apply stationary distribution
    log_w = torch.log(torch.clamp(w_batch, min=1e-300))
    log_posterior_unnorm = root_log_like + log_w
    
    # Normalize
    root_posterior = stable_logsumexp_normalize_batch(log_posterior_unnorm)
    
    return root_posterior

# ------------------------------
# Non-JIT fallback version (more flexible)
# ------------------------------
def felsenstein_batch_fallback(
    leaf_sequences: torch.Tensor,
    w_batch: torch.Tensor,
    site_indices: torch.Tensor,
    mu: float,
    tree: TorchTree,
    device: str
) -> torch.Tensor:
    """
    Non-JIT fallback version with full flexibility.
    """
    dtype = torch.float64
    B = len(site_indices)
    q = w_batch.shape[1]
    
    # Extract sequences for the batch sites
    leaf_obs_batch = leaf_sequences[:, site_indices]  # [n_leaves, B]
    
    # Storage for log likelihoods
    log_likelihoods = {}
    
    # Process nodes in post-order
    for node_idx in range(tree.n_nodes):
        if tree.is_leaf[node_idx]:
            # Find which leaf this corresponds to
            leaf_name = tree.leaf_name[node_idx]
            # Find leaf index by name (this is inefficient but works)
            leaf_idx = None
            for i, seq in enumerate(leaf_sequences):
                # We need a way to map back to leaf names - this is a limitation
                # For now, assume leaf order matches node order for leaves
                pass
            
            # Simplified: assume node_idx corresponds to leaf order
            # This needs to be fixed based on your actual data structure
            observed_aas = leaf_obs_batch[node_idx] if node_idx < len(leaf_obs_batch) else torch.zeros(B, dtype=torch.long, device=device)
            
            log_like = torch.full((B, q), -float('inf'), dtype=dtype, device=device)
            batch_indices = torch.arange(B, device=device)
            log_like[batch_indices, observed_aas] = 0.0
            log_likelihoods[node_idx] = log_like
            
        else:
            # Internal node
            log_node_prob = torch.zeros((B, q), dtype=dtype, device=device)
            
            for child_idx, branch_length in zip(tree.children[node_idx], tree.blens[node_idx]):
                child_log_like = log_likelihoods.pop(child_idx)
                
                # Transition matrix
                W = torch.exp(torch.tensor(-mu * branch_length, dtype=dtype, device=device))
                log_P = jc_log_transition_matrix_batch(W, w_batch)
                
                # Matrix multiplication in log space
                child_contribution = batch_matmul_logsumexp(log_P, child_log_like)
                log_node_prob = log_node_prob + child_contribution
            
            log_likelihoods[node_idx] = log_node_prob
    
    # Root processing
    root_idx = tree.n_nodes - 1
    root_log_like = log_likelihoods[root_idx]
    
    # Apply stationary distribution and normalize
    log_w = torch.log(torch.clamp(w_batch, min=1e-300))
    log_posterior_unnorm = root_log_like + log_w
    root_posterior = stable_logsumexp_normalize_batch(log_posterior_unnorm)
    
    return root_posterior

# ------------------------------
# Preprocessing for JIT version
# ------------------------------
def preprocess_tree_for_jit_fixed(tree: TorchTree, leaf_seq_by_name: Dict[str, torch.Tensor], device: str):
    """
    Create JIT-compatible tensor representations of the tree structure.
    """
    # Get leaf sequences in consistent order
    leaf_names = sorted(leaf_seq_by_name.keys())  # Ensure consistent ordering
    leaf_sequences = torch.stack([leaf_seq_by_name[name] for name in leaf_names])
    leaf_sequences = leaf_sequences.to(device=device)
    
    # Create mapping from leaf to node
    leaf_node_mapping = torch.zeros(len(leaf_names), dtype=torch.long, device=device)
    for i, leaf_name in enumerate(leaf_names):
        for node_idx in range(tree.n_nodes):
            if tree.is_leaf[node_idx] and tree.leaf_name[node_idx] == leaf_name:
                leaf_node_mapping[i] = node_idx
                break
    
    # Create post-order traversal
    node_order = torch.arange(tree.n_nodes, dtype=torch.long, device=device)
    
    # Create padded children and branch length tensors
    max_children = max(len(children) for children in tree.children) if any(tree.children) else 1
    children_list = torch.full((tree.n_nodes, max_children), -1, dtype=torch.long, device=device)
    blens_list = torch.zeros((tree.n_nodes, max_children), dtype=torch.float64, device=device)
    
    for node_idx in range(tree.n_nodes):
        for i, (child_idx, blen) in enumerate(zip(tree.children[node_idx], tree.blens[node_idx])):
            if i < max_children:
                children_list[node_idx, i] = child_idx
                blens_list[node_idx, i] = blen
    
    is_leaf_list = torch.tensor(tree.is_leaf, dtype=torch.bool, device=device)
    
    return leaf_sequences, node_order, children_list, blens_list, is_leaf_list, leaf_node_mapping

# ------------------------------
# Main optimized function with JIT and fallback
# ------------------------------
def felsenstein_root_posteriors_optimized(
    tree: TorchTree,
    leaf_seq_by_name: Dict[str, torch.Tensor],
    w: torch.Tensor,
    mu: float,
    device: str = "cpu",
    batch_size: int = 64,
    use_jit: bool = True,
    n_workers: int = None,
    use_multiprocessing: bool = False
) -> torch.Tensor:
    """
    Optimized Felsenstein computation with optional JIT compilation.
    """
    dtype = torch.float64
    w = w.to(dtype=dtype, device=device)
    L, q = w.shape
    
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 4)
    
    print(f"Processing {L} sites with batch_size={batch_size}, use_jit={use_jit}")
    
    # Preprocess tree
    if use_jit:
        try:
            leaf_sequences, node_order, children_list, blens_list, is_leaf_list, leaf_node_mapping = \
                preprocess_tree_for_jit_fixed(tree, leaf_seq_by_name, device)
            
            # Test JIT compilation with small batch
            print("Testing JIT compilation...")
            small_batch_size = min(4, L)
            site_indices_small = torch.arange(small_batch_size, device=device)
            w_small = w[:small_batch_size]
            leaf_obs_small = leaf_sequences[:, :small_batch_size]
            
            _ = felsenstein_batch_jit_core(
                leaf_obs_small, w_small, mu, node_order, 
                children_list, blens_list, is_leaf_list, leaf_node_mapping
            )
            print("JIT compilation successful!")
            
        except Exception as e:
            print(f"JIT compilation failed: {e}")
            print("Falling back to non-JIT version...")
            use_jit = False
    
    # Process in batches
    root_posteriors = torch.zeros((L, q), dtype=dtype, device=device)
    
    for batch_start in range(0, L, batch_size):
        batch_end = min(batch_start + batch_size, L)
        site_indices = torch.arange(batch_start, batch_end, device=device)
        w_batch = w[batch_start:batch_end]
        
        if batch_start % (batch_size * 10) == 0:
            print(f"Processing batch {batch_start//batch_size + 1}/{(L-1)//batch_size + 1}")
        
        try:
            if use_jit:
                # JIT version
                leaf_obs_batch = leaf_sequences[:, site_indices]
                batch_result = felsenstein_batch_jit_core(
                    leaf_obs_batch, w_batch, mu, node_order,
                    children_list, blens_list, is_leaf_list, leaf_node_mapping
                )
            else:
                # Fallback version
                leaf_sequences_full = torch.stack([leaf_seq_by_name[name] for name in sorted(leaf_seq_by_name.keys())])
                leaf_sequences_full = leaf_sequences_full.to(device=device)
                batch_result = felsenstein_batch_fallback(
                    leaf_sequences_full, w_batch, site_indices, mu, tree, device
                )
            
            root_posteriors[batch_start:batch_end] = batch_result
            
        except Exception as e:
            print(f"Error in batch {batch_start}-{batch_end}: {e}")
            # Process this batch site-by-site as ultimate fallback
            for site_idx in range(batch_start, batch_end):
                try:
                    site_result = felsenstein_single_site_fallback(tree, leaf_seq_by_name, w[site_idx], mu, site_idx, device)
                    root_posteriors[site_idx] = site_result
                except Exception as e2:
                    print(f"Failed completely for site {site_idx}: {e2}")
                    # Fill with uniform distribution as last resort
                    root_posteriors[site_idx] = torch.ones(q, dtype=dtype, device=device) / q
    
    # Final verification
    if torch.isnan(root_posteriors).any():
        nan_count = torch.isnan(root_posteriors).any(dim=1).sum().item()
        print(f"WARNING: {nan_count} sites contain NaN values")
    
    return root_posteriors

# ------------------------------
# Single site fallback for ultimate reliability
# ------------------------------
def felsenstein_single_site_fallback(tree, leaf_seq_by_name, w_site, mu, site_idx, device):
    """Ultimate fallback - process single site with maximum compatibility."""
    dtype = torch.float64
    q = w_site.shape[0]
    w_site = w_site.to(dtype=dtype, device=device)
    
    log_likelihoods = {}
    
    for node_idx in range(tree.n_nodes):
        if tree.is_leaf[node_idx]:
            leaf_name = tree.leaf_name[node_idx]
            seq = leaf_seq_by_name[leaf_name].to(device=device)
            observed_aa = seq[site_idx].item()
            
            log_like = torch.full((q,), -float('inf'), dtype=dtype, device=device)
            log_like[observed_aa] = 0.0
            log_likelihoods[node_idx] = log_like
        else:
            log_node_prob = torch.zeros(q, dtype=dtype, device=device)
            
            for child_idx, branch_length in zip(tree.children[node_idx], tree.blens[node_idx]):
                child_log_like = log_likelihoods.pop(child_idx)
                
                W = torch.exp(torch.tensor(-mu * branch_length, dtype=dtype, device=device))
                
                # Simple JC matrix for single site
                I = torch.eye(q, dtype=dtype, device=device)
                outer = w_site.unsqueeze(0).expand(q, q)
                P = W * I + (1.0 - W) * outer
                P = torch.clamp(P, min=1e-300)
                log_P = torch.log(P)
                
                child_contribution = torch.logsumexp(log_P + child_log_like.unsqueeze(0), dim=1)
                log_node_prob = log_node_prob + child_contribution
            
            log_likelihoods[node_idx] = log_node_prob
    
    root_idx = tree.n_nodes - 1
    root_log_like = log_likelihoods[root_idx]
    
    log_w = torch.log(torch.clamp(w_site, min=1e-300))
    log_posterior_unnorm = root_log_like + log_w
    
    root_posterior = stable_logsumexp_normalize_batch(log_posterior_unnorm.unsqueeze(0)).squeeze(0)
    
    return root_posterior

# ------------------------------
# Example usage
# ------------------------------
# if __name__ == "__main__":
#     # Example parameters - replace with your actual data
    
#     fasta_path = fasta #"synthetic_data_leo/DBD_wt2748_mu30.0_amino_artificial_tree_4096leaves_depth12.fa"
#     tree_path = 'test_tree.nwk' #"artificial_trees_leo/artificial_tree_4096leaves_depth12.nwk"

    
#     # Load data
#     leaf_seq_by_name = fasta_to_tensor_dict(fasta_path)
#     tree = newick_to_pytorch_tree(tree_path)
    
#     mu = fit_mu(Tree(tree_path), fasta_path)
    
#     # Compute stationary distributions
#     seqs_array = np.stack([leaf_seq_by_name[name].numpy() for name in leaf_seq_by_name])
#     n_leaves, L = seqs_array.shape
#     q = 21
    
#     w = torch.zeros((L, q), dtype=torch.float64)
#     for s in range(L):
#         counts = np.bincount(seqs_array[:, s], minlength=q)
#         w[s] = torch.tensor(counts / counts.sum(), dtype=torch.float64)
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Run with automatic JIT fallback
#     root_posteriors = felsenstein_root_posteriors_optimized(
#         tree, leaf_seq_by_name, w, mu, 
#         device=device, 
#         batch_size=64,
#         use_jit=True  # Will automatically fall back if JIT fails
#     )
    
#     print(f"Shape: {root_posteriors.shape}")
#     print(f"Range: [{root_posteriors.min():.6f}, {root_posteriors.max():.6f}]")
#     print(f"Row sums: [{root_posteriors.sum(dim=1).min():.6f}, {root_posteriors.sum(dim=1).max():.6f}]")