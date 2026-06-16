"""
User-friendly helper functions for phyloDCA.
Simplifies complex operations into single-line calls.
"""

import os
import numpy as np
import torch
from pathlib import Path
from ete3 import Tree

def setup_paths(project_root, dataset_name):
    """
    Setup and validate all required paths.
    Returns dict with all paths ready to use.
    """
    paths = {
        'root': project_root,
        'data': f'{project_root}/data_{dataset_name}/',
        'msa': f'{project_root}/data_{dataset_name}/extant_msa/',
        'tree': f'{project_root}/data_{dataset_name}/tree_phylogeny/',
        'output_asr': f'{project_root}/output_{dataset_name}_asr/',
        'output_candidates': f'{project_root}/output_{dataset_name}_candidates/',
    }
    
    # Create output dirs
    for key in ['output_asr', 'output_candidates']:
        Path(paths[key]).mkdir(exist_ok=True)
    
    print("✓ Paths configured and output directories created")
    return paths


def load_data(msa_file, params_file):
    """
    Load MSA and DCA parameters with error checking.
    Returns: msa_array, headers, fields, couplings
    """
    from utils import read_fasta2, read_potts_parameters_proteins
    
    print(f"Loading MSA from: {msa_file}")
    if not os.path.exists(msa_file):
        raise FileNotFoundError(f"MSA file not found: {msa_file}")
    
    msa_array, _, headers = read_fasta2(str(msa_file))
    print(f"  ✓ Loaded {len(headers)} sequences of length {msa_array.shape[1]}")
    
    print(f"Loading DCA parameters from: {params_file}")
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Parameters file not found: {params_file}")
    
    fields, couplings = read_potts_parameters_proteins(params_file)
    print(f"  ✓ Loaded fields shape {fields.shape}, couplings shape {couplings.shape}")
    
    return msa_array, headers, fields, couplings

def run_asr_pipeline(msa_file, tree_file, output_dir):
    """
    Complete ASR workflow in one call.
    Returns: root_posteriors, sequence_name
    """
    from utils.inferringMu import fit_mu
    from utils.asr_torch_perfect import (
        felsenstein_root_posteriors_optimized,
        fasta_to_tensor_dict,
        newick_to_pytorch_tree
    )
    
    # Extract sequence name
    seq_name = Path(msa_file).stem
    
    print(f"\n{'='*60}")
    print(f"RUNNING ASR FOR: {seq_name}")
    print(f"{'='*60}")
    
    import time
    start = time.time()
    
    # Step 1: Infer mu
    print("\n1. Inferring mutation rate (mu)...")
    try:
        mu = fit_mu(Tree(tree_file), msa_file)
        print(f"   ✓ mu = {mu:.5f}")
    except Exception as e:
        print(f"   ⚠ Warning: Could not infer mu: {e}")
        mu = 0.01  # fallback
        print(f"   Using default mu = {mu}")
    
    # Step 2: Load sequences and tree
    print("\n2. Loading sequences and tree...")
    leaf_seq_by_name = fasta_to_tensor_dict(msa_file)
    tree_torch = newick_to_pytorch_tree(tree_file)
    print(f"   ✓ Loaded {len(leaf_seq_by_name)} leaf sequences")
    
    # Step 3: Compute stationary distribution
    print("\n3. Computing site-wise amino acid frequencies...")
    seqs_array = np.stack([leaf_seq_by_name[name].numpy() for name in leaf_seq_by_name])
    L = seqs_array.shape[1]
    q = 21
    w = torch.zeros((L, q), dtype=torch.float64)
    for s in range(L):
        counts = np.bincount(seqs_array[:, s], minlength=q)
        w[s] = torch.tensor(counts / counts.sum(), dtype=torch.float64)
    print(f"   ✓ Computed frequencies for {L} sites")
    
    # Step 4: Compute root posteriors
    print("\n4. Computing ancestral state posteriors...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    root_posteriors = felsenstein_root_posteriors_optimized(
        tree_torch, leaf_seq_by_name, w, mu,
        device=device,
        batch_size=64,
        use_jit=True
    )
    print(f"   ✓ Posteriors computed: shape {root_posteriors.shape}")
    
    # Step 5: Handle any NaN values
    posteriors_np = root_posteriors.cpu().numpy()
    nan_count = np.sum(np.isnan(posteriors_np))
    if nan_count > 0:
        print(f"\n   ⚠ Found {nan_count} NaN values, fixing...")
        posteriors_np[np.isnan(posteriors_np)] = 1.0 / q
        posteriors_np = posteriors_np / posteriors_np.sum(axis=1, keepdims=True)
        root_posteriors = torch.from_numpy(posteriors_np)
        print(f"   ✓ Fixed NaN values")
    
    # Save results
    Path(output_dir).mkdir(exist_ok=True)
    save_file = f"{output_dir}{seq_name}_posteriors.npy"
    np.save(save_file, posteriors_np)
    print(f"\n✓ Saved posteriors to: {save_file}")
    
    elapsed = time.time() - start
    print(f"✓ ASR completed in {elapsed:.1f} seconds\n")
    
    return posteriors_np, seq_name


def sample_and_rank_sequences(posteriors, num_samples=1000, num_top=10, 
                              fields=None, couplings=None, temperature=0.2):
    """
    Sample sequences from posterior with optional MCMC reshuffling.
    
    Args:
        posteriors: ASR posterior probabilities [L, q]
        num_samples: Number of sequences to sample
        num_top: Number of top candidates to return per source
        fields: DCA fields (optional, for reshuffling)
        couplings: DCA couplings (optional, for reshuffling)
        temperature: MCMC temperature for reshuffling (0-1, default 0.2)
    
    Returns: list of top candidates, reshuffled MSA, original MSA
    """
    from utils import create_MSA_profile
    
    print(f"Sampling {num_samples} sequences from posterior...")
    msa_original = create_MSA_profile(posteriors, cardinal=num_samples)
    print(f"   ✓ Generated MSA shape: {msa_original.shape}")
    
    # Step 2: Optional MCMC reshuffling
    msa_reshuffled = msa_original.copy()
    
    if fields is not None and couplings is not None:
        print(f"\nApplying MCMC reshuffling (T={temperature})...")
        try:
            # Import and patch entropy function
            try:
                from utils.ci_and_cd_entropy import context_independent_entropy
            except ImportError:
                from utils.MCMC_reshuffling_torch_perfect import context_independent_entropy
            
            from utils.MCMC_reshuffling_torch_perfect import MCMC_columns_pytorch_efficient
            
            # Patch into function globals
            MCMC_columns_pytorch_efficient.__globals__['context_independent_entropy'] = context_independent_entropy
            
            msa_reshuffled = MCMC_columns_pytorch_efficient(
                MSA=msa_original,
                couplings=couplings,
                fields_=fields,
                T=temperature,
                scale=1.0,
                device="cuda" if torch.cuda.is_available() else "cpu",
                q=21,
                verbose=False
            )
            print(f"   ✓ MCMC reshuffling completed")
        except Exception as e:
            print(f"   ⚠ MCMC reshuffling failed: {e}")
            print(f"   Using original sampled MSA instead")
            msa_reshuffled = msa_original
    else:
        print("\nSkipping MCMC reshuffling (fields/couplings not provided)")
    
    # Step 3: Rank sequences by posterior likelihood
    print(f"\nRanking sequences by posterior likelihood...")
    
    # Score original sampled sequences
    scores_original = []
    for seq in msa_original:
        score = sum(np.log(posteriors[i, int(seq[i])] + 1e-10) 
                   for i in range(len(seq)))
        scores_original.append(score)
    
    # Score reshuffled sequences
    scores_reshuffled = []
    for seq in msa_reshuffled:
        score = sum(np.log(posteriors[i, int(seq[i])] + 1e-10) 
                   for i in range(len(seq)))
        scores_reshuffled.append(score)
    
    scores_original = np.array(scores_original)
    scores_reshuffled = np.array(scores_reshuffled)
    
    # Get top num_top from each set
    top_orig_idx = np.argsort(scores_original)[-num_top:][::-1]
    top_resh_idx = np.argsort(scores_reshuffled)[-num_top:][::-1]
    
    # Combine and sort by score
    candidates = []
    
    # Add from original
    for rank, idx in enumerate(top_orig_idx):
        candidates.append({
            'sequence': msa_original[idx],
            'score': scores_original[idx],
            'source': 'ASR_posterior',
            'rank': rank + 1
        })
    
    # Add from reshuffled
    for rank, idx in enumerate(top_resh_idx):
        candidates.append({
            'sequence': msa_reshuffled[idx],
            'score': scores_reshuffled[idx],
            'source': 'MCMC_reshuffled',
            'rank': rank + 1
        })
    
    # Sort all by score
    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:num_top]
    
    print(f"✓ Top {num_top} candidates ranked from both sets\n")
    return candidates, msa_reshuffled, msa_original


def save_candidates_to_fasta(candidates, posteriors, msa_original, msa_reshuffled, output_file):
    """
    Save MAP, consensus, and top 10 from both original and reshuffled sets to FASTA.
    Total: 1 MAP + 1 consensus + 10 ASR + 10 MCMC = 22 sequences
    """
    aa_map = {
        0: '-', 1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
        10: 'L', 11: 'M', 12: 'N', 13: 'P', 14: 'Q', 15: 'R', 16: 'S', 17: 'T', 18: 'V', 19: 'W', 20: 'Y'
    }
    
    def int_to_str(seq_int):
        return ''.join([aa_map.get(int(aa), 'X') for aa in seq_int])
    
    fasta_entries = []
    
    # 1. MAP sequence (maximum a posteriori)
    map_seq = np.argmax(posteriors, axis=1)
    fasta_entries.append(('>1_MAP_sequence', int_to_str(map_seq)))
    
    # 2. Consensus sequence (most common state at each position)
    consensus_seq = np.argmax(posteriors, axis=1)
    fasta_entries.append(('>2_consensus_sequence', int_to_str(consensus_seq)))
    
    # 3. Score all sequences and get top 10 from each source
    scores_original = []
    for seq in msa_original:
        score = sum(np.log(posteriors[i, int(seq[i])] + 1e-10) 
                   for i in range(len(seq)))
        scores_original.append(score)
    
    scores_reshuffled = []
    for seq in msa_reshuffled:
        score = sum(np.log(posteriors[i, int(seq[i])] + 1e-10) 
                   for i in range(len(seq)))
        scores_reshuffled.append(score)
    
    scores_original = np.array(scores_original)
    scores_reshuffled = np.array(scores_reshuffled)
    
    # Get top 10 indices from each
    top_orig_idx = np.argsort(scores_original)[-10:][::-1]
    top_resh_idx = np.argsort(scores_reshuffled)[-10:][::-1]
    
    print(f"\nSaving candidates:")
    print(f"  - 1 MAP sequence")
    print(f"  - 1 consensus sequence")
    print(f"  - 10 top from ASR posterior")
    print(f"  - 10 top from MCMC reshuffled")
    print(f"  = 22 total sequences")
    
    # Add top 10 from ASR posterior
    for rank, idx in enumerate(top_orig_idx):
        header = f">3_ASR_posterior_rank{rank+1:02d}_score{scores_original[idx]:.2f}"
        fasta_entries.append((header, int_to_str(msa_original[idx])))
    
    # Add top 10 from MCMC reshuffled
    for rank, idx in enumerate(top_resh_idx):
        header = f">4_MCMC_reshuffled_rank{rank+1:02d}_score{scores_reshuffled[idx]:.2f}"
        fasta_entries.append((header, int_to_str(msa_reshuffled[idx])))
    
    # Write file
    Path(output_file).parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        for header, seq in fasta_entries:
            f.write(f"{header}\n{seq}\n")
    
    print(f"\n✓ Saved {len(fasta_entries)} sequences to: {output_file}\n")
    return output_file
