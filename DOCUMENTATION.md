
# phyloDCA Utilities Documentation

## Overview
phyloDCA is a toolkit for ancestral sequence recovery using phylogenetic ancestral state reconstruction (ASR) and Direct Coupling Analysis (DCA) models.

---

## Table of Contents
1. [User Helper Functions](#user-helper-functions)
2. [Data Loading](#data-loading)
3. [ASR Functions](#asr-functions)
4. [MCMC Reshuffling](#mcmc-reshuffling) 
5. [Troubleshooting](#troubleshooting)

---

## User Helper Functions

### `setup_paths(project_root, dataset_name)`

**Purpose:** Setup and validate all required project paths.

**Parameters:**
- `project_root` (str): Root directory of phyloDCA project
- `dataset_name` (str): Name of dataset (e.g., 'betaLac')

**Returns:**
- `paths` (dict): Dictionary containing all utility paths

**Example:**
```python
from utils.user_helper_functions import setup_paths
paths = setup_paths('/home/user/phyloDCA_public/', 'betaLac')
print(paths['output_asr'])  # Output: /home/user/phyloDCA_public/output_betaLac_asr/
```

**Output Directories Created:**
- `output_{dataset}_asr/` - ASR results
- `output_{dataset}_candidates/` - Candidate sequences

---

### `load_data(msa_file, params_file)`

**Purpose:** Load MSA and DCA parameters with error checking.

**Parameters:**
- `msa_file` (str): Path to FASTA format MSA file
- `params_file` (str): Path to DCA parameters file

**Returns:**
- `msa_array` (np.ndarray): MSA as integer array [N_sequences, L_sites]
- `headers` (list): Sequence identifiers
- `fields` (np.ndarray): DCA field parameters
- `couplings` (np.ndarray): DCA coupling parameters

**Raises:**
- `FileNotFoundError`: If MSA or parameters file not found

**Example:**
```python
from utils.user_helper_functions import load_data
msa, headers, fields, couplings = load_data(
    'data_betaLac/extant_msa/sequences.faa',
    'data_betaLac/Parameters.dat'
)
print(f"Loaded {len(headers)} sequences, length {msa.shape[1]}")
```
---


## Data Loading

### Module: `utils` (from existing phyloDCA)

#### `read_fasta2(filename)`

**Purpose:** Read FASTA file into numpy array and metadata.

**Returns:**
- `msa_array` (np.ndarray): Integer-encoded sequences [N, L]
- `msa_dict`: Dictionary version
- `headers` (list): Sequence names

---

#### `read_potts_parameters_proteins(filename)`

**Purpose:** Load DCA parameters in Potts model format.

**Returns:**
- `fields` (np.ndarray): Field parameters [L, 21]
- `couplings` (np.ndarray): Coupling parameters [L, L, 21, 21]

---

#### `create_MSA_profile(posteriors, cardinal)`

**Purpose:** Sample sequences from posterior distribution.

**Parameters:**
- `posteriors` (np.ndarray): Posterior probabilities [L, 21]
- `cardinal` (int): Number of sequences to sample

**Returns:**
- `MSA` (np.ndarray): Sampled sequences [cardinal, L]

---

### `run_asr_pipeline(msa_file, tree_file, output_dir)`

**Purpose:** Complete ancestral state reconstruction workflow in one call.

**Parameters:**
- `msa_file` (str): Path to cleaned MSA (FASTA format)
- `tree_file` (str): Path to phylogenetic tree (Newick format)
- `output_dir` (str): Directory to save posteriors

**Returns:**
- `posteriors_np` (np.ndarray): Root ancestral state probabilities [L_sites, 21_states]
- `seq_name` (str): Extracted sequence name from filename

**Workflow Steps (automatic):**
1. Infer mutation rate (mu) using Felsenstein algorithm
2. Load leaf sequences and phylogenetic tree
3. Compute site-wise amino acid frequencies
4. Compute root posterior probabilities using Felsenstein dynamic programming
5. Handle and fix any NaN values
6. Save posteriors to `.npy` file

**Example:**
```python
from utils.user_helper_functions import run_asr_pipeline
posteriors, seq_name = run_asr_pipeline(
    msa_file='cleaned_data_and_inferred_tree/alignment/betaLac_collapsed.fasta',
    tree_file='cleaned_data_and_inferred_tree/tree/betaLactree_collapsed_noonlychild_midpointrooted.nwk',
    output_dir='output_betaLac_asr/'
)
print(f"Posteriors shape: {posteriors.shape}")  # (202, 21) for 202 sites, 21 amino acid states
```

**Output Files:**
- `{output_dir}{seq_name}_posteriors.npy` - Numpy array of posteriors

---

### `sample_and_rank_sequences(posteriors, num_samples=1000, num_top=10, fields=None, couplings=None, temperature=0.2)`

**Purpose:** Sample sequences from posterior distribution with optional MCMC reshuffling and ranking.

**Parameters:**
- `posteriors` (np.ndarray): ASR posteriors [L, 21]
- `num_samples` (int): Number of sequences to sample (default: 1000)
- `num_top` (int): Number of top candidates per source (default: 10)
- `fields` (np.ndarray, optional): DCA fields for MCMC reshuffling
- `couplings` (np.ndarray, optional): DCA couplings for MCMC reshuffling
- `temperature` (float): MCMC temperature 0-1 (default: 0.2)
  - Lower = closer to Maximum A Posteriori (MAP)
  - Higher = more explore

**Returns:**
- `candidates` (list): List of dicts with keys: 'sequence', 'score', 'source', 'rank'
- `msa_reshuffled` (np.ndarray): MCMC reshuffled MSA [num_samples, L]
- `msa_original` (np.ndarray): Original sampled MSA [num_samples, L]

**Example:**
```python
from utils.user_helper_functions import sample_and_rank_sequences
candidates, msa_resh, msa_orig = sample_and_rank_sequences(
    posteriors=posteriors,
    num_samples=1000,
    num_top=10,
    fields=fields,
    couplings=couplings,
    temperature=0.2
)
print(f"Top candidate score: {candidates[0]['score']:.2f}")
print(f"Top candidate source: {candidates[0]['source']}")
```

**Sampling Process:**
1. Generate `num_samples` sequences by sampling from posterior at each site
2. If fields/couplings provided: Apply MCMC reshuffling to enforce DCA constraints
3. Score all sequences by posterior likelihood
4. Return top `num_top` from each source (ASR_posterior, MCMC_reshuffled)

---

### `save_candidates_to_fasta(candidates, posteriors, msa_original, msa_reshuffled, output_file)`

**Purpose:** Export top candidates plus MAP and consensus to FASTA file.

**Parameters:**
- `candidates` (list): List of candidate dicts from `sample_and_rank_sequences()`
- `posteriors` (np.ndarray): Root posteriors [L, 21]
- `msa_original` (np.ndarray): Original sampled MSA
- `msa_reshuffled` (np.ndarray): MCMC reshuffled MSA
- `output_file` (str): Path to output FASTA file

**Returns:**
- `output_file` (str): Path to saved FASTA file

**Output FASTA Headers:**
- `>1_MAP_sequence` - Maximum A Posteriori sequence
- `>2_consensus_sequence` - Consensus sequence
- `>3_ASR_posterior_rankXX_scoreYY` - Top 10 from ASR posterior sampling
- `>4_MCMC_reshuffled_rankXX_scoreYY` - Top 10 from MCMC reshuffled sampling

**Total Output:** 22 sequences (1 MAP + 1 consensus + 10 ASR + 10 MCMC)

**Example:**
```python
from utils.user_helper_functions import save_candidates_to_fasta
output = save_candidates_to_fasta(
    candidates=candidates,
    posteriors=posteriors,
    msa_original=msa_orig,
    msa_reshuffled=msa_resh,
    output_file='output_betaLac_candidates/candidates.fasta'
)
print(f"Saved to: {output}")
```

---

## ASR Functions

### Module: `utils.asr_torch_perfect`

#### `felsenstein_root_posteriors_optimized(tree, leaf_seq_dict, w, mu, device='cuda', batch_size=64, use_jit=True)`

**Purpose:** Compute ancestral state probabilities at tree root using Felsenstein algorithm.

**Parameters:**
- `tree`: PyTorch tree structure
- `leaf_seq_dict`: Dictionary of leaf sequences
- `w` (torch.Tensor): Site-wise equilibrium frequencies [L, 21]
- `mu` (float): Mutation rate
- `device` (str): 'cuda' or 'cpu'
- `batch_size` (int): Batch size for computation
- `use_jit` (bool): Use JIT compilation for speed

**Returns:**
- `posteriors` (torch.Tensor): Root posterior probabilities [L, 21]

---

#### `fasta_to_tensor_dict(fasta_file)`

**Purpose:** Load FASTA sequences into dictionary of PyTorch tensors.

**Parameters:**
- `fasta_file` (str): Path to FASTA file

**Returns:**
- `seq_dict` (dict): {sequence_name: tensor}

---

#### `newick_to_pytorch_tree(newick_file)`

**Purpose:** Parse Newick format tree into PyTorch structure.

**Parameters:**
- `newick_file` (str): Path to tree file (Newick format)

**Returns:**
- `tree`: PyTorch tree object

---

### Module: `utils.inferringMu`

#### `fit_mu(tree, msa_file)`

**Purpose:** Estimate mutation rate from data using maximum likelihood.

**Parameters:**
- `tree`: ETE3 tree object
- `msa_file` (str): Path to MSA file

**Returns:**
- `mu` (float): Estimated mutation rate

---

## MCMC Reshuffling

### Module: `utils.MCMC_reshuffling_torch_perfect`

#### `MCMC_columns_pytorch_efficient(MSA, couplings, fields_, T, scale=1.0, device='cuda', q=21, verbose=True)`

**Purpose:** Apply MCMC reshuffling to enforce DCA constraints on sequences.

**Parameters:**
- `MSA` (np.ndarray): Input MSA [N, L]
- `couplings` (np.ndarray): DCA coupling parameters
- `fields_` (np.ndarray): DCA field parameters
- `T` (float): Temperature (0-1, lower = stronger DCA constraints)
- `scale` (float): Scaling factor for MCMC steps
- `device` (str): 'cuda' or 'cpu'
- `q` (int): Alphabet size (21 for proteins)
- `verbose` (bool): Print progress

**Returns:**
- `MSA_reshuffled` (np.ndarray): MCMC reshuffled MSA

**Note:** Requires `context_independent_entropy` function in globals (auto-patched by helpers).

---

## Troubleshooting

### Issue: "NaN values in posteriors"
**Solution:** Handled automatically by `run_asr_pipeline()`. These typically occur at sites with no variation.

### Issue: MCMC reshuffling is slow
**Solution:** 
- Use `temperature` closer to 1.0 (fewer MCMC steps)
- Ensure CUDA is available: `torch.cuda.is_available()`
- Reduce `num_samples`

### Issue: "Tree leaf names don't match MSA"
**Solution:** Use cleaned tree and MSA from `cleaned_data_and_inferred_tree/` directory. Names are normalized there.

### Issue: Out of memory (GPU)
**Solution:**
- Reduce `batch_size` in `run_asr_pipeline()`
- Use `device='cpu'` instead
- Reduce `num_samples`

---

## Configuration Example

**Full configuration for a new dataset:**

```python
PROJECT_ROOT = '/path/to/phyloDCA_public/'
DATASET_NAME = 'myprotein'

# Files in data_myprotein/ directory
MSA_FILENAME = 'sequences.faa'
PARAMS_FILENAME = 'parameters.dat'

# Sampling parameters
NUM_SAMPLES = 1000
NUM_TOP_CANDIDATES = 10
MCMC_TEMPERATURE = 0.2
```

---

## Citation

If you use phyloDCA, please cite our preprint: https://www.biorxiv.org/content/10.64898/2026.06.08.731024v1

---

## Version
- phyloDCA version: Development
- Last updated: 2026-06-16
- Documentation generated automatically

---

## Contact & Support
For issues, see the example_usage_simple.ipynb notebook for complete working examples.
