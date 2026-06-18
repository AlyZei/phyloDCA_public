# phyloDCA Utilities Documentation

## Overview

`phyloDCA` provides utilities for coevolution-aware ancestral sequence reconstruction (ASR). The code combines phylogenetic ASR, Direct Coupling Analysis (DCA/Potts models), posterior sequence sampling, DCA-guided MCMC reshuffling, MSA/tree preprocessing, entropy/PCA analysis, and paper figure/export helpers.

Most reusable functions live in [utils/](utils/). The package-level [utils/__init__.py](utils/__init__.py) imports the public contents of the utility modules, so functions can often be imported either from their module or directly from `utils`.

```python
from utils.user_helper_functions import run_asr_pipeline
from utils.PottsEnergies import read_potts_parameters_proteins

# or, because utils/__init__.py re-exports many names:
from utils import read_fasta2, create_MSA_profile
```

## Data Conventions

- Protein sequences are usually integer encoded with 21 states.
- State `0` is the gap character `-`.
- States `1..20` correspond to amino acids `A C D E F G H I K L M N P Q R S T V W Y`.
- MSAs are usually NumPy arrays or PyTorch tensors with shape `[N, L]`, where `N` is the number of sequences and `L` is the number of sites.
- DCA fields usually have shape `[L, q]`.
- DCA couplings usually have shape `[L, L, q, q]`.
- ASR posteriors usually have shape `[L, q]`.

Some utilities call external command-line programs:

- `FastTree` is used by tree-building helpers.
- `cd-hit` is used by CD-HIT alignment-cleaning helpers.
- Plot/export utilities may require `matplotlib`, `sklearn`, `ete3`, `torch`, `numpy`, and optional LaTeX support.

## Quick Start Workflow

The notebook-oriented helper functions in [utils/user_helper_functions.py](utils/user_helper_functions.py) wrap the common workflow:

```python
from utils.user_helper_functions import (
    setup_paths,
    load_data,
    run_asr_pipeline,
    sample_and_rank_sequences,
    save_candidates_to_fasta,
)

paths = setup_paths("/home/user/phyloDCA_public", "betaLac")
msa_file = "data_betaLac/extant_msa/PF13354_noinsert_max19gaps_nodupl_noclose_BetaLact.faa"
params_file = "data_betaLac/Parameters_conv_Matteo_pc_BetaLact.dat"
tree_file = "cleaned_data_and_inferred_tree/tree/betaLactree_collapsed_noonlychild_midpointrooted.nwk"
msa, headers, fields, couplings = load_data(
    msa_file,
    params_file,
)
posteriors, seq_name = run_asr_pipeline(msa_file, tree_file, paths["output_asr"])
candidates, msa_reshuffled, msa_original = sample_and_rank_sequences(
    posteriors,
    num_samples=1000,
    num_top=10,
    fields=fields,
    couplings=couplings,
    temperature=0.2,
)
save_candidates_to_fasta(
    candidates,
    posteriors,
    msa_original,
    msa_reshuffled,
    f"{paths['output_candidates']}/{seq_name}_candidates.fasta",
)
```

If the cleaned tree file is not already present, generate it with the alignment/tree setup utilities or point `tree_file` to any matching Newick tree whose leaf names match the MSA headers.

## Module Reference

### `utils.user_helper_functions`

High-level, user-facing wrappers for launching the example ASR workflow.

| Function | Purpose |
| --- | --- |
| `setup_paths(project_root, dataset_name)` | Builds standard data/output paths for a dataset and creates ASR/candidate output folders. Returns a path dictionary. |
| `load_data(msa_file, params_file)` | Loads an integer-encoded FASTA MSA and DCA parameters. Returns `msa_array`, `headers`, `fields`, `couplings`. |
| `run_asr_pipeline(msa_file, tree_file, output_dir)` | Infers `mu`, loads the tree and leaf sequences, estimates site-wise frequencies, computes root posteriors with optimized Felsenstein ASR, fixes NaNs if present, saves `*_posteriors.npy`, and returns `(posteriors_np, seq_name)`. |
| `sample_and_rank_sequences(posteriors, num_samples=1000, num_top=10, fields=None, couplings=None, temperature=0.2)` | Samples posterior sequences, optionally applies DCA-guided MCMC reshuffling, scores candidates by posterior likelihood, and returns `(candidates, msa_reshuffled, msa_original)`. |
| `save_candidates_to_fasta(candidates, posteriors, msa_original, msa_reshuffled, output_file)` | Writes MAP, consensus, top posterior-sampled, and top reshuffled candidates to FASTA. |

### `utils.asr_torch_perfect`

Optimized PyTorch implementation of Felsenstein root posterior inference.

| Function/Class | Purpose |
| --- | --- |
| `TorchTree` | Lightweight tree container with child indices, branch lengths, leaf flags, and leaf names. |
| `newick_to_pytorch_tree(newick_file)` | Converts a Newick tree to `TorchTree` using postorder traversal. |
| `fasta_to_tensor_dict(fasta_path)` | Reads FASTA into `{sequence_name: torch.LongTensor}` using the project integer encoding. |
| `jc_log_transition_matrix_batch(W, w_batch)` | Builds batched log transition matrices for the Felsenstein propagator. |
| `stable_logsumexp_normalize_batch(log_probs)` | Numerically stable batched normalization from log probabilities to probabilities. |
| `batch_matmul_logsumexp(log_P, child_log_like)` | Batched log-space matrix multiplication for child likelihood propagation. |
| `felsenstein_batch_jit_core(...)` | TorchScript-compatible batched Felsenstein core. |
| `felsenstein_batch_fallback(...)` | Non-JIT batched fallback for more flexible tree handling. |
| `preprocess_tree_for_jit_fixed(tree, leaf_seq_by_name, device)` | Converts `TorchTree` and leaf tensors into JIT-friendly tensors. |
| `felsenstein_root_posteriors_optimized(tree, leaf_seq_by_name, w, mu, device='cuda', batch_size=64, use_jit=True, n_workers=None, use_multiprocessing=False)` | Main optimized root posterior routine. Returns `[L, q]` posteriors. |
| `felsenstein_single_site_fallback(tree, leaf_seq_by_name, w_site, mu, site_idx, device)` | Maximum-compatibility fallback for one site. |

### `utils.allThingsFelsenstein`

Original and exploratory Felsenstein/likelihood utilities. These functions are useful for validation, plotting, sampling, and frequency inference.

| Function | Purpose |
| --- | --- |
| `softmax_2d(array, axis=-1)` | Applies softmax to a 2D array along an axis. |
| `analyticalLogLikelihoodPerSite(MSA, emp_freq, w, site, q=21)` | Analytical site log-likelihood in the equilibrium limit. |
| `analyticalLikelihood(MSA, emp_freq, w)` | Product/total analytical likelihood over sites. |
| `FelsensteinNoNormalization(t, mu, w_site, site)` | Felsenstein posterior for a site without internal normalization. |
| `FelsensteinWithExponent(t, mu, w_site, site)` | Felsenstein posterior using the exponential transition form. |
| `FelsensteinWithExponentLogLikelihood(t, mu, w_site, site)` | Felsenstein likelihood in log-likelihood form. |
| `FelsensteinWithExponentLogLikelihoodNoPrior(t, mu, w_site, site)` | Log-likelihood variant without the root prior. |
| `FelsensteinAncestral(t, mu, w_site, site)` | Returns the ancestral posterior distribution for a chosen site/node. |
| `neg_log_likelihood(t, mu, w_site, site)` | Negative log-likelihood helper for optimization. |
| `plotLikelihoodConvergence(root_sequence, path_to_tree, mu_test=400, w_nb_to_test=10)` | Diagnostic plot for likelihood convergence as `mu` changes. |
| `plotPearsonCorrsWithReplicates(dict_dirs, title='Correlation analysis with replicates')` | Plots Pearson correlation analyses across replicate result dictionaries. |
| `plot_felsenstein_inference(data_folder, save_folder)` | Plots inferred/empirical frequencies against ground-truth frequencies. |
| `samplingAndLikelihood(path_to_tree, path_to_alignment, mu)` | Samples on a tree and evaluates likelihood for testing. |
| `softmax_projection(params)` | Projects unconstrained parameters to probabilities. |
| `objective(params, t, mu, site)` | Optimization objective for frequency/field inference. |
| `optimize_h(mu, h, t, site, lr=0.01, num_steps=500)` | Optimizes site fields/frequencies. |
| `inferring_w_from_MSA(...)` | Infers equilibrium frequencies from an MSA and tree. Saves inferred arrays and optional gradients. |
| `inferring_h_with_sampling(...)` | Infers fields using sampled tree data. |
| `pearsonCorrsH(...)` | Runs and stores Pearson correlations for inferred fields across `mu` values. |
| `launchFelsensteinAncestral(...)` | End-to-end ancestral posterior launch helper for a chosen tree node. |
| `samplingFelsenstein(probabilities, T, M=1000)` | Samples sequences from a probability dictionary using temperature weighting. |
| `plot_felsenstein_sampling_with_temperature(...)` | Plots energy and Hamming-distance behavior of Felsenstein samples across temperatures. |

Note: this module contains duplicate historical definitions of `optimize_h` and `inferring_w_from_MSA`; the later definitions override earlier ones when the module is imported.

### `utils.inferringMu`

Mutation-rate fitting from a tree and leaf MSA.

| Function | Purpose |
| --- | --- |
| `fit_mu(tree, file_path, max_leaves=1000, gap_code=0)` | Estimates the mutation rate `mu` by comparing tree distances and Hamming distances among leaves, optionally subsampling leaves. |

### `utils.plot_mu_inference_curves`

Plotting utilities for `mu` inference diagnostics.

| Function | Purpose |
| --- | --- |
| `fit_mu_and_get_plot_data(tree_path, fasta_path)` | Fits `mu` and returns binned/fit data for plotting. |
| `plot_mu_inference_curves_4panel(tree_path, fasta_paths_dict, mu_values, wt_name, figsize=(...), save_path=None)` | Creates a four-panel `mu` inference figure for different generated datasets. |

### `utils.PottsEnergies`

Potts/DCA parameter loading, coupling normalization, and energy computation.

| Function | Purpose |
| --- | --- |
| `ensure_tensor(x, dtype=None)` | Converts NumPy/array-like inputs to tensors while preserving tensors. |
| `reshape_couplings(couplings_original)` | Reshapes couplings from `[L, q, L, q]` to `[L, L, q, q]`. |
| `symmetrize_couplings(couplings)` | Fills missing coupling entries using `J[i,j,a,b] = J[j,i,b,a]`. |
| `verify_symmetry(couplings, tol=1e-6)` | Raises an error if couplings are not symmetric. |
| `readParametersFraZ(file_path, q=21)` | Reads Francesco/FraZ-format Potts parameters. Returns `(couplings, fields)`. |
| `get_fields_and_couplings_from_params(potts_parameters_lorenzo, device=None)` | Loads Lorenzo/bmDCA-format parameters via `utils_lore.load_params`. Returns `(fields, couplings)`. |
| `read_potts_parameters_proteins(potts_parameters_path, q=21, device=None)` | Format-flexible loader. Tries Lorenzo/bmDCA format, falls back to FraZ format, verifies symmetry, and returns `(fields, couplings)`. |
| `energy_site_gibbs(sequence, site, couplings, fields, q=21, device=None)` | Computes site-wise energies for all possible residues at one site. |
| `energy(sequence, couplings, fields)` | Computes total Potts energy for one sequence. |
| `energy_site_MCMC(sequence, site, amino_acid, couplings, fields)` | Computes mutation energy contribution for one proposed site/state. |
| `energy_site_MCMC_batch(sequences, sites, amino_acids, couplings, fields)` | Vectorized batch version for multiple proposed mutations. |
| `energy_of_msa(msa, fields, couplings)` | Computes energies for all sequences in an MSA. |
| `energy_of_sequence(sequence, fields, couplings)` | Optimized single-sequence Potts energy. |
| `getEnergyProfile(MSA, couplings, fields)` | Returns an energy vector/profile for an MSA. |
| `plotEnergyProfiles(MSAs, names, ML_seq, gibbs_seq, couplings, fields, save_path='')` | Plots energy distributions for MSAs plus selected reference sequences. |
| `plotEnergyProfilesGenerated(MSAs, names, root_seq, couplings, fields, title='', save_path='')` | Plots generated-sequence energy profiles against a root sequence. |
| `mutate_gibbs_steps(sequence, nb_steps, fields_, couplings, q=21)` | Gibbs-samples mutations from an initial sequence for a fixed number of steps. |

### `utils.MCMC_reshuffling_torch_perfect`

DCA-guided MCMC reshuffling of sampled alignments while preserving column composition.

| Function | Purpose |
| --- | --- |
| `sample_site_probability(MSA, type='entropy')` | Computes probabilities for choosing sites during MCMC: `uniform`, `entropy`, or `2^entropy`. |
| `ensure_tensor(x, dtype=None)` | Converts inputs to tensors if needed. |
| `get_deltaE(idx, chain, residue_old, residue_new, params, L, q)` | Efficiently computes energy differences for proposed swaps. |
| `integer_to_onehot(sequences, q)` | Converts integer-encoded `[N, L]` sequences to one-hot `[N, L, q]`. |
| `MCMC_columns_pytorch_efficient(MSA, couplings, fields_, T=0.2, scale=1.0, device='cuda', q=21, verbose=False)` | Main column-reshuffling MCMC. Proposes swaps that preserve site composition while favoring low Potts energy. Returns reshuffled MSA. |
| `shuffle_and_save_pytorch_optimized(...)` | Batch helper to reshuffle and save generated MSAs across sequences and `mu` values. |

### `utils.ci_and_cd_entropy`

Context-independent and context-dependent entropy analyses.

| Function | Purpose |
| --- | --- |
| `context_independent_entropy(MSA)` | Computes per-site entropy from MSA amino-acid frequencies. |
| `context_dependent_entropy(sequence, site, fields, couplings, q=21)` | Computes context-dependent entropy at one site in one sequence using Potts parameters. |
| `context_dependent_entropy_sequence(sequence, fields, couplings, q=21)` | Computes context-dependent entropy across all sites for one sequence. |
| `context_dependent_entropy_msa_torch(sequences, fields, couplings, q=21, device=...)` | Parallel PyTorch computation of context-dependent entropy for an MSA. |
| `plot_entropy_distributions(entropy_data, labels, colors=None, figsize=(...), linewidth=..., alpha=..., title=..., xlabel=..., ylabel=..., save_path=None, dpi=300)` | Plots one or more entropy distributions. |
| `compute_and_plot_entropy(prob_files, labels, **plot_kwargs)` | Loads probability files, computes entropy, and plots distributions. |

### `utils.potts_entropy`

Entropy estimation for Potts models.

| Function | Purpose |
| --- | --- |
| `entropy_ais(params, n_chains=..., n_beta_steps=..., n_sweeps_per_step=..., device=..., seed=...)` | Estimates Potts entropy using Annealed Importance Sampling from an independent-site reference model. |
| `entropy_field_variant_ti(params_ref, log_Z_ref, mean_E_ref, chains_ref, new_bias, n_quad_points=..., n_sweeps_per_point=..., n_final_sweeps=..., n_chains=..., device=...)` | Fast thermodynamic-integration entropy estimate for a model with changed fields and shared couplings. |
| `entropy_batch_field_variants(params_ref, log_Z_ref, mean_E_ref, chains_ref, variant_biases, n_quad_points=..., n_sweeps_per_point=..., n_chains=..., device=...)` | Applies the field-variant entropy estimate to many field sets. |

### `utils.toolsForTreesAndMSAs`

General FASTA/MSA/tree manipulation, statistics, plotting, and sequence encoding helpers.

| Function | Purpose |
| --- | --- |
| `convert_array_from_julia(sequence)` | Converts Julia-style amino-acid indices to Python encoding. |
| `read_fasta1(file_path)` | Reads FASTA and returns only the MSA array. |
| `read_fasta2(file_path)` | Reads FASTA and returns `(MSA, name_to_index, headers)`. |
| `read_fasta3(file_path)` | Reads FASTA and returns `(MSA, name_to_index, sequence_to_name)`. |
| `amino_acid_seq_to_int(amino_acid_seq, mapping_dict=amino_acid_mapping)` | Converts amino-acid string to integer sequence. |
| `int_to_amino_acid_seq(int_list)` | Converts integer sequence back to amino-acid string. |
| `write_MSA_to_fasta(cleaned_seqs, output_file, full_alignment_path=None, seq_header=None)` | Writes integer MSA to FASTA using original headers or a provided header. |
| `getShortAlignment(full_alignment, short_alignment, seq_nb)` | Randomly samples sequences from a larger alignment. |
| `keep_unique_sequences(input_file, output_file)` | Removes duplicate sequences from FASTA and writes unique sequences. |
| `remove_distance0_sequences_vectorized(unique_alignment, output_file)` | Removes distance-zero duplicate sequences using vectorized integer encodings. |
| `buildTree(alignment, tree_name)` | Builds a Newick tree from an alignment using FastTree. |
| `remove_gapped_sequences(alignment_file, percentage=0.2)` | Removes sequences whose gap fraction exceeds `percentage`. |
| `count_short_leaf_branches(tree_path, epsilon=1e-6)` | Counts very short branches leading to leaves. |
| `count_short_internal_branches(tree_path, epsilon=1e-6)` | Counts very short branches leading to internal nodes. |
| `prune_only_close_leaves(tree, cleaned_tree_name, threshold=1e-6)` | Prunes leaves connected by branches shorter than a threshold. |
| `binaryTree(path_to_tree, binary_tree_name)` | Converts/saves a binary tree representation. |
| `prune_subtree(path_to_tree, leaves_number, save_path='')` | Extracts a subtree with approximately the requested number of leaves. |
| `markChosenNode(t, chosen_index, node_name='chosen_node')` | Names a selected internal node. |
| `reroot(path_to_tree, chosen_index)` | Reroots a tree at a chosen node. |
| `fetchMSAFromLeafNames(tree_path, full_MSA_path, save_path='')` | Extracts MSA rows matching tree leaves. |
| `MSAFromLeafSequences(t, save_path='')` | Builds an MSA from sequences stored on tree leaves. |
| `gaps_count(seq)` | Counts gaps in a sequence. |
| `calculate_hamming_distance(seq1, seq2)` | Computes Hamming distance including gaps. |
| `calculate_hamming_distance_no_gaps(seq1, seq2)` | Computes Hamming distance ignoring positions with gaps. |
| `normalized_hamming_no_gaps(seq1, seq2)` | Computes normalized no-gap Hamming distance. |
| `average_distance_from_leaves_to_root(t)` | Computes average root-to-leaf distance. |
| `plot_hamming_vs_tree_distance(path_to_tree, path_to_MSA, nogaps=False, title=...)` | Plots sequence Hamming distance against tree distance for sibling pairs. |
| `plot_distance_child_to_parent_tree(path_to_tree, num_bins=5000, title=...)` | Plots branch length distribution from children to parents. |
| `plot_branch_length_distribution(path_to_tree)` | Plots branch-length histogram. |
| `plot_branch_length_distributions(path_to_full_tree, path_to_collapsed_tree, yscale_log=False, format=0)` | Compares branch-length distributions between two trees. |
| `collapse_short_branches(tree, outfile, length_threshold=1e-6)` | Collapses branches shorter than a threshold while preserving topology. |
| `collapse_only_children(tree, outfile)` | Collapses internal nodes with one child. |
| `midpoint_reroot(path_to_tree, format=0)` | Midpoint-reroots a tree. |
| `plot_topological_depth(path_to_tree, title='')` | Plots topological depth distribution. |
| `plot_leaf_to_root_depth(path_to_tree, title='')` | Plots leaf-to-root depth distribution. |
| `plot_compared_leaf_to_root_depth(path_to_tree1, path_to_tree2, title='', labels=[...])` | Compares leaf-to-root depths between trees. |
| `plot_compared_topological_depth(path_to_full_tree, path_to_collapsed_tree)` | Compares topological depth distributions. |
| `create_MSA_profile(prob_non_normalized, cardinal=100)` | Samples an MSA from per-site probability profiles. |
| `oneHotEncode(MSA)` | One-hot encodes an integer MSA with 21 states. |
| `get_pairwise_hamming_dist(msa, dist_matrix=False)` | Computes pairwise Hamming histogram, optionally a full distance matrix. |
| `get_eff(msa, eff_cutoff=0.8)` | Computes sequence weights/effective counts. |
| `getMeff(msa, eff_cutoff=0.8)` | Computes effective number of sequences. |
| `get_stats_MSA(MSA)` | Computes one-point frequencies, two-point frequencies, and covariance. |
| `plotStats(MSAs, names, save_path='')` | Plots MSA statistics for multiple alignments. |
| `plotWeights(weights, title, save_path='')` | Plots sequence weights. |
| `getFrequencyDistribution(MSA)` | Computes per-site amino-acid frequency distribution. |
| `normalizePerRow(w)` | Row-normalizes a matrix/profile. |
| `getReweightedDistribution(MSA, save_path='')` | Computes reweighted site-frequency distribution. |
| `randomW(L, q)` | Generates a random frequency/profile matrix. |
| `get_maximum_likelihood_sequence(probabilities)` | Returns the argmax state at each site. |
| `plotHistogramSimilarityScore(MSA)` | Plots pairwise similarity scores within an MSA. |
| `plot_histograms_together(hist_list, title, legend_labels=None, save_path=None, num_bins=54)` | Overlays histograms. |
| `hamiltonianFromW(distribution, epsilon=1e-70)` | Converts probabilities to Hamiltonian values with `-log(P + epsilon)`. |
| `find_max_indices2D(masked_array)` | Finds max indices in a 2D array while handling NaNs/masks. |
| `max_indices(array)` | Returns per-row argmax indices. |
| `reshape_array(arr)` | Reshapes `[a, b, c]` to `[a, b*c]`. |
| `kl_divergence_2d(p, q, epsilon=1e-8)` | Computes KL divergence between 2D discrete distributions. |
| `rescale_tree(newick_file, output_file)` | Rescales branches so leaf-to-root distances are normalized. |
| `get_tree_depth(tree)` | Computes maximum tree depth. |
| `get_nodes_at_depth(tree, depth)` | Returns tree nodes at a given depth. |
| `soloTreeLeveling(path_to_tree, path_to_new_tree)` | Levels a tree using an iterative strategy. |
| `soloTreeLevelingRecursive(path_to_tree, path_to_new_tree)` | Recursive tree-leveling variant. |
| `renaming_tree_nodes(tree_path, strategy='preorder', starting_index=0, outfile_name=None)` | Renames tree nodes by traversal order. |
| `append_sequence_to_fasta(input_file, output_file, header, sequence)` | Appends one sequence to a FASTA and writes a new file. |
| `reformat_fasta_to_single_line(input_fasta_path, output_fasta_path)` | Converts multi-line FASTA records to single-line sequence records. |
| `get_subtree_consensus(node_of_interest, tree_path, msa_path, save_name, save_folder, reweighted=False)` | Computes consensus for leaves under a chosen subtree. |

### `utils.alignment_cleaning_cdhit`

Helpers for running CD-HIT on aligned sequences while preserving gap positions.

| Function | Purpose |
| --- | --- |
| `read_fasta(path)` | Reads FASTA into an ordered header-to-sequence mapping. |
| `write_fasta(seqs, path)` | Writes a header-to-sequence mapping to FASTA. |
| `mask_gaps(seq, gap_char='-', mask_char='X')` | Replaces gaps before CD-HIT clustering. |
| `restore_gaps(masked_seq, original_seq, gap_char='-', mask_char='X')` | Restores gap positions after clustering. |
| `run_cdhit(input_fasta, output_fasta, identity)` | Runs `cd-hit` with the requested identity threshold. |
| `cdhit_on_alignment(aligned_fasta, identity=0.97, gap_char='-', mask_char='X', output_fasta=None)` | Complete aligned-FASTA CD-HIT workflow with gap masking/restoration. |

### `utils.alignmentAndTreeSetUp`

Higher-level alignment and tree cleaning workflows.

| Function | Purpose |
| --- | --- |
| `cleanAlignmentAndTree_cdhit(full_alignment, tree_folder, alignment_folder, family_name='betaLac', prune=False, leaf_number=300, length_threshold=1e-6, save_folder='PCA/', identity=0.97)` | Cleans an alignment with CD-HIT, removes problematic sequences, builds/collapses/reroots a tree, and optionally prunes. |
| `differentGapPercentages(unique_alignment, gaps_ensemble=[0.05, 0.1, 0.15, 0.2], save_folder='PCA/', family_name='DBD')` | Generates alignments filtered at multiple gap thresholds. |
| `cleanAlignmentAndTree(full_alignment, tree_save_folder, alignment_save_folder, family_name='betaLac', prune=False, leaf_number=300, length_threshold=1e-6, save_folder='PCA/')` | Legacy cleaning workflow without the CD-HIT-specific wrapper. |

### `utils.samplerOnTrees`

Forward simulation of sequence evolution along trees.

| Function | Purpose |
| --- | --- |
| `mutate_felsenstein(tt, mu, w)` | Evolves sequences down a tree using the Felsenstein propagator and site profiles. Modifies the tree in place. |
| `mutate_metropolis(tt, mu, fields_, couplings, q=21)` | Evolves sequences down a tree using Metropolis mutations under a Potts model. |
| `mutate_gibbs(tt, mu, fields_, couplings, q=21)` | Evolves sequences down a tree using Gibbs mutations under a Potts model. |
| `get_gibbs_sampled_MSA_node(path_to_tree, mu, node_name, root_sequence, fields, couplings, chosen_index=..., size_=..., save=...)` | Samples descendant sequences from a chosen node/root sequence and can save the generated MSA. |

### `utils.utils_lore`

General sequence/parameter I/O utilities adapted for multiple alphabets.

| Function | Purpose |
| --- | --- |
| `get_tokens(alphabet)` | Returns token alphabet for `"protein"`, `"rna"`, `"dna"`, or a custom token string. |
| `encode_sequence(sequence, tokens)` | Encodes one sequence or a list/array of sequences into integers. |
| `decode_sequence(sequence, tokens)` | Decodes integer sequences back to token strings. |
| `import_from_fasta(fasta_name, tokens=None, filter_sequences=True, remove_duplicates=True)` | Imports FASTA headers/sequences with optional encoding, alphabet filtering, and duplicate removal. |
| `write_fasta(fname, headers, sequences, numeric_input=False, remove_gaps=False, tokens='protein')` | Writes sequences to FASTA, optionally decoding numeric input and removing gaps. |
| `compute_weights(data, th=0.8, device=..., dtype=...)` | Computes sequence reweighting factors based on identity threshold. |
| `validate_alphabet(sequences, tokens)` | Ensures sequences only contain allowed alphabet tokens. |
| `set_zerosum_gauge(params)` | Applies zero-sum gauge to Potts coupling parameters. |
| `load_params(fname, tokens='protein', device=..., dtype=...)` | Loads Potts parameters into a dictionary with `bias` and `coupling_matrix`. |

### `utils.utils`

Small general-purpose helpers.

| Function | Purpose |
| --- | --- |
| `generate_colors(n)` | Returns `n` colors from the Viridis colormap. |
| `to_numpy(tensor)` | Converts a tensor to a CPU NumPy array; leaves non-tensors unchanged. |
| `to_tensor(array, device='cpu')` | Converts a NumPy array to a tensor on the requested device; leaves non-arrays unchanged. |
| `createFolder(folder_name)` | Creates a folder if it does not exist. |
| `get_variable_name(var)` | Returns the caller-local variable name for an object. |
| `get_all_file_paths(base_path)` | Recursively lists files under a directory. |
| `remove_charsequence(longer_string, sequence_to_remove)` | Removes a substring from a string. |

### `utils.pca_tools`

PCA utilities for comparing natural, generated, consensus, and reshuffled MSAs.

| Function | Purpose |
| --- | --- |
| `one_hot_encode_msa(input, alphabet=None, ignore_gaps_in_encoding=True, seq_number_limit=np.inf)` | One-hot encodes a FASTA path, MSA array, or sequence list for PCA. |
| `plot_pca_grid_msas_leo_trees(...)` | Creates grid PCA plots for tree-generated MSAs, roots, consensus sequences, and optional reshuffled samples. |
| `compute_weights(data, th=0.8, device=..., dtype=...)` | Computes sequence reweighting factors for PCA/resampling. |
| `resample_sequences(data, weights, nextract)` | Samples sequences with replacement according to weights. |

### `utils.improved_sklearn_pca`

Simpler sklearn-based PCA comparison plots.

| Function | Purpose |
| --- | --- |
| `read_fasta_lit(file_path)` | Reads FASTA into `(MSA, name_to_index, headers)`. |
| `compare_msas_pca(fasta_paths, labels=None, n_components=2, save_folder=None, ignore_gaps_in_encoding=True, file_name='pca_comparison')` | Compares multiple FASTA/MSA files in PCA space and optionally saves the figure. |

### `utils.convert_potts_params_format`

Command-line and programmatic conversion between numeric-state and token-state Potts parameter files.

| Function | Purpose |
| --- | --- |
| `detect_param_format(path)` | Detects whether `h`/`J` parameter lines use numeric states or alphabet tokens. |
| `convert_numeric_to_token(in_path, out_path, tokens=DEFAULT_TOKENS)` | Converts numeric amino-acid indices to token labels. |
| `convert_token_to_numeric(in_path, out_path, tokens=DEFAULT_TOKENS)` | Converts token labels back to numeric amino-acid indices. |
| `main()` | CLI entry point. |

Example:

```bash
python -m utils.convert_potts_params_format --input input.dat --output output.dat --to token
```

### `utils.export_candidate_fastas`

Export top reconstructed candidates to FASTA files for downstream inspection or figure 5-style analyses.

| Function | Purpose |
| --- | --- |
| `export_candidate_fastas_for_fig5(output_dir, sequences, mu_values_reduced, GT_sequences, msa_folder, posterior_folder, consensus_directory, fields_, couplings, M=..., T=..., data_prefix='Beta', energy_keep_pct=..., topN=10, n_roots=5, remove_gaps=False)` | Writes one FASTA per `mu` containing GT, MAP, consensus, and top candidate sequences. Returns written paths. |
| `export_candidate_fastas_from_export_config(export_config, output_dir, topN=10, n_roots=5, remove_gaps=False)` | Wrapper that reads the standard paper-export config dictionary. |
| `main()` | CLI entry point. |

### `utils.paper_figures_export`

Large plotting/export module for reproducing paper figures. It includes loaders, scoring utilities, figure builders, and PDF export orchestration.

Core scoring/loading helpers:

| Function | Purpose |
| --- | --- |
| `yang_score(sequence, posterior_prob)` | Average posterior probability of a sequence under an ancestral posterior. |
| `get_pairwise_hamming_dist(msa_tensor, dist_matrix=False, batch_size=500)` | Memory-efficient pairwise Hamming histogram. |
| `load_ancestral_posterior(folder, seq, mu, prefix='DBD')` | Loads an ancestral posterior probability matrix. |
| `load_Felsenstein_samples(msa_folder, seq, mu, M)` | Loads Felsenstein-sampled MSA files. |
| `load_reshuffled_MSA(msa_folder, seq, mu, M, T)` | Loads DCA-reshuffled MSA files. |
| `compute_intra_pairwise_hamming(msa)` | Computes intra-MSA pairwise Hamming distances. |
| `normalize(v)` | Normalizes values to `[0, 1]`. |
| `compute_all_scores(dca_seqs, posterior=None)` | Computes energy/posterior/diversity scoring metrics for candidate sequences. |
| `select_topN(dca_seqs, energies, posterior=None, scoring='pairwise', percentage=None, topN=10)` | Selects top candidate sequences by the requested scoring rule. |

Figure builders:

| Function | Purpose |
| --- | --- |
| `create_figure2(...)` | Builds Figure 2 panels for generated leaf diversity, pairwise distances, and optional PCA overlays. |
| `create_supplementary_figure2(...)` | Supplementary Figure 2 using GT-to-leaf Hamming distance as x-axis. |
| `create_figure3(...)` | MAP/consensus/GT distance comparisons. |
| `create_supplementary_figure3(...)` | Supplementary Figure 3 with transformed x-axis. |
| `create_supplementary_figure3_bis(...)` | Supplementary Figure 3 bis variant. |
| `create_supplementary_figure4(...)` | Supplementary panels derived from Figure 3 quantities. |
| `create_supplementary_map_confidence_vs_gt_distance(...)` | MAP confidence versus GT distance supplementary figure. |
| `create_figure4(...)` | Candidate ensemble diversity/energy/reconstruction figure. |
| `create_figure5_boxplots(...)` | Boxplot version of Figure 5 analyses. |
| `create_figure5_bis(...)` | Figure 5 bis candidate scoring/selection figure. |
| `create_figure5_bis_transposed(...)` | Transposed layout for Figure 5 bis. |
| `save_figure5_bis_transposed_legend(out_path, use_latex=True)` | Saves standalone legend for the transposed Figure 5 bis. |
| `create_supplementary_figure5_bis_legacy(...)` | Legacy supplementary Figure 5 bis. |
| `create_figure5_quat(...)` | Additional Figure 5 variant. |
| `create_figure5_bis_supplementary(...)` | Supplementary Figure 5 bis figure. |
| `create_supplementary_top10_on_gridplot(...)` | Plots top-10 candidates on a gridplot. |
| `create_supplementary_top10_on_reweighted_pca(...)` | Plots top-10 candidates in reweighted PCA space. |
| `save_supplementary_top10_on_reweighted_pca_legend(...)` | Saves standalone legend for the reweighted PCA supplementary figure. |
| `create_supplementary_context_dependent_entropy_figure(...)` | Builds context-dependent entropy distributions with highlighted WT values. |
| `run_supplementary_context_entropy_from_defaults(project_root, family, out_path, use_latex=True, max_msa_sequences=None)` | Runs only the entropy supplementary figure using default family paths. |
| `create_figure14(...)` | Posterior score versus GT distance by GT sequence. |
| `create_figure15(...)` | Posterior score versus GT distance by `mu` and GT sequence. |
| `create_figure16(...)` | Pearson correlation between posterior score and GT distance across `mu`. |

Export orchestration:

| Function | Purpose |
| --- | --- |
| `export_all_figures_to_pdf(...)` | Exports the main paper figures into one PDF. |
| `export_all_figures_to_individual_pdfs(config, set_of_figures=[2,3,4,5])` | Exports selected figures to individual PDF files. |
| `quick_export(config, set_of_figures=[2,3,4,5,6,7,8,9,10,11], individual=False)` | Convenience wrapper around the figure export routines. |

### `utils.export_supp_figures`

Standalone supplementary figure export pipeline for tree/MSA cleaning diagnostics.

| Function/Class | Purpose |
| --- | --- |
| `configure_matplotlib(use_tex=True)` | Configures matplotlib style and optional LaTeX rendering. |
| `BetaLacPaths` | Dataclass containing standard beta-lactamase input/output paths. |
| `default_paths(project_root)` | Builds `BetaLacPaths` from a project root. |
| `sibling_hamming_vs_tree_distance(tree_path, msa_path, nogaps)` | Computes sibling Hamming distance versus tree distance. |
| `child_parent_branch_lengths(tree_path)` | Returns child-parent branch lengths. |
| `node_depths(tree_path)` | Returns node topological depths. |
| `branch_lengths(tree_path)` | Returns all branch lengths. |
| `pairwise_hamming_histogram(msa, seq_len, pair_sample_limit, seed)` | Computes pairwise Hamming histogram, optionally with pair subsampling. |
| `one_hot_21(msa)` | One-hot encodes integer MSA with 21 states. |
| `sample_rows(arr, limit, seed)` | Subsamples rows reproducibly. |
| `fit_reference_pca(reference_msa, max_ref_sequences, seed)` | Fits scaler/PCA on a reference MSA. |
| `project_msa(msa, scaler, pca, max_sequences, seed)` | Projects an MSA into an existing PCA space. |
| `save_fig(fig, out_path)` | Saves a matplotlib figure with consistent options. |
| `add_large_panel_legend(ax, fontsize, ncol, loc)` | Adds a readable legend to a panel. |
| `plot_hist_logy(...)`, `plot_scatter(...)`, `plot_two_hist_overlay(...)`, `plot_depth_overlay(...)`, `plot_pairwise_histograms(...)`, `plot_reweighted_pca(...)` | Individual plotting helpers. |
| `plot_full_panel(...)` | Builds the full supplementary diagnostic panel. |
| `build_full_panel_legend_text()` | Returns the legend text used by the full panel. |
| `run_export(project_root, out_fig_dir, temp_dir, rerun_cleaning=False, use_tex=True, max_pca_ref=..., max_pca_proj=..., pair_sample_limit=...)` | Runs the full supplementary export workflow. |
| `build_arg_parser()` | CLI parser builder. |
| `main()` | CLI entry point. |

### `utils.pca_tools`, `utils.improved_sklearn_pca`, and Figure PCA Utilities

Use `pca_tools` for the paper's tree/grid PCA workflows and `improved_sklearn_pca` for simpler multi-MSA PCA comparisons. Both use one-hot sequence encodings and can ignore gaps during encoding.

### `utils.export_candidate_fastas`, `utils.paper_figures_export`, and `utils.export_supp_figures`

These modules are primarily analysis/export layers. They assume the file naming conventions used by the paper workflow. For new projects, prefer using the lower-level loaders/scorers first, then adapt the export config dictionaries.

## Common Imports by Task

ASR:

```python
from utils.inferringMu import fit_mu
from utils.asr_torch_perfect import newick_to_pytorch_tree, fasta_to_tensor_dict
from utils.asr_torch_perfect import felsenstein_root_posteriors_optimized
```

Potts/DCA:

```python
from utils.PottsEnergies import read_potts_parameters_proteins, energy_of_msa
from utils.MCMC_reshuffling_torch_perfect import MCMC_columns_pytorch_efficient
```

MSA/tree manipulation:

```python
from utils.toolsForTreesAndMSAs import read_fasta2, write_MSA_to_fasta, buildTree
from utils.toolsForTreesAndMSAs import collapse_short_branches, midpoint_reroot
```

Candidate export:

```python
from utils.user_helper_functions import sample_and_rank_sequences, save_candidates_to_fasta
from utils.export_candidate_fastas import export_candidate_fastas_for_fig5
```

Entropy/PCA:

```python
from utils.ci_and_cd_entropy import context_independent_entropy
from utils.ci_and_cd_entropy import context_dependent_entropy_msa_torch
from utils.pca_tools import one_hot_encode_msa
```

## Troubleshooting

- If `read_potts_parameters_proteins` fails, check whether the parameter file uses numeric states or amino-acid tokens. Use `utils.convert_potts_params_format` to convert formats.
- If ASR returns NaNs, inspect zero-frequency sites in `w`; `run_asr_pipeline` replaces NaNs with uniform probabilities as a fallback.
- If `buildTree` fails, make sure `FastTree` is installed and available on `PATH`.
- If CD-HIT cleaning fails, make sure `cd-hit` is installed and available on `PATH`.
- If CUDA is unavailable, most PyTorch utilities fall back to CPU, but large MCMC/ASR jobs may be slow.
- If figure export fails with LaTeX errors, rerun export helpers with `use_latex=False`.
