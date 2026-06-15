from typing import Tuple, Iterable, Dict, List
import numpy as np
import random
from ete3 import Tree

from utils.improved_sklearn_pca import compare_msas_pca
from utils.toolsForTreesAndMSAs import buildTree, count_short_internal_branches, count_short_leaf_branches, fetchMSAFromLeafNames, gaps_count, get_eff, getMeff, int_to_amino_acid_seq, keep_unique_sequences, midpoint_reroot, plot_branch_length_distributions, plot_compared_leaf_to_root_depth, plot_distance_child_to_parent_tree, plot_hamming_vs_tree_distance, plot_histograms_together, prune_subtree, read_fasta1, remove_distance0_sequences_vectorized, remove_gapped_sequences, collapse_short_branches, collapse_only_children
from utils.utils import createFolder, remove_charsequence
from utils.alignment_cleaning_cdhit import cdhit_on_alignment

def cleanAlignmentAndTree_cdhit(full_alignment, tree_folder, alignment_folder, family_name='betaLac', prune=False, leaf_number=300, length_threshold=10**(-6), save_folder='PCA/', identity = 0.97):
    """
    - The full alignment from the DBD family is downloaded directly so its name is kept as was: "DBD_alignment.uniref90.cov80.a2m"
	
	- It contains multiple sequences that are identical: we remove them using np.unique and store the resulting alignment in "DBD_alignment.uniref90.cov80_nodupli.fasta" (to do)
	
	- To generate a tree, we also want to remove sequences that are at distance 0 from one another considering the "gapped Hamming distance" that returns 0 for dist_nogaps('AV', 'A_'). Warning: this distance function is not transitive and therefore the process might give different results depending on the order in which we go through the alignment. Regardless, we consider this negligible, run this process once, check that it got rid of all couples of sequences at distance 0 from one another, and store the resulting alignment in "DBD_cleaned.fasta"
	
	- The last interesting alignment is the one we get from the leaves of the "collapsed tree" (see 2. Trees for more details). It is a sub-alignment of "DBD_cleaned.fasta" and is stored in "DBD_collapsed.fasta"
	
	- Finally, from the collapsed tree with no only children, we can prune a subtree of any given number of leaves for our experiments. The resulting alignment from the leaves of the subtree is a sub-alignment of "DBD_collapsed.fasta" and is stored in "DBD_collapsed_prunedsubtree301.fasta", 301 being the number of leaves of that specific subtree.
    
    TREES

    - The first tree generated with FastTree is from "DBD_cleaned.fasta". It is generated with the parameters -gamma -wag -pseudo (see Annex for the justification of the parameters), and stored in "DBDtree_fromcleaned.nwk"
	
	- The second tree aims to collapse branches whose length is below a certain threshold (10^-6), and set the name of the parent as the name of its collapsed child. Done on the leaves AND the internal nodes. The resulting tree is stored in "DBDtree_collapsed.nwk"
	
	Please note that the root of these two trees is random. These trees should be rerooted at midpoint with the ete3 library if needed.
	
	- The third interesting tree is the one with collapsed branches, from which we also chose to remove nodes that are "only children", meaning that they do not have siblings, by setting the only child's children as the only child's parent's children, and setting the resulting branch length as the sum of the two previous ones. This tree is stored in "DBDtree_collapsed_noonlychild.nwk"
	
	- The midpoint rooted version of the trees is stored with the same names as their random rooted version, with the addition of "midpointrooted" at the end.
	
	For example, the midpoint rooted version of "DBDtree_collapsed_noonlychild.nwk" is stored in "DBDtree_collapsed_noonlychild_midpointrooted.nwk"
	
	- From "DBDtree_collapsed_noonlychild_midpointrooted.nwk", we prune a subtree of approximately 300 leaves (turns out this specific one has 301 leaves), and we store it (without rerooting it) in "DBDtree_collapsed_noonlychild_midpointrooted_prunedsubtree301.nwk"

    """
    # Making sure the folders exist
    createFolder(alignment_folder)
    createFolder(tree_folder)
    if save_folder:
        createFolder(save_folder)

    # Defining the names of the alignments and the trees:
    unique_alignment=remove_charsequence(full_alignment, '.a2m')+'_nodupli.fasta'
    cleaned_alignment=alignment_folder+family_name+f'_cleaned_{identity}.fasta'
    collapsed_alignment=alignment_folder+family_name+'_collapsed.fasta'

    clean_tree=tree_folder+family_name+'tree_fromcleaned.nwk'
    midpoint_rooted_clean_tree=tree_folder+family_name+'tree_fromcleaned_midpointrooted.nwk'
    collapsed_tree=tree_folder+family_name+'tree_collapsed.nwk'
    collapsed_no_only_children_tree=tree_folder+family_name+"tree_collapsed_noonlychild.nwk"
    collapsed_no_only_children_midpoint_tree = tree_folder+family_name+"tree_collapsed_noonlychild_midpointrooted.nwk"
    if prune:
        pruned_tree=tree_folder+family_name+"tree_collapsed_noonlychild_midpointrooted_prunedsubtree"+str(leaf_number)+".nwk"

    # Getting only unique sequences with np.unique
    unique_sequences=keep_unique_sequences(full_alignment, unique_alignment)

    # Getting the cleaned alignment using cd-hit
    cdhit_on_alignment(
    aligned_fasta=unique_alignment,
    identity=identity,
    output_fasta=cleaned_alignment
    )

    # Build the tree with FastTree
    buildTree(cleaned_alignment, clean_tree)

    # Plot tree info
    print('On the clean tree')
    plot_distance_child_to_parent_tree(clean_tree, num_bins=5000, title='Clean tree - time on the tree between leaf and parent')

    print('On the clean tree, counting gaps')
    plot_hamming_vs_tree_distance(clean_tree, cleaned_alignment, nogaps=False, title='Clean tree pairwise Hamming distance vs. tree distance (w gaps)')

    print('On the clean tree, not counting gaps')
    plot_hamming_vs_tree_distance(clean_tree, cleaned_alignment, nogaps=True, title='Clean tree pairwise Hamming distance vs. tree distance (no gaps)')

    # Midpoint Reroot the clean tree
    midpoint_reroot(clean_tree)
    #t=Tree(midpoint_rooted_clean_tree)

    # Plot the compared distance from leaves to root between cleaned tree and midpoint-rerooted cleaned tree
    plot_compared_leaf_to_root_depth(clean_tree, midpoint_rooted_clean_tree, title='Distance from leaf to root after midpoint rerooting', labels=['clean tree', 'clean tree rerooted'])

    # Print the number of problematic internal nodes + the number of problematic leaves
    print('Number of leaf-branches below the threshold', str(length_threshold), 'is', count_short_leaf_branches(clean_tree, epsilon=length_threshold))

    print('Number of strictly internal branches below the threshold', str(length_threshold), 'is', count_short_internal_branches(clean_tree, epsilon=length_threshold))

    t = Tree(clean_tree, format = 0)

    # Collapse all branches (internal included) below 'length threshold'
    collapse_short_branches(t, collapsed_tree, length_threshold)

    # Compare branch length repartition on collapsed and non-collapsed trees
    plot_branch_length_distributions(clean_tree, collapsed_tree, yscale_log=True, format = 0)

    # Collapse nodes that have an only child. Maintain branch length.
    t=Tree(collapsed_tree, format = 0)
    collapse_only_children(t, collapsed_no_only_children_tree)

    #Midpoint reroot at the end
    midpoint_reroot(collapsed_no_only_children_tree)

    # Get the corresponding alignment
    fetchMSAFromLeafNames(collapsed_no_only_children_tree, cleaned_alignment, save_path=collapsed_alignment)

    # Compare the cleaned alignment and the collapsed one
    collapsed_sequences=read_fasta1(collapsed_alignment)
    compare_msas_pca([cleaned_alignment, collapsed_alignment], ['Cleaned MSA', 'Collapsed MSA'],
                    save_folder=save_folder,
                    file_name=f"{family_name}_cleaned_vs_collapsed_PCA.png")

    # Compare the pairwise Hamming distances
    original_msa=read_fasta1(full_alignment)
    cleaned_msa=read_fasta1(cleaned_alignment)
    L= len(original_msa[0])
    original_dist=get_pairwise_hamming_dist(original_msa, dist_matrix=False)
    clean_dist=get_pairwise_hamming_dist(cleaned_msa, dist_matrix=False)
    collapsed_dist=get_pairwise_hamming_dist(collapsed_sequences, dist_matrix=False)
        # Between the original and the cleaned

    plot_histograms_together([original_dist, clean_dist], 'Comparison of pairwise hamming distances', legend_labels=['original MSA', 'cleaned MSA'], save_path=None, num_bins=L+1)
        # Between the cleaned and the collapsed
    plot_histograms_together([clean_dist, collapsed_dist], 'Comparison of pairwise hamming distances', legend_labels=['cleaned MSA', 'collapsed MSA'], save_path=None, num_bins=L+1)

    if prune:
        prune_subtree(collapsed_no_only_children_tree, leaf_number, pruned_tree)
        pruned_alignment = alignment_folder + family_name + '_collapsed_prunedsubtree' + str(leaf_number) + '.fasta'
        # Plot the resulting PCA
        pruned_sequences=fetchMSAFromLeafNames(pruned_tree, cleaned_alignment, save_path=pruned_alignment)
        compare_msas_pca([cleaned_alignment, pruned_alignment], ['Cleaned MSA', 'MSA from pruned tree'],
                    save_folder=save_folder,
                    file_name=f"{family_name}_cleaned_vs_pruned_PCA.png")

    print('All alignments saved to', alignment_folder)
    print('Final tree is', collapsed_no_only_children_tree)
    print('Final full alignment is', collapsed_alignment)

    print(f'Full alignment contains {len(original_msa)}, its Meff={getMeff(original_msa)}')
    print(f'Full alignment without duplicates contains {len(unique_sequences)}, its Meff={getMeff(unique_sequences)}')
    print(f'There are {len(cleaned_msa)} sequences at distance>=1 from each other, their Meff={getMeff(cleaned_msa)}')
    print(f'There are {len(collapsed_sequences)} leaves on the final tree, their Meff={getMeff(collapsed_sequences)}')
    return original_dist, clean_dist, collapsed_dist
    

def differentGapPercentages(unique_alignment, gaps_ensemble=[0.05, 0.1, 0.15, 0.2], save_folder='PCA/', family_name='DBD'):
    for gap in gaps_ensemble:
        gapped_msa, gapped_alignment = remove_gapped_sequences(unique_alignment, percentage=gap)
        compare_msas_pca([unique_alignment, gapped_alignment], ['Original MSA', f'MSA with {gap*100} percent gaps'],
                    save_folder=save_folder,
                    file_name=f"{family_name}_original_vs_{gap}_gapped_PCA.png")


##############################
#----------------------------
##############################

# # v1 code - much longer, but used for the paper
# from typing import Tuple, Iterable, Dict, List
# import numpy as np
# import random
# from ete3 import Tree

# from utils.improved_sklearn_pca import compare_msas_pca
# from utils.toolsForTreesAndMSAs import buildTree, collapse_only_children, collapse_short_branches, count_short_internal_branches, count_short_leaf_branches, fetchMSAFromLeafNames, gaps_count, get_eff, get_pairwise_hamming_dist, getMeff, int_to_amino_acid_seq, keep_unique_sequences, midpoint_reroot, plot_branch_length_distributions, plot_compared_leaf_to_root_depth, plot_distance_child_to_parent_tree, plot_hamming_vs_tree_distance, plot_histograms_together, prune_subtree, read_fasta1, remove_distance0_sequences, remove_gapped_sequences
# from utils.utils import createFolder, remove_charsequence

def cleanAlignmentAndTree(full_alignment, tree_folder, alignment_folder, family_name='betaLac', prune=False, leaf_number=300, length_threshold=10**(-6), save_folder='PCA/'):
    """
    - The full alignment from the DBD family is downloaded directly so its name is kept as was: "DBD_alignment.uniref90.cov80.a2m"
	
	- It contains multiple sequences that are identical: we remove them using np.unique and store the resulting alignment in "DBD_alignment.uniref90.cov80_nodupli.fasta" (to do)
	
	- To generate a tree, we also want to remove sequences that are at distance 0 from one another considering the "gapped Hamming distance" that returns 0 for dist_nogaps('AV', 'A_'). Warning: this distance function is not transitive and therefore the process might give different results depending on the order in which we go through the alignment. Regardless, we consider this negligible, run this process once, check that it got rid of all couples of sequences at distance 0 from one another, and store the resulting alignment in "DBD_cleaned.fasta"
	
	- The last interesting alignment is the one we get from the leaves of the "collapsed tree" (see 2. Trees for more details). It is a sub-alignment of "DBD_cleaned.fasta" and is stored in "DBD_collapsed.fasta"
	
	- Finally, from the collapsed tree with no only children, we can prune a subtree of any given number of leaves for our experiments. The resulting alignment from the leaves of the subtree is a sub-alignment of "DBD_collapsed.fasta" and is stored in "DBD_collapsed_prunedsubtree301.fasta", 301 being the number of leaves of that specific subtree.
    
    TREES

    - The first tree generated with FastTree is from "DBD_cleaned.fasta". It is generated with the parameters -gamma -wag -pseudo (see Annex for the justification of the parameters), and stored in "DBDtree_fromcleaned.nwk"
	
	- The second tree aims to collapse branches whose length is below a certain threshold (10^-6), and set the name of the parent as the name of its collapsed child. Done on the leaves AND the internal nodes. The resulting tree is stored in "DBDtree_collapsed.nwk"
	
	Please note that the root of these two trees is random. These trees should be rerooted at midpoint with the ete3 library if needed.
	
	- The third interesting tree is the one with collapsed branches, from which we also chose to remove nodes that are "only children", meaning that they do not have siblings, by setting the only child's children as the only child's parent's children, and setting the resulting branch length as the sum of the two previous ones. This tree is stored in "DBDtree_collapsed_noonlychild.nwk"
	
	- The midpoint rooted version of the trees is stored with the same names as their random rooted version, with the addition of "midpointrooted" at the end.
	
	For example, the midpoint rooted version of "DBDtree_collapsed_noonlychild.nwk" is stored in "DBDtree_collapsed_noonlychild_midpointrooted.nwk"
	
	- From "DBDtree_collapsed_noonlychild_midpointrooted.nwk", we prune a subtree of approximately 300 leaves (turns out this specific one has 301 leaves), and we store it (without rerooting it) in "DBDtree_collapsed_noonlychild_midpointrooted_prunedsubtree301.nwk"

    """
    # Making sure the folders exist
    createFolder(alignment_folder)
    createFolder(tree_folder)
    if save_folder:
        createFolder(save_folder)

    # Defining the names of the alignments and the trees:
    unique_alignment=remove_charsequence(full_alignment, '.a2m')+'_nodupli.fasta'
    cleaned_alignment=alignment_folder+family_name+'_cleaned.fasta'
    collapsed_alignment=alignment_folder+family_name+'_collapsed.fasta'

    clean_tree=tree_folder+family_name+'tree_fromcleaned.nwk'
    midpoint_rooted_clean_tree=tree_folder+family_name+'tree_fromcleaned_midpointrooted.nwk'
    collapsed_tree=tree_folder+family_name+'tree_collapsed_midpointrooted.nwk'
    collapsed_no_only_children_tree=tree_folder+family_name+"tree_collapsed_noonlychild_midpointrooted.nwk"
    if prune:
        pruned_tree=tree_folder+family_name+"tree_collapsed_noonlychild_midpointrooted_prunedsubtree"+str(leaf_number)+".nwk"

    # Getting the unique sequences
    unique_sequences=keep_unique_sequences(full_alignment, unique_alignment)

    # Getting the sequences at distance >= 1 from one another 
    cleaned_msa = remove_distance0_sequences_vectorized(unique_alignment, cleaned_alignment)    
    
    # Build the tree with FastTree
    buildTree(cleaned_alignment, clean_tree)

    # Plot tree info
    print('On the clean tree')
    plot_distance_child_to_parent_tree(clean_tree, num_bins=5000, title='Clean tree - time on the tree between leaf and parent')

    print('On the clean tree, counting gaps')
    plot_hamming_vs_tree_distance(clean_tree, cleaned_alignment, nogaps=False, title='Clean tree pairwise Hamming distance vs. tree distance (w gaps)')

    print('On the clean tree, not counting gaps')
    plot_hamming_vs_tree_distance(clean_tree, cleaned_alignment, nogaps=True, title='Clean tree pairwise Hamming distance vs. tree distance (no gaps)')

    # Midpoint Reroot the clean tree
    midpoint_reroot(clean_tree)
    t=Tree(midpoint_rooted_clean_tree)

    # Plot the compared distance from leaves to root between cleaned tree and midpoint-rerooted cleaned tree
    plot_compared_leaf_to_root_depth(clean_tree, midpoint_rooted_clean_tree, title='Distance from leaf to root after midpoint rerooting', labels=['clean tree', 'clean tree rerooted'])

    # Print the number of problematic internal nodes + the number of problematic leaves
    print('Number of leaf-branches below the threshold', str(length_threshold), 'is', count_short_leaf_branches(clean_tree, epsilon=length_threshold))

    print('Number of strictly internal branches below the threshold', str(length_threshold), 'is', count_short_internal_branches(clean_tree, epsilon=length_threshold))

    # Collapse all branches (internal included) below 'length threshold'
    collapse_short_branches(t, collapsed_tree, length_threshold)

    # Compare branch length repartition on collapsed and non-collapsed trees
    plot_branch_length_distributions(clean_tree, collapsed_tree, yscale_log=True)

    # Collapse nodes that have an only child. Maintain branch length.
    t=Tree(collapsed_tree)
    collapse_only_children(t, collapsed_no_only_children_tree)

    # Get the corresponding alignment
    fetchMSAFromLeafNames(collapsed_no_only_children_tree, cleaned_alignment, save_path=collapsed_alignment)

    # Compare the cleaned alignment and the collapsed one
    collapsed_sequences=read_fasta1(collapsed_alignment)
    compare_msas_pca([cleaned_alignment, collapsed_alignment], ['Cleaned MSA', 'Collapsed MSA'],
                    save_folder=save_folder,
                    file_name=f"{family_name}_cleaned_vs_collapsed_PCA.png")

    # Compare the pairwise Hamming distances
    L=len(unique_sequences[0])
    original_msa=read_fasta1(full_alignment)
    original_dist=get_pairwise_hamming_dist(original_msa, dist_matrix=False)
    clean_dist=get_pairwise_hamming_dist(cleaned_msa, dist_matrix=False)
    collapsed_dist=get_pairwise_hamming_dist(collapsed_sequences, dist_matrix=False)
        # Between the original and the cleaned
    plot_histograms_together([original_dist, clean_dist], 'Comparison of pairwise hamming distances', legend_labels=['original MSA', 'cleaned MSA'], save_path=None, num_bins=L+1)
        # Between the cleaned and the collapsed
    plot_histograms_together([clean_dist, collapsed_dist], 'Comparison of pairwise hamming distances', legend_labels=['cleaned MSA', 'collapsed MSA'], save_path=None, num_bins=L+1)

    if prune:
        prune_subtree(collapsed_no_only_children_tree, leaf_number, pruned_tree)
        pruned_alignment = alignment_folder + family_name + '_collapsed_prunedsubtree' + str(leaf_number) + '.fasta'
        # Plot the resulting PCA
        pruned_sequences=fetchMSAFromLeafNames(pruned_tree, cleaned_alignment, save_path=pruned_alignment)
        compare_msas_pca([cleaned_alignment, pruned_alignment], ['Cleaned MSA', 'MSA from pruned tree'],
                    save_folder=save_folder,
                    file_name=f"{family_name}_cleaned_vs_pruned_PCA.png")

    print('All alignments saved to', alignment_folder)
    print('Final tree is', collapsed_no_only_children_tree)
    print('Final full alignment is', collapsed_alignment)

    print(f'Full alignment contains {len(original_msa)}, its Meff={getMeff(original_msa)}')
    print(f'There are {len(unique_sequences)} unique sequences, their Meff={getMeff(unique_sequences)}')
    print(f'There are {len(cleaned_msa)} sequences at distance>=1 from each other, their Meff={getMeff(cleaned_msa)}')
    print(f'There are {len(collapsed_sequences)} leaves on the final tree, their Meff={getMeff(collapsed_sequences)}')