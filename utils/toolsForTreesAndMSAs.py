import random
import subprocess
import numpy as np
# import jax
# import jax.numpy as jnp
import matplotlib.pyplot as plt
from ete3 import Tree
from utils.utils import get_variable_name, remove_charsequence, createFolder



def convert_array_from_julia(sequence: np.array)-> np.array:
    """
    Converts an array from Julia to Python, shifting the indices of the amino acids accordingly 
    (gap in Julia is 21, and indexing starts at 1).

    Args:
        sequence (np.array): Array of numeral amino acids in Julia format.

    Returns:
        np.array: Array of numeral amino acids in Python format.
    """
    py_seq=[]
    for i in sequence:
        if i==21:
            py_seq.append(0)
        else:
            py_seq.append(i)
    return np.array(py_seq)


amino_acid_mapping = {
    '-': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
    'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,'X': 0,'B' : 0, 'Z': 0
}

def read_fasta1(file_path: str)-> np.ndarray: 
    """Reads from a fasta file and returns the list of sequences.

    Args:
        file_path (str): file containing the sequences in fasta format.

    Returns:
        np.ndarray: M x L array of sequences (MSA) where M is the number of sequences and L is the length of the sequences.
    """

    sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line.startswith('>'):
                sequences.append([amino_acid_mapping[char] for char in line])
    return sequences


def read_fasta2(file_path: str) -> tuple[np.ndarray, dict, list]:
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
                        seq_encoded = [amino_acid_mapping[char] for char in sequence_str if char in amino_acid_mapping]
                        
                        if expected_length is None:
                            expected_length = len(seq_encoded)

                        if len(seq_encoded) != expected_length:
                            print(f"⚠️ Skipping sequence '{current_sequence_name}' - Length {len(seq_encoded)}, expected {expected_length}")
                        else:
                            sequences.append(seq_encoded)
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
                seq_encoded = [amino_acid_mapping[char] for char in sequence_str if char in amino_acid_mapping]

                if expected_length is None:
                    expected_length = len(seq_encoded)

                if len(seq_encoded) != expected_length:
                    print(f"⚠️ Skipping sequence '{current_sequence_name}' - Length {len(seq_encoded)}, expected {expected_length}")
                else:
                    sequences.append(seq_encoded)
                    name_to_index[current_sequence_name] = len(sequences) - 1
                    names.append(current_sequence_name)

            except KeyError as e:
                print(f"❌ Error: Unexpected character '{e}' in sequence '{current_sequence_name}'. Check the FASTA file.")
                pass  # Skip this sequence

    return np.array(sequences, dtype=int), name_to_index, names


def read_fasta3(file_path: str)-> tuple[np.ndarray, dict, dict]:
    """
    Reads from a fasta file and returns the list of sequences, a dictionary mapping each sequence name to its index in the list, and a dictionary mapping each sequence to its name.

    Args:
        file_path (str): file containing the sequences in fasta format.

    Returns:
        tuple[np.ndarray, dict, dict]: M x L array of sequences (MSA) where M is the number of sequences and L is the length of the sequences, a dictionary mapping each sequence name to its index in the MSA array, and a dictionary mapping each sequence to its name.
    """
    sequences = []
    name_to_index={}
    sequence_to_name={}
    index=0
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                current_sequence_name = line[1:]
                name_to_index[current_sequence_name]=index
                index+=1
            else:
                seq=np.array([amino_acid_mapping[char] for char in line])
                sequences.append(seq)
                if tuple(seq) not in sequence_to_name.keys():
                    sequence_to_name[tuple(seq)]=current_sequence_name
    return sequences,name_to_index, sequence_to_name

amino_acid_mapping_noX_noB = {
    '-': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
    'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
}

def amino_acid_seq_to_int(amino_acid_seq: str, mapping_dict: dict = amino_acid_mapping) -> list:
    """
    Converts a string of amino acids to an array of numbers based on the provided mapping.

    Args:
    - amino_acid_seq (str): String of amino acids to be converted.
    - mapping_dict (dict): Dictionary mapping amino acids to numbers.

    Returns:
    - int_list (list): List of numbers representing the amino acids.
    """
    return [mapping_dict[aa] for aa in amino_acid_seq]

reverse_mapping = {v: k for k, v in amino_acid_mapping_noX_noB.items()}

def int_to_amino_acid_seq(int_list: list)-> str:
    """
    Converts an array of numbers to a string of amino acids based on the provided mapping.

    Args:
    - numbers (list): List of numeral amino acids to be converted.
    - mapping_dict(dict): Dictionary mapping numbers to amino acids.

    Returns:
    - amino_acids (str): String of amino acids.
    """
    return ''.join(reverse_mapping[i] for i in int_list)

def write_MSA_to_fasta(cleaned_seqs:np.ndarray, output_file:str, full_alignment_path:str = None, seq_header:str = None)-> None:
    """
    Writes the cleaned sequences extracted from the full alignemnent contained in full_alignment_path to a fasta file.

    Args:
        cleaned_seqs (np.ndarray): Array of cleaned sequences that are included in the full alignment.
        full_alignment_path (str): Path to the full alignment file.
        output_file (str): Path to the output file where the cleaned sequences will be stored, with names matching those of the full_alignment.
    """
    if full_alignment_path:
        seq_to_name=read_fasta3(full_alignment_path)[2]

        with open(output_file, 'w') as f:
            for i, seq in enumerate(cleaned_seqs):
                amino_acid_seq = int_to_amino_acid_seq(seq)
                f.write(f">{seq_to_name[tuple(seq)]}\n")
                f.write(f"{amino_acid_seq}\n")
    elif seq_header:
        with open(output_file, 'w') as f:
            for i, seq in enumerate(cleaned_seqs):
                amino_acid_seq = int_to_amino_acid_seq(seq)
                f.write(f">{seq_header}_seq_{i}\n")
                f.write(f"{amino_acid_seq}\n")
    
    else:
        with open(output_file, 'w') as f:
            for i, seq in enumerate(cleaned_seqs):
                amino_acid_seq = int_to_amino_acid_seq(seq)
                f.write(f">seq_{i}\n")
                f.write(f"{amino_acid_seq}\n")

def getShortAlignment(full_alignment:str, short_alignment:str, seq_nb:int)-> None: 
    """
    Extracts a random sample of sequences from the full alignment and writes them to a new fasta file.

    Args:
        full_alignment (str): Path to the full alignment file.
        short_alignment (str): Path to the output file where the short alignment will be stored.
        seq_nb (int): Number of sequences to be extracted from the full alignment.
    """
    sequences, name_to_index, seq_to_name=read_fasta3(full_alignment)
    sequences=random.shuffle(sequences)[:seq_nb]
    write_MSA_to_fasta(sequences, full_alignment, short_alignment)

def keep_unique_sequences(input_file:str, output_file:str)-> np.ndarray:
    """
    Reads a fasta file, removes duplicate sequences, and writes the unique sequences to a new file.

    Args:
        input_file (str): Path to the input fasta file.
        output_file (str): Path to the output file where the unique sequences will be stored.

    Returns:
        np.ndarray: Array of unique sequences.
    """
    sequences=read_fasta1(input_file)
    unique_sequences=np.unique(sequences, axis=0)
    print('There are',len(unique_sequences), 'unique sequences out of', len(sequences))
    write_MSA_to_fasta(unique_sequences, output_file, input_file, )
    print("Path to alignment without duplicate sequences is", output_file)
    return unique_sequences

# def remove_distance0_sequences(unique_alignment:str, output_file:str)-> tuple[np.ndarray, list]:
#         """
#         Cleans the unique alignment by removing sequences that are at distance 0 from one another.
#         Here, we consider as a measure of distance the Hamming distance between sequences, excluding gaps.
#         Therefore, we use the function calculate_hamming_distance_no_gaps to compute the distance between sequences.

#         Args:
#             unique_alignment (str): A fasta file that contains sequences without duplicates.
#             output_file (str): Path to the output file where the cleaned sequences will be stored.

#         Returns:
#             tuple[np.ndarray, list]: A tuple containing the cleaned sequences and a list of indices of problematic sequences.
#         """
#         def cleaning_no_gaps(unique_alignment: str, output_file: str)-> np.ndarray:
#             """
#             Cleans the unique alignment by removing sequences that are at distance 0 from one another.

#             Args:
#                 unique_alignment (str):  A fasta file that contains sequences without duplicates.
#                 output_file (str): The path to the output file where the cleaned sequences will be stored.

#             Returns:
#                 np.ndarray: Array of cleaned sequences.
#             """
#             # Initialize the cleaned sequences as a set of tuples for... reasons
#             cleaned_seqs=[] 
#             unique_sequences=read_fasta1(unique_alignment)
#             for seq in unique_sequences:
#                 # A marker to indicate whether we need to replace an existing sequence of cleaned_seqs
#                 replace=False 
#                 # A marker to indicate whether or not seq has already been added, modulo the gaps, to cleaned_seqs
#                 new=True 
#                 for unique in cleaned_seqs:
#                     # Check that we haven't already added a cousin (an equivalent sequence in terms of distance) to the clean set
#                     if calculate_hamming_distance_no_gaps(tuple(seq), unique)==0:
#                         # Then seq is not a 'new' one
#                         new=False 
#                         # If we have an equivalent sequence in cleaned_seqs, test to find the better sequence to add to the clean set
#                         # If seq has fewer gaps, remove unique (the one already there) and add best_seq (seq) in its place
#                         if gaps_count(seq)<gaps_count(unique): 
#                                 # The better sequence to add to the clean set
#                                 best_seq=tuple(seq) 
#                                 # The sequence to remove
#                                 rem_seq=unique 
#                                 replace=True
#                         break
#                 # If seq is a new sequence at distance >=1 from the whole clean set, add it to the clean set
#                 if new:
#                     cleaned_seqs.append(tuple(seq))
                
#                 else:
#                     # If there is at least one equivalent sequence in cleaned_seqs, remove all the sequences from clean set that are at distance 0 from seq
#                     if replace:
#                         cleaned_seqs = [tup for tup in cleaned_seqs if calculate_hamming_distance_no_gaps(tup, rem_seq)!=0]
#                         # Add the best sequence best_seq to the clean set
#                         cleaned_seqs.append(best_seq)
            
#             # Write the cleaned sequences to a new fasta file
#             write_MSA_to_fasta(cleaned_seqs, output_file, unique_alignment)

#             return cleaned_seqs

#         # Rechecking the cleaned sequences to ensure that there are no sequences at distance 0 from one another
#         cleaned_seqs=cleaning_no_gaps(unique_alignment, output_file)
#         # problematic_indices=[]
#         # for i in range(len(cleaned_seqs)-1):
#         #     for j in range(i+1, len(cleaned_seqs)):
#         #         if calculate_hamming_distance_no_gaps(cleaned_seqs[i], cleaned_seqs[j])==0:
#         #             print(f'The sequences at indices {i} and {j} are at distance 0 from one another.')
#         #             problematic_indices.append((i,j))

#         print("Path to alignment without distance 0 is", output_file)

#         return cleaned_seqs#, problematic_indices

def remove_distance0_sequences_vectorized(unique_alignment: str, output_file: str) -> np.ndarray:
    """
    Vectorized approach using NumPy for integer-encoded sequences.
    Gaps are encoded as 0.
    """
    unique_sequences = read_fasta1(unique_alignment)
    
    # Convert to NumPy array if not already
    if not isinstance(unique_sequences, np.ndarray):
        seq_array = np.array(unique_sequences)
    else:
        seq_array = unique_sequences
    
    # Create signatures by converting non-gap positions to tuples
    signatures = []
    gap_counts = []
    
    for seq in seq_array:
        # Extract non-gap elements (where value != 0)
        non_gaps = seq[seq != 0]
        # Convert to tuple for hashing
        sig = tuple(non_gaps)
        signatures.append(sig)
        gap_counts.append(np.sum(seq == 0))
    
    # Use pandas for efficient grouping
    import pandas as pd
    df = pd.DataFrame({
        'signature': signatures,
        'gap_count': gap_counts,
        'seq_idx': np.arange(len(seq_array))
    })
    
    # Keep row with minimum gap_count for each signature
    best_indices = df.loc[df.groupby('signature')['gap_count'].idxmin(), 'seq_idx'].values
    
    cleaned_seqs = [tuple(seq_array[i]) for i in best_indices]
    
    write_MSA_to_fasta(cleaned_seqs, output_file, unique_alignment)
    print("Path to alignment without distance 0 is", output_file)
    
    return np.array(cleaned_seqs)

def buildTree(alignment: str, tree_name: str)-> str:
        """
        Builds a phylogenetic tree from a multiple sequence alignment using FastTree, using the Whelan-Goldman model, and adding a pseudo count.
        By default, FastTree accounts for variable rates of evolution across sites by assigning each site to one of 20 categories, with the rates geometrically spaced from 0.05 to 20. 
        FastTree sets each site to its most likely category by using a Bayesian approach with a gamma prior. 
        This prevents overfitting on small alignments.
        Also uses the gamma option that rescales the branch lengths and computes a Gamma20-based likelihood.

        Args:
            alignment (str): Path to the multiple sequence alignment file.
            tree_name (str): Name of the output tree file (use a .nwk extension for better readability).

        Returns:
            str: Name of the output tree file.
        """
        fasttree_command ="fasttree -gamma -wag -pseudo {} > {}".format(alignment, tree_name)
    
        # Execute the FastTree command
        process = subprocess.Popen(fasttree_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for the process to finish and capture the output
        stdout, stderr = process.communicate()

        # Decode the output from bytes to string
        stdout_str = stdout.decode()
        stderr_str = stderr.decode()

        # Print the output
        print("Standard Output:")
        print(stdout_str)
        print("Standard Error:")
        print(stderr_str)

        # Check if any error occurred during execution
        if process.returncode != 0:
            print("FastTree command failed with error code", process.returncode)
        
        return tree_name

def remove_gapped_sequences(alignment_file: str, percentage=0.2)-> tuple[np.ndarray, str]:
    """
    Removes sequences from an MSA where the percentage of gaps exceeds a given threshold.

    Args:
        alignment_file (str): Path to the alignment file in fasta format.
        percentage (float, optional): The threshold for the admissible gap percentage in a sequence. Defaults to 0.2.

    Returns:
        tuple[np.ndarray, str]: A tuple containing the cleaned sequences and the path to the output file where the cleaned sequences are stored.
    """
    # Read the alignment file
    MSA=read_fasta1(alignment_file)

    # The length of the sequences
    L=len(MSA[0])

    # The number of admissible gaps in a sequence
    cutoff=L*percentage

    few_gaps=[]
    for sequence in MSA:
        if gaps_count(sequence)<cutoff:
            few_gaps.append(sequence)

    # Write the cleaned sequences to a new fasta file
    alignment_few_gaps=remove_charsequence(alignment_file, '.fasta')+'_'+str(percentage)+'gaps.fasta'

    write_MSA_to_fasta(few_gaps, alignment_file, alignment_few_gaps)

    print("Cleaned alignment with fewer gaps has been stored in", alignment_few_gaps)

    return few_gaps, alignment_few_gaps

def count_short_leaf_branches(tree_path: str, epsilon=10**(-6))-> int:
    """
    Counts the number of short branches (shorter than epsilon) leading to leaves in a phylogenetic tree.

    Args:
        tree_path (str): Path to the tree file.
        epsilon (float, optional): The critical branch length to account for. Defaults to 10**(-6).

    Returns:
        int: The number of short (shorter than epsilon) branches leading to leaves.
    """
    tree=Tree(tree_path)
    counter=0
    for node in tree.iter_leaves():
        if not node.is_root():
            if node.dist<epsilon:
                #prob=node
                #prob_name=node.name
                #print(node.dist)
                counter+=1
    return counter

def count_short_internal_branches(tree_path: str, epsilon: float = 10**(-6))-> int:
    """
    Counts the number of short branches (shorter than epsilon) leading to internal nodes in a phylogenetic tree.

    Args:
        tree_path (str): _description_
        epsilon (float, optional): _description_. Defaults to 10**(-6).

    Returns:
        int: _description_
    """
    tree=Tree(tree_path)
    counter=0
    for node in tree.traverse():
        if not node.is_root() and not node.is_leaf():
            if node.dist<epsilon:
                #prob=node
                #prob_name=node.name
                #print(node.dist)
                counter+=1
    return counter

def prune_only_close_leaves(tree, cleaned_tree_name, threshold = 10**(-6)):
    sis_nb=[]
    for leaf in tree.iter_leaves():
        # Check if the leaf's branch length is below the threshold
        if leaf.dist < threshold:
            parent = leaf.up
            parent.name=leaf.name
            # Identify the sister leaf (if exists)
            sisters = [s for s in parent.get_descendants()]#get_children() if s != leaf]
            sis_nb.append(len(sisters))
            leaf.detach()
    
    tree.write(outfile=cleaned_tree_name, format=1)
    return cleaned_tree_name, sis_nb

def binaryTree(path_to_tree, binary_tree_name):
    t=Tree(path_to_tree)
    # Enforce binary branching on the tree
    t.resolve_polytomy()
    # Save the resulting binary tree to a Newick file
    t.write(outfile=binary_tree_name, format=1)
    print('Saved binary tree file to', binary_tree_name)
    return binary_tree_name

def prune_subtree(path_to_tree: str, leaves_number: int, save_path='')-> Tree:
    """Prunes a subtree of approximately 'leaves_number' leaves. 
    Saves the pruned tree in save_path if specified.
    Returns the pruned tree. 

    Args:
        tree (Tree): _description_
        leaves_number (int): _description_
        save_path (str, optional): _description_. Defaults to ''.

    Returns:
        Tree: The pruned tree of the appropriate size.

    """
    try:
        tree=Tree(path_to_tree, format=0)
    except Exception as e:
        print("Error reading tree as format 0 ")
        print("Trying to read tree as format 1")
        try:
            tree = Tree(path_to_tree, format=1)
        except Exception as e:
            print("Error reading tree as format 1 ", e)
            return None

    L=[]
    interval=0
    while len(L)==0:
      L=[node for node in tree.traverse() if leaves_number+interval>=len(node.get_leaves()) >= leaves_number]
      interval+=5
      #print(interval)
    node = random.choice(L)

    # Extract the subtree rooted at the selected node
    subtree = node.detach()

    if save_path:
       subtree.write(outfile=save_path, format=1)
    print('number of leaves', len(subtree.get_leaves()))
    return subtree

def markChosenNode(t: Tree, chosen_index:int, node_name='chosen_node'):
    index=0
    worked=False
    #find it in the tree and mark it as chosen_node
    for node in t.traverse("levelorder"):
        if not node.is_leaf():
            node.name=str(index)
            if index==chosen_index:
                node.name=node_name
                worked=True
            index+=1
    if not worked:
        print('Pick a lower index - cannot reroot from a leaf.')
        return 
    return t

def reroot(path_to_tree, chosen_index):

    t=Tree(path_to_tree)
    node_name='chosen_node'
    worked=False
    index=0
    for node in t.traverse("levelorder"):
        if not node.is_leaf():
            node.name=str(index)
            if index==chosen_index:
                node.name=node_name
                worked=True
            index+=1

    if not worked:
        print('Pick a lower index - cannot reroot from a leaf.')
        return 

    # 1. Find the chosen node
    chosen_node=t.search_nodes(name=node_name)[0]

    # 2. Try to set it as root
    #t.set_outgroup(t&node_name)
    t.set_outgroup(chosen_node)

    # 3. But it has a parent (not more, i hope)
    almighty_root=chosen_node.up

    # 4. Get the time to the parent
    time=chosen_node.get_distance(almighty_root)

    #print('time', time)

    # 5. Set the almighty root as a child of the chosen node
    chosen_node.detach()
    #print('chosen tree', chosen_tree.get_ascii(show_internal=True))

    #print('Rest of the tree', t.get_ascii(show_internal=True))

    chosen_node.add_child(t, dist=time)

    # 6. Print everything
    print('Final tree', chosen_node.get_ascii(show_internal=True))

    # 7. Write the new tree to a path 
    path_to_new_tree=path_to_tree+'_chosen_index_'+str(chosen_index)
    chosen_node.write(outfile=path_to_new_tree, format=1)

    return path_to_new_tree

def fetchMSAFromLeafNames(tree_path, full_MSA_path, save_path=''):
  tree=Tree(tree_path)
  fullMSA, name_to_index, _ = read_fasta2(full_MSA_path)
  MSA=[]
  for leaf in tree.get_leaves():
    MSA.append(fullMSA[name_to_index[leaf.name]])
  MSA=np.array(MSA)
  if save_path:
    write_MSA_to_fasta(MSA, save_path, full_MSA_path)
    print("MSA from the leaves has been saved in", save_path)
  return MSA

def MSAFromLeafSequences(t: Tree, save_path='')-> np.array: 
    """Returns an MSA from the leaves of the tree. 
    Stores it in a fasta file in save_path, if specified.

    Args:
        t (Tree): an ete3 Tree with sequences on the leaves.
        save_path (str, optional): path to save the MSA in a fasta format. Defaults to ''.

    Returns:
        np.array[np.array]: an MSA derived from the leaves of the tree t.
    """
    if save_path:
       with open(save_path, 'w') as fasta_file:
            for node in t.iter_leaves():
                sequence=int_to_amino_acid_seq(node.sequence)
                fasta_file.write(f'>{node.name}\n{sequence}\n')
    L=[]
    for node in t.iter_leaves():
        L.append(node.sequence)
    return L 

def gaps_count(seq):
    return sum(c==0 for c in seq)

def calculate_hamming_distance(seq1, seq2):
    """Calculate the Hamming distance between two sequences."""
    #return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
    seq1=np.array(list(seq1))
    seq2=np.array(list(seq2))

    return np.sum(seq1 != seq2)

def calculate_hamming_distance_no_gaps(seq1, seq2):
    """Calculate the Hamming distance between two sequences."""
    return sum((c1 != c2 and c1!=0 and c2!=0)for c1, c2 in zip(seq1, seq2))

def normalized_hamming_no_gaps(seq1,seq2):
  norm=0
  hamming_dist=0
  for c1, c2 in zip(seq1, seq2):
    if (c1!=0 and c2!=0):
      norm+=1
      if c1 != c2:
        hamming_dist+=1
  return hamming_dist/norm

def average_distance_from_leaves_to_root(t):
  #gets the average time between all the leaves and the root
  L=[] #length nb of leaves
  root=t.get_tree_root()
  for leaf in t:
    if leaf.is_leaf():
        dist=t.get_distance(root, leaf, topology_only=False)
        L.append(dist)
  L=np.array(L)
  avg_dist=L.mean()
  return avg_dist

def plot_hamming_vs_tree_distance(path_to_tree, path_to_MSA, nogaps=False, title='Hamming distance vs. tree distance for sibling pairs'):

    def treeAndHammingDistOnTree(path_to_tree, path_to_MSA, nogaps=False):
        tree=Tree(path_to_tree)
        counter=0
        sequences, name_to_index, _ = read_fasta2(path_to_MSA)
        hamming_tree_dists_nat=[]
        for leaf in tree.iter_leaves():
            # Get the sibling of the current leaf
            for sibling in leaf.get_sisters():
                if sibling.is_leaf():
                    # Calculate the distance between the current leaf and its sibling
                    distance_to_sibling = tree.get_distance(leaf, sibling)
                    #print(distance_to_sibling)
                    if distance_to_sibling>0:
                        if sibling.name:
                            seq1=sequences[name_to_index[leaf.name]]
                            seq2=sequences[name_to_index[sibling.name]]
                            if nogaps:
                                hamming_tree_dists_nat.append((calculate_hamming_distance_no_gaps(seq1, seq2),distance_to_sibling))
                            else: 
                                hamming_tree_dists_nat.append((calculate_hamming_distance(seq1, seq2),distance_to_sibling))
                            #print('Appended dist')
                            counter+=1
                        else:
                            print('no name sibling for leaf', leaf.name, 'distance to sibling:',distance_to_sibling)
        print(counter)
        return hamming_tree_dists_nat

    distances=treeAndHammingDistOnTree(path_to_tree, path_to_MSA, nogaps)

    def plot_hamming_and_tree_distances(distances):
        hamming_distances, tree_distances = zip(*distances)
        plt.scatter(tree_distances, hamming_distances)
        plt.xlabel('Distance on the tree')
        plt.ylabel('Hamming distance')
        plt.title(title)
        plt.show()
    
    plot_hamming_and_tree_distances(distances)

def plot_distance_child_to_parent_tree(path_to_tree, num_bins=5000, title='Time on the tree between the leaves and their parent'):
    def distance_child_to_parent_tree(path_to_tree):
        t=Tree(path_to_tree)
        time_to_parent=[]
        index=0
        for leaf in t.iter_leaves():
            parent = leaf.up
            distance_to_ancestor = leaf.get_distance(parent)
            if distance_to_ancestor<0.01:
                index+=1
            time_to_parent.append(distance_to_ancestor)
        return time_to_parent
   
    time_to_parent=distance_child_to_parent_tree(path_to_tree)

    def plot_time_distibution(time_to_parent, num_bins):
        bin_edges = [i * (max(time_to_parent) - min(time_to_parent)) / num_bins + min(time_to_parent) for i in range(num_bins + 1)]

        # Create the histogram
        plt.hist(time_to_parent, bins=bin_edges, edgecolor='black')

        # Add labels and title
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.title(title)

        # Show the plot
        plt.show()
    
    plot_time_distibution(time_to_parent, num_bins)
    return time_to_parent

def plot_branch_length_distribution(path_to_tree):
    t=Tree(path_to_tree)
    branch_lengths=[]
    for node in t.traverse('levelorder'):
        branch_lengths.append(node.dist)

    #plot
    bins = np.logspace(-10, 1, 50)  # 50 bins between 10^-10 and 10^1

    plt.hist(branch_lengths, bins)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-10, 1e1)

    plt.xlabel('Branch length')
    plt.ylabel('Count')
    plt.title('Repartition of branch lengths over the original tree')
    plt.show()
    return branch_lengths


def plot_branch_length_distributions(path_to_full_tree, path_to_collapsed_tree, yscale_log=False, format = 0):
    t_ori=Tree(path_to_full_tree)
    collapsed_tree=Tree(path_to_collapsed_tree, format = format)
    branch_lengths_ori=[]
    branch_lengths_cleaned=[]
    for node in collapsed_tree.traverse('levelorder'):
        branch_lengths_cleaned.append(node.dist)

    for node in t_ori.traverse('levelorder'):
        branch_lengths_ori.append(node.dist)

    bins = np.logspace(-10, 1, 50)  # 50 bins between 10^-10 and 10^1

    plt.hist(branch_lengths_ori, bins, alpha=0.5, label='Original tree branch lengths')

    plt.hist(branch_lengths_cleaned, bins, alpha=0.5, label='Collapsed tree branch lengths')

    # Set logarithmic scales for both axes
    plt.xscale('log')
    if yscale_log:
        plt.yscale('log')

    # Set axis limits
    plt.xlim(1e-10, 1e1)

    # Add labels, title, and legend
    plt.xlabel('Branch length')
    plt.ylabel('Count')
    plt.title('Repartition of branch lengths (Collapsed vs. Original)')
    plt.legend()

    # Show the plot
    plt.show()

    return branch_lengths_ori, branch_lengths_cleaned

# def collapse_short_branches(tree, outfile, length_threshold=10**(-6)):
#     """
#     Collapse branches shorter than a given threshold while maintaining dependent nodes.
    
#     :param tree: An ETE3 Tree object.
#     :param length_threshold: Branch length below which the branch is collapsed.
#     """
#     for node in tree.traverse("postorder"):  # Traverse the tree in postorder to process children first
#         if not node.is_root():  # Skip the root node
#             if node.dist < length_threshold:  # Check if branch length is below the threshold
#                 parent = node.up
#                 # Add the node's children to its parent
#                 for child in node.children:
#                     parent.add_child(child)
#                 # Remove the current node from its parent
#                 parent.remove_child(node)
#     tree.write(outfile=outfile, format=1)
#     return tree, outfile



# def collapse_only_children(tree, outfile):
#     """
#     Collapse branches leading to only one child. Sum the branch lengths.
    
#     :param tree: An ETE3 Tree object.
#     :param length_threshold: Branch length below which the branch is collapsed.
#     """
#     counter=0
#     for node in tree.traverse("postorder"):  # Traverse the tree in postorder to process children first
#         if not node.is_root():  # Skip the root node
#             if len(node.children)==1:  # Check if it has an only child
#                 #print('WARNING')
#                 counter+=1
#                 parent = node.up
#                 # Add the node's children to its parent
#                 for child in node.children:
#                     dist=node.dist+child.dist
#                     parent.add_child(child)
#                     child.dist=dist
#                 # Remove the current node from its parent
#                 parent.remove_child(node)
#     print('Collapsed', counter, 'nodes with only children')
#     tree.write(outfile=outfile, format=1)

#     return tree

def collapse_short_branches(tree, outfile, length_threshold=1e-6):
    """
    Collapse branches shorter than a given threshold while maintaining topology.
    Preserves leaf names by keeping children and updating parents if needed.
    """
    to_collapse = [
        node for node in tree.traverse("postorder")
        if (not node.is_root() and node.dist < length_threshold)
    ]
    
    print(f"Collapsing {len(to_collapse)} short branches")
    
    for node in to_collapse:
        parent = node.up
        if parent is None:
            continue
        
        # Propagate name if parent has no name
        if not parent.name:
            parent.name = node.name if node.name else None
        
        # Reattach children to parent
        for child in list(node.children):
            if child.dist is None:
                child.dist = 0.0
            if node.dist is None:
                node.dist = 0.0
            child.dist += node.dist
            node.remove_child(child)
            parent.add_child(child)
        
        # Remove the node from its parent
        parent.remove_child(node)
    
    tree.dist = 0.0
    tree.write(outfile=outfile, format=5)
    Tree(outfile, format=5)  # sanity check
    
    return tree, outfile


def collapse_only_children(tree, outfile):
    """
    Collapse internal nodes that have only one child.
    Parent is removed, child is promoted, and branch lengths are summed.
    Propagates names to ensure no leaf or internal node is unnamed.
    """
    counter = 0
    to_collapse = [
        node for node in tree.traverse("postorder")
        if (not node.is_root() and len(node.children) == 1)
    ]
    
    for node in to_collapse:
        counter += 1
        parent = node.up
        child = node.children[0]
        
        # Sum branch lengths
        if child.dist is None:
            child.dist = 0.0
        if node.dist is None:
            node.dist = 0.0
        child.dist += node.dist
        
        # Propagate name if parent has no name
        if not parent.name:
            parent.name = node.name if node.name else child.name
        
        # Remove node and attach child to grandparent
        node.remove_child(child)
        parent.add_child(child)
        parent.remove_child(node)
    
    # Promote child if root has only one child
    while not tree.is_leaf() and len(tree.children) == 1:
        child = tree.children[0]
        tree.children = child.children
        tree.name = child.name
        tree.dist = 0.0
        for c in tree.children:
            c.up = tree
        counter += 1
    
    tree.dist = 0.0
    print(f'Collapsed {counter} nodes with only children')
    
    tree.write(outfile=outfile, format=5)
    Tree(outfile, format=5)  # sanity check
    
    return tree



def midpoint_reroot(path_to_tree, format = 0):
    t=Tree(path_to_tree, format = format)
    R = t.get_midpoint_outgroup()
    # and set it as tree outgroup
    t.set_outgroup(R)
    midpoint_rooted_tree=remove_charsequence(path_to_tree, ".nwk")+"_midpointrooted.nwk"
    t.write(outfile=midpoint_rooted_tree, format=1)
    return midpoint_rooted_tree

def plot_topological_depth(path_to_tree, title=''):
    t=Tree(path_to_tree)
    depths=[]
    node_dists=[]

    root=t.get_tree_root()

    for node in t.traverse():
        depth= root.get_distance(node, topology_only=True)
        depths.append(depth)
        node_dists.append(node.dist)

    plt.bar(depths, node_dists)
    plt.xlabel('Depth of the node (topological distance to the root)')
    plt.ylabel('Distance to parent')
    plt.title(title)
    plt.show()
    return depths

def plot_leaf_to_root_depth(path_to_tree, title=''):
    t=Tree(path_to_tree)
    depths=[]

    root=t.get_tree_root()

    for node in t.traverse():
        depth= root.get_distance(node, topology_only=False)
        depths.append(depth)
   
    plt.hist(depths, bins=100, color='blue', edgecolor='black')
    plt.xlabel('Distance to the root')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()
    return depths

def plot_compared_leaf_to_root_depth(path_to_tree1, path_to_tree2, title='', labels=['tree 1', 'tree2']):
    t1=Tree(path_to_tree1)
    t2=Tree(path_to_tree2)
    depths1=[]
    depths2=[]

    root1=t1.get_tree_root()
    root2=t2.get_tree_root()

    for node in t1.traverse():
        depth= root1.get_distance(node, topology_only=False)
        depths1.append(depth)
    for node in t2.traverse():
        depth= root2.get_distance(node, topology_only=False)
        depths2.append(depth)

    plt.hist(depths1, bins=100, color='blue', edgecolor='black', alpha=0.5, label=labels[0])
    plt.hist(depths2, bins=100, color='orange', edgecolor='black', alpha=0.5, label=labels[1])

    plt.xlabel('Distance to the root')
    plt.ylabel('Count')
    plt.legend()
    plt.title(title)
    plt.show()
    return depths1, depths2

def plot_compared_topological_depth(path_to_full_tree, path_to_collapsed_tree):
    depths_full=plot_topological_depth(path_to_full_tree, 'Raw tree')
    depths_collapsed=plot_topological_depth(path_to_collapsed_tree, 'Collapsed tree')
    plt.hist(depths_full, label='raw tree')
    plt.hist(depths_collapsed, label='cleaned tree without removing siblings')
    plt.xlabel('Depth of the node (topological distance to the root)')
    plt.ylabel('Count')
    plt.legend()
    plt.show()


def create_MSA_profile(prob_non_normalized: np.array, cardinal = 100):
  
  """Input: a profile giving the probability distribution of each amino acid for each site. 
            cardinal is the number of sequences wanted in the MSA.

  Output: an MSA according to this distribution.

  Example usage:
  create_MSA_profile(np.array([[0.25]*21,[1]*21,[2]*21]).reshape(3,21))"""

  MSA = np.zeros((cardinal,prob_non_normalized.shape[0]), dtype=int)
  for k in range(MSA.shape[1]):
    MSA[:,k] = np.random.choice(np.arange(21), size = cardinal, p = prob_non_normalized[k]/ prob_non_normalized[k].sum())
  return MSA

import torch
import torch.nn.functional as F

def oneHotEncode(MSA):
    """One-hot encodes an MSA with 21 possible amino acids, returns integers."""
    MSA_numpy = np.array(MSA, dtype=np.int64)
    MSA_tensor = torch.tensor(MSA_numpy, dtype=torch.long) # (M, L)
    return F.one_hot(MSA_tensor, num_classes=21).to(torch.int) # (M, L, 21)


# def get_pairwise_hamming_dist(msa: torch.Tensor, dist_matrix=False):
#     """
#     Returns the distance matrix of an MSA if dist_matrix is True.
#     Returns the distance distribution otherwise.
#     All distances and histograms are integers.
#     """
#     msa_onehot = oneHotEncode(msa)
#     msa_arg = msa_onehot.argmax(dim=-1)  # shape [N,L], dtype=int
#     N, L = msa_arg.shape

#     if dist_matrix:
#         # Compute pairwise Hamming distance matrix
#         dist = (msa_arg[:, None, :] != msa_arg[None, :, :]).sum(dim=-1)  # [N,N] integer
#         # Flatten to 1D for histogram
#         dist_flat = dist.flatten()
#         # bincount requires non-negative integers
#         hist = torch.bincount(dist_flat, minlength=L+1)
#         return hist
#     else:
#         # Compute all pairwise distances
#         dist = (msa_arg[:, None, :] != msa_arg[None, :, :]).sum(dim=-1)  # [N,N] integer
#         # Exclude diagonal (distance of sequence to itself)
#         mask = 1 - torch.eye(N, dtype=torch.int, device=msa_arg.device)
#         dist_values = dist[mask.bool()]
#         hist = torch.bincount(dist_values, minlength=L+1)
#         return hist


def get_pairwise_hamming_dist(msa: torch.Tensor, dist_matrix=False):
    """
    Returns the distance histogram of an MSA without storing full distance matrix.
    PyTorch implementation.
    
    Args:
        msa: Input MSA as torch.Tensor
        dist_matrix: If True, returns histogram of full distance matrix (memory intensive)
    
    Returns:
        Histogram of pairwise Hamming distances
    """
    # Convert to torch tensor if not already
    if not isinstance(msa, torch.Tensor):
        msa = torch.tensor(msa)
    msa_onehot = oneHotEncode(msa)
    msa_arg = msa_onehot.argmax(dim=-1)  # [N, L]
    
    N, L = msa_arg.shape
    
    if dist_matrix:
        # Full distance matrix (exclude diagonal/self-comparison)
        # Compute all pairwise distances
        msa_ident = (1 - torch.tensordot(msa_onehot, msa_onehot, dims=([1, 2], [1, 2])) / L) * L
        # Exclude diagonal
        mask = ~torch.eye(N, dtype=torch.bool)
        distances = msa_ident[mask]
        hist = torch.histogram(distances, bins=L, range=(0, L))[0]
        return hist
    else:
        # Memory-efficient: compute distances row-by-row excluding self
        all_hists = []
        for i in range(N):
            # Exclude self-comparison
            mask = torch.ones(N, dtype=torch.bool)
            mask[i] = False
            dist_ = (1 - (msa_arg[i] == msa_arg[mask]).float().mean(dim=-1)) * L
            hist = torch.histc(dist_, bins=L+1, min=0, max=L)
            all_hists.append(hist)
        
        total_hist = torch.stack(all_hists).sum(dim=0)
        return total_hist

def get_eff(msa, eff_cutoff=0.8):
    """Returns the weighted frequencies from an MSA."""
    # Check that the msa is a numpy array
    if not isinstance(msa, np.ndarray):
        #raise TypeError("Input MSA must be a 2D numpy array")
        # Convert to 2D numpy array
        msa = np.array(msa, dtype=np.int64)

    msa_onehot = oneHotEncode(msa)
    n_seq = msa_onehot.shape[0]

    if n_seq > 10000:
        msa_arg = msa_onehot.argmax(dim=-1)

        def get_w(seq):
            mask = (seq == msa_arg).float().mean(dim=-1) > eff_cutoff
            return 1.0 / mask.sum()

        weights = torch.tensor([get_w(seq) for seq in msa_arg])
        return weights
    else:
        # Compute pairwise sequence identities
        msa_flat = msa_onehot.view(n_seq, -1)
        identity = torch.matmul(msa_flat, msa_flat.T) / msa.shape[1]
        weights = 1.0 / (identity >= eff_cutoff).sum(dim=-1)
        return weights

def getMeff(msa, eff_cutoff=0.8):
    """Returns the effective number of unique sequences/non-redundant information in an MSA."""
    return get_eff(msa, eff_cutoff).sum()


def get_stats_MSA(MSA: torch.Tensor):
    """
    From an MSA, returns single-point frequencies, double-point frequencies, covariance matrix.
    """
    MSA_onehot = oneHotEncode(MSA)
    n_seq = MSA_onehot.shape[0]

    f_i = MSA_onehot.mean(dim=0)  # single-site frequencies
    f_ij = torch.tensordot(MSA_onehot, MSA_onehot, dims=([0],[0])) / n_seq  # double-site frequencies

    # Compute covariance
    # f_i has shape [L, 21], f_ij has shape [L,21,L,21]
    f_i_expand1 = f_i[:, :, None, None]  # [L,21,1,1]
    f_i_expand2 = f_i[None, None, :, :]  # [1,1,L,21]
    c_ij = f_ij - f_i_expand1 * f_i_expand2

    return f_i, f_ij, c_ij


# def oneHotEncode(MSA):
#   """One-hot encodes an MSA."""
#   return jax.nn.one_hot(jnp.array(MSA),21)

# def get_pairwise_hamming_dist(msa: np.array, dist_matrix=False):
#     """Returns the distance matrix of an msa if dist_matrix is set to true.
#     Returns the distance distribution instead if not."""
#     if dist_matrix:
#         msa_ident = (1-jnp.tensordot(msa,msa,[[1,2],[1,2]])/msa.shape[1])*msa.shape[1]#distance matrix
#         return jnp.histogram(msa_ident, bins=msa.shape[1],range = (0,msa.shape[1]))
#     else:
#         msa=oneHotEncode(msa)
#         msa = msa.argmax(-1)
#         def get_w(seq): 
#             dist_ = (1-(seq==msa).mean(-1))*msa.shape[1]
#             return jnp.histogram(dist_,bins=msa.shape[1]+1,range = (0,msa.shape[1]))[0]
#         return (jax.lax.scan(lambda _,x:(_,get_w(x)),None,msa,unroll=2)[1]).sum(axis = 0)

# def get_eff(msa, eff_cutoff=0.8): 
#   """Returns the weighted frequencies from an MSA (Jeanne's function)."""
#   print('Warning: these weights DO NOT sum to 1.')
#   msa=oneHotEncode(msa)
#   if msa.shape[0] > 10000:
#     msa = msa.argmax(-1)
#     def get_w(seq): return 1/((seq==msa).mean(-1) > eff_cutoff).sum()
#     return jax.lax.scan(lambda _,x:(_,get_w(x)),None,msa,unroll=2)[1]
#   else:
#     msa_ident = jnp.tensordot(msa,msa,[[1,2],[1,2]])/msa.shape[1]
#     return 1/(msa_ident >= eff_cutoff).sum(-1)
  
  
# def getMeff(msa, eff_cutoff=0.8):
#    """Returns the effective number of unique sequences/non repeated information in a given MSA."""
#    return get_eff(msa, eff_cutoff).sum()

# def get_stats_MSA(MSA: np.array) -> tuple[np.array, np.array, np.array]: 
#   """
#   From an MSA, returns single point frequencies, double point frequencies, covariance matrix.
#   """
#   MSA=oneHotEncode(MSA)
#   n = None
#   f_i = jnp.mean(MSA,0)
#   #print(f_i)
#   f_ij = jnp.tensordot(MSA,MSA,[0,0])/MSA.shape[0]
#   c_ij = f_ij - f_i[:,:,n,n] * f_i[n,n,:,:]
#   return f_i, f_ij, c_ij 

def plotStats(MSAs: list, names: np.array, save_path='')->None:
    """Plots the stats of all couples in a list of MSAs. 
    names is the list of the names of the MSAs for the captions and the titles."""

    #getting the stats of the MSAs
    stats=[get_stats_MSA(MSAs[i]) for i in range(len(MSAs))]

    #Plotting the stats
    for i in range(len(stats)-1):
      
      for j in range(i+1):
        #plotting the covariance of each couple of MSAs    
        plt.scatter(stats[i][2].ravel(),stats[j][2].ravel())
        plt.xlabel("CIJ "+names[i])
        plt.ylabel("CIJ "+names[j])
        plt.title('Covariance of the '+names[i]+" vs covariance of the "+names[j])
        if save_path:
            plt.savefig(save_path+'Covariance of the '+names[i]+" vs covariance of the "+names[j])
        plt.show()

        plt.scatter(stats[i][0].ravel(),stats[j][0].ravel())
        plt.xlabel("FI "+names[i])
        plt.ylabel("FI "+names[j])
        plt.title('Single point frequencies of the '+names[i]+" vs "+names[j])
        if save_path:
          plt.savefig(save_path+'Single point frequencies of the '+names[i]+" vs "+names[j])
        plt.show()

def plotWeights(weights, title, save_path=''):
    """_summary_

    Args:
        weights (_type_): _description_
        title (_type_): _description_
        save_path (str, optional): _description_. Defaults to ''.
    """
    M=len(weights)
    plt.figure(figsize=(40, 6))
    plt.subplot(1,2,1)
    plt.plot(weights,'o')
    plt.axhline(y = 1/M, color = 'r', linestyle = '-')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def getFrequencyDistribution(MSA):
  """Returns the frequency distribution of a given MSA. One point frequency of each amino acid per each site."""
  return np.mean(np.array(oneHotEncode(MSA)),axis = 0).astype('float64')

def normalizePerRow(w):
  """
  Normalizes an array per row.
  """
  row_sums = np.sum(w, axis=1, keepdims=True)
  w_normalized = w / row_sums
  return w_normalized

# def getReweightedDistribution(MSA,save_path=''):
#     """
#     Getting the weighted frequencies of a given MSA, normalized per row.
#     """
#     # Converting MSA to numpy array if needed
#     if not isinstance(MSA, np.ndarray):
#         MSA = np.array(MSA, dtype=np.int64)
    
#     #L=len(MSA[0]);    
#     L=MSA.shape[1]
#     weights=get_eff(MSA)

#     msa=reshape_array(oneHotEncode(MSA));M=msa.shape[0]

#     w_true_reweighted=(np.sum((msa.T*weights).T,0)/M).reshape(L, 21)

#     w_true_reweighted=normalizePerRow(w_true_reweighted)
    
#     #saving it
#     if save_path:
#         fname=save_path;
#         np.savetxt(fname, np.array(w_true_reweighted), fmt='%.18e', delimiter=' ', newline='\n', comments='# ')
    
#     return w_true_reweighted

def getReweightedDistribution(MSA, save_path=''):
    """
    Getting the weighted frequencies of a given MSA, normalized per row.
    Safely handles both numpy arrays and pytorch tensors.
    """
    import torch
    import numpy as np
    
    # Converting MSA to numpy array if needed
    if isinstance(MSA, torch.Tensor):
        MSA = MSA.detach().cpu().numpy()
    if not isinstance(MSA, np.ndarray):
        MSA = np.array(MSA, dtype=np.int64)
    
    L = MSA.shape[1]
    weights = get_eff(MSA)
    
    # Convert weights to numpy if it's a tensor
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    
    msa = reshape_array(oneHotEncode(MSA))
    M = msa.shape[0]
    
    # Convert msa to numpy if it's a tensor
    if isinstance(msa, torch.Tensor):
        msa = msa.detach().cpu().numpy()
    
    # Now all operations are guaranteed to be numpy
    w_true_reweighted = (np.sum((msa.T * weights).T, 0) / M).reshape(L, 21)
    
    w_true_reweighted = normalizePerRow(w_true_reweighted)
    
    # Convert back to numpy if normalizePerRow returned a tensor
    if isinstance(w_true_reweighted, torch.Tensor):
        w_true_reweighted = w_true_reweighted.detach().cpu().numpy()
    
    # Saving it
    if save_path:
        fname = save_path
        np.savetxt(fname, np.array(w_true_reweighted), fmt='%.18e', delimiter=' ', newline='\n', comments='# ')
    
    return w_true_reweighted

def randomW(L,q):
  random_array = np.random.rand(L, q)

  # Normalize the array along the lines (rows) so that the sum over each line is 1
  normalized_array = random_array / random_array.sum(axis=1, keepdims=True)

  return normalized_array

def get_maximum_likelihood_sequence(probabilities):
    """
    Compute the maximum likelihood sequence from amino acid probabilities.

    Parameters:
        probabilities (list or np.ndarray): A 2D array where each row contains
                                            probabilities of 21 amino acids at a site.

    Returns:
        list: The maximum likelihood sequence represented as indices (0 to 20).
    """
    if not isinstance(probabilities, np.ndarray):
        probabilities = np.array(probabilities)
    
    # Check the input dimensions
    if probabilities.ndim != 2 or probabilities.shape[1] != 21:
        raise ValueError("Input must be a 2D array with 21 columns (one for each amino acid).")
    
    # Compute the maximum likelihood sequence
    max_likelihood_indices = np.argmax(probabilities, axis=1)
    return max_likelihood_indices.tolist()



def plotHistogramSimilarityScore(MSA: np.array)-> None:

    """Computes similarity score between couples of Gibbs sampled sequences"""

    gibbs_hist = []
    num_sequences=len(MSA)
    ohe_MSA=oneHotEncode(MSA)
    for k in range(num_sequences):
        for j in range(k + 1, num_sequences):  # only iterate over each unique pair once
            similarity = np.abs(ohe_MSA[k] * ohe_MSA[j]).sum()
            gibbs_hist.append(similarity)

    plt.hist(np.array(gibbs_hist))
    plt.title('Similarity scores for '+get_variable_name(MSA))
    plt.show()

def plot_histograms_together(hist_list, title, legend_labels=None, save_path=None, num_bins=54): 
    num_plots = len(hist_list)
    bar_width = 0.8 / num_plots
    index = np.arange(num_bins)

    # Define a custom color palette
    custom_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8']

    plt.figure(figsize=(18, 6))  # Increase figure width for better spacing

    for i, hist in enumerate(hist_list):
        plt.bar(index + i * bar_width, hist, bar_width, color=custom_palette[i % len(custom_palette)], 
                label=legend_labels[i] if legend_labels else None)

    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title(title)

    # Adjust x-axis ticks: Show fewer ticks
    tick_spacing = max(1, num_bins // 10)  # Adjust spacing dynamically
    plt.xticks(index[::tick_spacing], range(num_bins)[::tick_spacing], rotation=45)  # Rotate for readability

    plt.xlim(min(index) - 1, max(index) + 1)

    if legend_labels:
        plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.yscale('log')  # Set y-axis to logarithmic scale

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


# def hamiltonianFromW(distribution, epsilon=10**-70):
#     return -jnp.log(distribution.astype(jnp.float64)+epsilon)

def hamiltonianFromW(distribution, epsilon=1e-70):
    """
    Converts a probability distribution (or weights) into a Hamiltonian: H = -log(P + epsilon)
    Fully compatible with PyTorch tensors.
    """
    distribution = distribution.to(torch.float64)  # ensure high precision
    return -torch.log(distribution + epsilon)


def find_max_indices2D(masked_array)->tuple[int, int]:
    """Returns the indices i,j of the maximum value contained in a 2D masked array. 
    Handles NaNs. """

    # Flatten the masked array to easily find the index of the maximum value
    flattened_array = masked_array.filled(fill_value=np.nan).flatten()
    
    # Find the index of the maximum value in the flattened array
    max_index = np.nanargmax(flattened_array)
    
    # Convert the flattened index to 2D index
    i, j = np.unravel_index(max_index, masked_array.shape)
    
    # Check if the maximum value is masked or NaN, if so, return None
    if masked_array.mask[i, j] or np.isnan(masked_array[i, j]):
        return None
    
    return i, j

def max_indices(array: np.ndarray) -> np.ndarray:
    """
    Given a 2D NumPy array of shape (L, 21), returns a 1D array of shape (L,)
    containing the index of the maximum value in each row.
    
    Parameters:
        array (np.ndarray): A 2D NumPy array of shape (L, 21).
        
    Returns:
        np.ndarray: A 1D NumPy array of shape (L,) with indices of max values.
    """
    return np.argmax(array, axis=1)

def reshape_array(arr): 
    """
    Reshape an array from shape (a, b, c) to shape (a, b * c).

    Parameters:
        arr (numpy.ndarray): Input array of shape (a, b, c).

    Returns:
        numpy.ndarray: Reshaped array of shape (a, b * c).
    """
    a, b, c = arr.shape
    new_shape = (a, b * c)
    reshaped_arr = arr.reshape(new_shape)
    return reshaped_arr

def kl_divergence_2d(p, q, epsilon=1e-8):
    """
    Computes the Kullback-Leibler divergence of two discrete distributions P and Q,
    where P and Q are represented as two-dimensional arrays.

    Args:
    p (numpy.ndarray): Two-dimensional array representing the first probability distribution P.
    q (numpy.ndarray): Two-dimensional array representing the second probability distribution Q.
    epsilon (float, optional): Small value to add to probabilities to avoid taking the logarithm of zero.

    Returns:
    float: The KL divergence D_KL(P || Q).
    """
    # Ensure p and q are numpy arrays and have the same shape
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("Arrays must have the same shape.")

    # Flatten the arrays to one-dimensional arrays
    p_flat = p.flatten()
    q_flat = q.flatten()

    # Ensure probabilities sum up to 1
   # if not np.isclose(np.sum(p_flat), 1.0) or not np.isclose(np.sum(q_flat), 1.0):
        #raise ValueError("Probabilities must sum up to 1.")

    # Add epsilon to probabilities to avoid taking the logarithm of zero
    p_flat = p_flat + epsilon
    q_flat = q_flat + epsilon

    # Compute KL divergence
    kl_div = np.sum(p_flat * np.log(p_flat / q_flat))
    return kl_div



def rescale_tree(newick_file, output_file):
    """
    Reads a Newick tree file, rescales each branch so that the distance 
    from any leaf to the root is exactly 1, and writes the new tree to an output file.
    
    Parameters:
        newick_file (str): Path to the input Newick file.
        output_file (str): Path to save the rescaled Newick tree.
    """
    # Load the tree
    tree = Tree(newick_file, format=1)

    # Compute the longest distance from the root to any leaf
    root = tree.get_tree_root()
    max_distance = max(tree.get_distance(leaf, root) for leaf in tree.iter_leaves())

    # Rescale each branch proportionally
    for node in tree.traverse():
        if node.dist is not None:
            node.dist = node.dist / max_distance  # Normalize branch lengths

    for leaf in tree.iter_leaves():
        initial_dist = tree.get_distance(leaf, root)
        leaf.dist = leaf.dist + (1 - initial_dist)  # Add the difference to the leaf distance

    # Save the rescaled tree
    tree.write(outfile=output_file, format=1)
    print(f"Rescaled tree saved to {output_file}")
    return tree

def get_tree_depth(tree):
    """
    Calculate the depth of a tree.
    
    Parameters:
        tree (ete3.Tree): The tree object.
        
    Returns:
        float: The depth of the tree.
    """
    node, depth = tree.get_tree_root().get_farthest_node(topology_only=True)

    print("Tree Depth:", depth)

    return depth

def get_nodes_at_depth(tree, depth):
    """
    Returns a list of nodes at a given depth in the tree.
    
    Parameters:
        tree (ete3.Tree): The input tree.
        depth (int): The depth at which to find nodes.
    
    Returns:
        list: A list of nodes at the specified depth.
    """
    return [node for node in tree.traverse() if node.get_distance(tree, topology_only=True) == depth]

def soloTreeLeveling(path_to_tree: str, path_to_new_tree: str):
    """_summary_

    Args:
        path_to_tree (str): _description_
        path_to_new_tree (str): _description_

    Returns:
        _type_: _description_
    
    Example usage:
        current_tree=leo_tree_rescaled
        for i in range(1,17):
            print('Level ', i)
            soloTreeLeveling(current_tree, remove_charsequence(leo_tree_rescaled, '.nwk')+f'_depth_{18-i}.nwk')
            current_tree=remove_charsequence(leo_tree_rescaled, '.nwk')+f'_depth_{18-i}.nwk'
    """
    # Rescale tree
    #t = rescale_tree(path_to_tree, path_to_tree+'rescaled.nwk')

    # Read tree
    t=Tree(path_to_tree)

    # Get depth of the tree
    node, depth = t.get_farthest_node(topology_only=True)
    print("Tree Depth:", depth)

    # Get all nodes of a specific depth
    nodes = get_nodes_at_depth(t, depth-1)
    print (len(nodes),' nodes at depth ', depth-1)

    # Prune them and regraft them to their parent
    for node in nodes:
        parent = node.up
        if node.is_leaf():
            print(f"Warning: Node {node.name} is a leaf, skipping...")
            continue
        
        dist=node.dist
        node.detach()
        for child in node.get_children():
            print(child.name)
            child_dist=child.dist
            child.detach()
            parent.add_child(child, dist=dist+child_dist)
    
    # Remove nodes with only children
    collapse_only_children(t, path_to_new_tree)
    
    print(f"New tree saved to {path_to_new_tree}")
    print('Depth of the new tree:', t.get_farthest_node(topology_only=True)[1])
    return t

def soloTreeLevelingRecursive(path_to_tree, path_to_new_tree):
    # Read the tree
    t = Tree(path_to_tree)
    
    # Rescale tree
    t = rescale_tree(t, 'rescaled_tree.nwk')
    
    iteration = 0
    while any(node.up and not node.is_leaf() for node in t.traverse()):
        nodes_to_remove = []
        for node in t.traverse():
            if not node.is_leaf() and node.up and all(child.is_leaf() for child in node.children):
                parent = node.up
                for child in node.children:
                    child.detach()
                    parent.add_child(child, dist=child.dist + node.dist)
                nodes_to_remove.append(node)
        
        for node in nodes_to_remove:
            node.delete()
        
        iteration += 1
        t.write(outfile=f"{path_to_new_tree}_iter{iteration}.nwk", format=1)
        print(f"Tree saved to: {path_to_new_tree}_iter{iteration}.nwk")
    
    t.write(outfile=path_to_new_tree, format=1)
    print(f"Final tree saved to: {path_to_new_tree}")
    return t

def renaming_tree_nodes(tree_path: str, strategy='preorder', starting_index=0, outfile_name=None):
    """_summary_

    Args:
        tree_path (str): Path to the newick tree file
        strategy (str, optional): strategy to traverse the tree. Can be "postorder", "preorder" or "levelorder". Defaults to 'levelorder'.

    Returns:
        Tree: a tree with the nodes indexed according to the traversing order. Does not rename leaves. 
    """
    tree=Tree(tree_path)
    index=starting_index # or 1?
    for node in tree.traverse(strategy=strategy):
        if not node.is_leaf():
            node.name="N"+str(index)
            index+=1
    try:
        if not outfile_name:
            tree.write(outfile=tree_path.replace(".nwk", "")+'_'+strategy+'.nwk', format=1)
            print(f"Tree saved to {tree_path.replace('.nwk', '')}_{strategy}.nwk")
        else:
            tree.write(outfile=outfile_name, format=1)
            print(f"Tree saved to {outfile_name}")
    except Exception as e:
        print(f"Failed to write tree: {e}")
    print(tree)
    return tree

def append_sequence_to_fasta(input_file, output_file, header, sequence):
    """
    Append a new sequence to an existing FASTA file and write to a new file.
    
    Parameters:
    -----------
    input_file : str
        Path to the input FASTA file
    output_file : str
        Path to the output FASTA file that will contain all original sequences plus the new one
    header : str
        Header for the new sequence (without '>')
    sequence : str
        The new sequence to append
    
    Returns:
    --------
    None
    """
    # Read the original file
    with open(input_file, 'r') as f:
        original_content = f.read()
    
    # Format the new sequence with proper FASTA format
    # Make sure the header starts with '>' and doesn't have extra line breaks
    if not header.startswith('>'):
        header = '>' + header
    
    # Remove any existing line breaks in the header and sequence
    header = header.strip()
    sequence = sequence.strip()
    
    # Format the sequence (optional: break into lines of 60 characters)
    formatted_sequence = ''
    for i in range(0, len(sequence), 60):
        formatted_sequence += sequence[i:i+60] + '\n'
    
    # Make sure the original content ends with a newline
    if original_content and not original_content.endswith('\n'):
        original_content += '\n'
    
    # Write to the new file
    with open(output_file, 'w') as f:
        f.write(original_content)
        f.write(f"{header}\n")
        f.write(formatted_sequence)
    
    print(f"Successfully appended sequence to {output_file}")

def reformat_fasta_to_single_line(input_fasta_path: str, output_fasta_path: str) -> None:
    """
    Reformat a FASTA file so that each sequence is on a single line.
    
    Args:
        input_fasta_path (str): Path to input FASTA file (sequences may be multi-line)
        output_fasta_path (str): Path to output FASTA file (sequences on single lines)
    """
    from Bio import SeqIO
    
    # Read sequences
    records = list(SeqIO.parse(input_fasta_path, "fasta"))
    
    # Write sequences with each sequence on a single line
    with open(output_fasta_path, 'w') as f:
        for record in records:
            f.write(f">{record.id}\n")
            f.write(f"{str(record.seq)}\n")
    
    print(f"Reformatted FASTA saved to: {output_fasta_path}")
    print(f"Processed {len(records)} sequences")

def get_subtree_consensus(node_of_interest: str, tree_path: str, msa_path: str, save_name: str, save_folder: str, reweighted: bool = False) -> np.ndarray:
    """
    Get the subtree consensus from a tree and a MSA.
    
    Args:
        tree_path (str): Path to the tree file.
        msa_path (str): Path to the MSA file.
        save_name (str): Name for the saved consensus tree.
        save_folder (str): Folder to save the consensus tree.
    
    Returns:
        The consensus sequence of the subtree of the node of interest. Saves it as a numpy array of integers.
    """
    
    createFolder(save_folder)
    # Read the tree
    t=Tree(tree_path, format=1)
    
    # Read the MSA
    seqs, name_to_index, names=read_fasta2(msa_path)
    
    # Add sequences to leaves
    for leaf in t.get_leaves():
        leaf.add_features(sequence=seqs[name_to_index[leaf.name]])
    
    if node_of_interest == 'root':
        # If the node of interest is the root, we can directly use the root node
        node_of_interest = t.get_tree_root()
    
    else:
        # Get the consensus sequence of the subtree of the node of interest
        try:
            node_of_interest = t.search_nodes(name=node_of_interest)[0]
        except IndexError:
            #raise ValueError
            print(f"Node '{node_of_interest}' not found in the tree.")
            # Prompt user to see if they want to use the root node instead
            prompt = input(f"Node '{node_of_interest}' not found. Do you want to use the root node instead? (y/n): ")
            if prompt.lower() == 'y':
                node_of_interest = t.get_tree_root()
            else:
                raise ValueError(f"Node '{node_of_interest}' not found in the tree and user chose not to use the root node.")
    subtree = node_of_interest.get_leaves()
    pre_sub_sequences= [leaf.sequence for leaf in subtree]
    sub_sequences = []
    for seq in pre_sub_sequences:
        if seq is torch.Tensor:
            seq=seq.numpy().astype(int)
        sub_sequences.append(seq)
    if reweighted:
        freqs=getReweightedDistribution(sub_sequences)
    else:
        freqs=getFrequencyDistribution(sub_sequences)
    consensus=get_maximum_likelihood_sequence(freqs)
    np.savetxt(save_folder+save_name, np.array(consensus).astype(int), fmt='%d')
    print("Consensus sequence saved to", save_folder+save_name) 
    print('Consensus sequence:', consensus)
    return consensus


####for more functions on Bart trees and Hamming distances see maybeUseful
