from functools import partial
import jax.numpy as jnp
import jax
from jax import config, jit
from utils.inferringMu import fit_mu
from utils.samplerOnTrees import mutate_felsenstein
from utils.utils import to_numpy, createFolder, get_all_file_paths, remove_charsequence
config.update("jax_enable_x64", True)
import numpy as np
from ete3 import Tree
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pickle
import os
from scipy.stats import linregress

from utils.toolsForTreesAndMSAs import MSAFromLeafSequences, get_maximum_likelihood_sequence, getFrequencyDistribution, getReweightedDistribution, normalizePerRow, randomW, read_fasta1, read_fasta2, reroot, calculate_hamming_distance, create_MSA_profile
from utils.PottsEnergies import readParametersFraZ, energy


def softmax_2d(array, axis=-1):
    """
    Apply the softmax function to a 2D JAX array along the specified axis.

    Parameters:
    array (jnp.ndarray): Input 2D array.
    axis (int): Axis along which to apply the softmax function. Default is -1.

    Returns:
    jnp.ndarray: Softmax-transformed array with the same shape as input.
    """
    # Convert the array to float64 for higher precision
    array_float64 = array.astype(jnp.float64)

    # Compute the exponentials of the array
    exp_array = jnp.exp(array_float64)

    # Compute the sum of the exponentials along the specified axis
    sum_exp_array = jnp.sum(exp_array, axis=axis, keepdims=True)

    # Compute the softmax by dividing the exponentials by their sum along the specified axis
    softmax_array = exp_array / sum_exp_array

    return softmax_array

def analyticalLogLikelihoodPerSite(MSA, emp_freq, w, site, q=21):
    """Log likelihood per site in the event of equilibrium, meaning mu --> infinity"""
    N=len(MSA)
    L=[
        emp_freq[site,i]*np.log(np.maximum(w[site, i], 10**(-70)))

        for i in range(q)
    ]
    return N*np.sum(np.array(L))

def analyticalLikelihood(MSA, emp_freq, w):
    """Total analytical likelihood or the product of the single site likelihood over all sites."""
    log_prod=0
    for site in range(len(MSA[0])):
        log_prod+=analyticalLogLikelihoodPerSite(MSA, emp_freq, w, site)
    return jnp.exp(log_prod)

@partial(jit,static_argnames=('t'))
def FelsensteinNoNormalization(t, mu, w_site, site): 

    """Returns L(s)*w[s]/sum_i(L(s_i)*w[s_i]) for the chosen node, as the probability distribution.
     P.S. Here we only consider w for one site (1D of size q, not 2D of size L*q) """
     

    def felsenstein_single_no_normalization(node, mu, w_site, site):
        """
        Function to run the Felsenstein algorithm on a single site.

        Returns:
            NodeProb: Probability distribution over amino acids for the node
        """
        log_node_prob = jnp.zeros(21, dtype=jnp.float64)

        for child in node.children:
            if child.is_leaf():
                sequence_value = jnp.take(child.sequence, site)
                child_prob = jnp.equal(jnp.arange(21), sequence_value).astype(jnp.float64) #returns an array of booleans, length is 21
            else:
                child_prob = felsenstein_single_no_normalization(child, mu, w_site, site)  # Recursive call

            W = jnp.array(jnp.exp(-mu * child.dist), dtype=jnp.float64)
            #print("Shape of w:", w_site.shape)
            #print("Shape of W:", W.shape)
            #print("Shape of outer product:", jnp.outer(w_site, jnp.ones(21)).shape)


            transition_prob =  jnp.outer(jnp.ones(21, dtype=jnp.float64), w_site) * (jnp.array(1,dtype=jnp.float64) - W) + W * jnp.eye(21, dtype=jnp.float64) #shape is (21,21)#jnp.outer(w_site, jnp.ones(21, dtype=jnp.float64)) * (jnp.array(1,dtype=jnp.float64) - W) + W * jnp.eye(21, dtype=jnp.float64) #shape is (21,21)
            log_node_prob += jnp.log(jnp.matmul(transition_prob,child_prob))


        #print("Data types:", transition_prob.dtype, child_prob.dtype)
        #print("Shapes:", transition_prob.shape, child_prob.shape)

        node_prob = jnp.exp(log_node_prob)
        #print("Shape of log node prob", log_node_prob.shape)
        #node_prob /= jnp.sum(node_prob)  # Normalize

        #node.add_features('proba'=node_prob)  # Update node with inferred probabilities
        return node_prob


    site_distribution =felsenstein_single_no_normalization(t.get_tree_root(), mu, w_site, site)*w_site
    print(site_distribution.shape)#felsenstein_single(t.get_tree_root(), mu, w)*w #jax.vmap(lambda a: likelihoodAncestral(chosen_node, mu, h, a, site)*jnp.exp(-h[a]))(a_range) #here the /jnp.sum(jnp.exp(-h)) simplifies so it is unnecessary
    return jnp.sum(site_distribution)#/jnp.sum(site_distribution) # A 1D array of length q. Represents the probability distribution for the site 'site'.

@partial(jit,static_argnames=('t'))
def FelsensteinWithExponent(t, mu, w_site, site): 

    """Returns L(s)*w[s]/sum_i(L(s_i)*w[s_i]) for the chosen node, as the probability distribution.
     P.S. Here we only consider w for one site (1D of size q, not 2D of size L*q) """
     

    def felsenstein_single_with_exponent(node, mu, w_site, site):
        """
        Function to run the Felsenstein algorithm on a single site.

        Returns:
            NodeProb: Probability distribution over amino acids for the node

        Example usage:
        path_to_tree='DBDtree_collapsed_noonlychild_midpointrooted_prunedsubtree301.nwk'
        t=Tree(path_to_tree)
        MSA, name_to_index = read_fasta2('seq1_mu42.88.fa')
        for leaf in t.get_leaves():
            leaf.add_features(sequence=np.array(MSA[name_to_index[leaf.name]]))
        w = getFrequencyDistribution(MSA)
        for site in [2,18,47,71]:
            print('Site', site)
            likelihood=FelsensteinWithExponent(t, 10, w[site], site)
            print(likelihood)
            print('Log likelihood', jnp.log(likelihood))
            print('Log likelihood2', FelsensteinWithExponentLogLikelihood(t, 10, w[site], site))
        """
        log_node_prob = jnp.zeros(21, dtype=jnp.float64)
        node_exponent=jnp.array(0,dtype=jnp.float64)

        for child in node.children:
            if child.is_leaf():
                sequence_value = jnp.take(child.sequence, site)
                child_prob, child_exponent = jnp.equal(jnp.arange(21), sequence_value).astype(jnp.float64), jnp.array(0,dtype=jnp.float64) #returns an array of booleans, length is 21
            else:
                child_prob, child_exponent = felsenstein_single_with_exponent(child, mu, w_site, site)  # Recursive call

            W = jnp.array(jnp.exp(-mu * child.dist), dtype=jnp.float64)
            #print("Shape of w:", w_site.shape)
            #print("Shape of W:", W.shape)
            #print("Shape of outer product:", jnp.outer(w_site, jnp.ones(21)).shape)


            transition_prob =  jnp.outer(jnp.ones(21, dtype=jnp.float64), w_site) * (jnp.array(1,dtype=jnp.float64) - W) + W * jnp.eye(21, dtype=jnp.float64) #shape is (21,21)#jnp.outer(w_site, jnp.ones(21, dtype=jnp.float64)) * (jnp.array(1,dtype=jnp.float64) - W) + W * jnp.eye(21, dtype=jnp.float64) #shape is (21,21)
            log_node_prob += jnp.log(jnp.matmul(transition_prob,child_prob))
            node_exponent+=child_exponent


        #print("Data types:", transition_prob.dtype, child_prob.dtype)
        #print("Shapes:", transition_prob.shape, child_prob.shape)

        node_prob = jnp.exp(log_node_prob)
        #print("Shape of log node prob", log_node_prob.shape)
        sum=jnp.sum(node_prob)
        node_prob /= sum  # Normalize

        #node.add_features(proba=node_prob)  # Update node with inferred probabilities

        node_exponent+=jnp.log(sum)

        #node.add_features(exponent=node_exponent)

        return node_prob, node_exponent


    root_prob, final_exponent =felsenstein_single_with_exponent(t.get_tree_root(), mu, w_site, site)
    site_distribution=root_prob*w_site
    print(site_distribution.shape)#felsenstein_single(t.get_tree_root(), mu, w)*w #jax.vmap(lambda a: likelihoodAncestral(chosen_node, mu, h, a, site)*jnp.exp(-h[a]))(a_range) #here the /jnp.sum(jnp.exp(-h)) simplifies so it is unnecessary
    return jnp.sum(site_distribution)*jnp.exp(final_exponent)#/jnp.sum(site_distribution) # A 1D array of length q. Represents the probability distribution for the site 'site'.




@partial(jit,static_argnames=('t'))
def FelsensteinWithExponentLogLikelihood(t, mu, w_site, site): 

    """Returns L(s)*w[s]/sum_i(L(s_i)*w[s_i]) for the chosen node, as the probability distribution.
     P.S. Here we only consider w for one site (1D of size q, not 2D of size L*q) """
     

    def felsenstein_single_with_exponent(node, mu, w_site, site):
        """
        Function to run the Felsenstein algorithm on a single site.

        Returns:
            NodeProb: Probability distribution over amino acids for the node
        """
        log_node_prob = jnp.zeros(21, dtype=jnp.float64)
        node_exponent=jnp.array(0,dtype=jnp.float64)

        for child in node.children:
            if child.is_leaf():
                sequence_value = jnp.take(child.sequence, site)
                child_prob, child_exponent = jnp.equal(jnp.arange(21), sequence_value).astype(jnp.float64), jnp.array(0,dtype=jnp.float64) #returns an array of booleans, length is 21
            else:
                child_prob, child_exponent = felsenstein_single_with_exponent(child, mu, w_site, site)  # Recursive call

            W = jnp.array(jnp.exp(-mu * child.dist), dtype=jnp.float64)
            #print("Shape of w:", w_site.shape)
            #print("Shape of W:", W.shape)
            #print("Shape of outer product:", jnp.outer(w_site, jnp.ones(21)).shape)


            transition_prob =  jnp.outer(jnp.ones(21, dtype=jnp.float64), w_site) * (jnp.array(1,dtype=jnp.float64) - W) + W * jnp.eye(21, dtype=jnp.float64) #shape is (21,21)#jnp.outer(w_site, jnp.ones(21, dtype=jnp.float64)) * (jnp.array(1,dtype=jnp.float64) - W) + W * jnp.eye(21, dtype=jnp.float64) #shape is (21,21)
            log_node_prob += jnp.log(jnp.matmul(transition_prob,child_prob))
            node_exponent+=child_exponent


        #print("Data types:", transition_prob.dtype, child_prob.dtype)
        #print("Shapes:", transition_prob.shape, child_prob.shape)
        #new version with normalization by summing the max and the min
        #log_norm_factor=(jnp.min(log_node_prob)+jnp.max(log_node_prob))/2 #set this as the log exponent #normalize by this 
        node_prob = jnp.exp(log_node_prob)
        #print("Shape of log node prob", log_node_prob.shape)
        sum=jnp.sum(node_prob)
        node_prob /= sum  # Normalize

        #node.add_features(proba=node_prob)  # Update node with inferred probabilities

        node_exponent+=jnp.log(sum)

        #node.add_features(exponent=node_exponent)

        return node_prob, node_exponent


    root_prob, final_exponent =felsenstein_single_with_exponent(t.get_tree_root(), mu, w_site, site)
    site_distribution=root_prob*w_site
    print(site_distribution.shape)#felsenstein_single(t.get_tree_root(), mu, w)*w #jax.vmap(lambda a: likelihoodAncestral(chosen_node, mu, h, a, site)*jnp.exp(-h[a]))(a_range) #here the /jnp.sum(jnp.exp(-h)) simplifies so it is unnecessary
    return jnp.log(jnp.sum(site_distribution))+final_exponent

def FelsensteinWithExponentLogLikelihoodNoPrior(t, mu, w_site, site): 

    """Returns L(s)*w[s]/sum_i(L(s_i)*w[s_i]) for the chosen node, as the probability distribution.
     P.S. Here we only consider w for one site (1D of size q, not 2D of size L*q) """
     

    def felsenstein_single_with_exponent(node, mu, w_site, site):
        """
        Function to run the Felsenstein algorithm on a single site.

        Returns:
            NodeProb: Probability distribution over amino acids for the node
        """
        log_node_prob = jnp.zeros(21, dtype=jnp.float64)
        node_exponent=jnp.array(0,dtype=jnp.float64)

        for child in node.children:
            if child.is_leaf():
                sequence_value = jnp.take(child.sequence, site)
                child_prob, child_exponent = jnp.equal(jnp.arange(21), sequence_value).astype(jnp.float64), jnp.array(0,dtype=jnp.float64) #returns an array of booleans, length is 21
            else:
                child_prob, child_exponent = felsenstein_single_with_exponent(child, mu, w_site, site)  # Recursive call

            W = jnp.array(jnp.exp(-mu * child.dist), dtype=jnp.float64)
            #print("Shape of w:", w_site.shape)
            #print("Shape of W:", W.shape)
            #print("Shape of outer product:", jnp.outer(w_site, jnp.ones(21)).shape)


            transition_prob =  jnp.outer(jnp.ones(21, dtype=jnp.float64), w_site) * (jnp.array(1,dtype=jnp.float64) - W) + W * jnp.eye(21, dtype=jnp.float64) #shape is (21,21)#jnp.outer(w_site, jnp.ones(21, dtype=jnp.float64)) * (jnp.array(1,dtype=jnp.float64) - W) + W * jnp.eye(21, dtype=jnp.float64) #shape is (21,21)
            log_node_prob += jnp.log(jnp.matmul(transition_prob,child_prob))
            node_exponent+=child_exponent


        #print("Data types:", transition_prob.dtype, child_prob.dtype)
        #print("Shapes:", transition_prob.shape, child_prob.shape)
        #new version with normalization by summing the max and the min
        #log_norm_factor=(jnp.min(log_node_prob)+jnp.max(log_node_prob))/2 #set this as the log exponent #normalize by this 
        node_prob = jnp.exp(log_node_prob)
        #print("Shape of log node prob", log_node_prob.shape)
        sum=jnp.sum(node_prob)
        node_prob /= sum  # Normalize

        #node.add_features(proba=node_prob)  # Update node with inferred probabilities

        node_exponent+=jnp.log(sum)

        #node.add_features(exponent=node_exponent)

        return node_prob, node_exponent


    root_prob, final_exponent =felsenstein_single_with_exponent(t.get_tree_root(), mu, w_site, site)
    #site_distribution=root_prob*w_site
    #print(site_distribution.shape)#felsenstein_single(t.get_tree_root(), mu, w)*w #jax.vmap(lambda a: likelihoodAncestral(chosen_node, mu, h, a, site)*jnp.exp(-h[a]))(a_range) #here the /jnp.sum(jnp.exp(-h)) simplifies so it is unnecessary
    return jnp.log(jnp.sum(root_prob))+final_exponent

def plotLikelihoodConvergence(root_sequence, path_to_tree, mu_test=400, w_nb_to_test=10):
    """Testing the likelihood for mu -> inifnity 
    To optimize because it does not converge exactly for some reason."""
    L=len(root_sequence)
    q=21
    t=Tree(path_to_tree)
    num_leaves=len([leaf for leaf in t.iter_leaves()])
    #Generating a list of random w
    ws=[jnp.array(randomW(L,q)) for i in range(w_nb_to_test)]
    exactL=[]
    limitL=[]

    sites=jnp.arange(L)

    #initializing the root sequence
    t.get_tree_root().add_features(sequence=root_sequence)

    for w in ws: #w is the ground truth

        print('w', w)

        #sample on the tree with felsenstein
        mutate_felsenstein(t.get_tree_root(),mu_test,w)

        #Calculating the likelhood with our function
        exactL.append(jax.vmap(lambda site: -FelsensteinWithExponentLogLikelihoodNoPrior(t.get_tree_root(), mu_test, w[site], site))(sites))

        #Then with the analytical formula
        generated_sequences=MSAFromLeafSequences(t)
        emp_freq=getFrequencyDistribution(generated_sequences)
        print('Empirical frequencies shape', emp_freq.shape)

        limitL.append([])
        for site in sites:
            limitL[-1].append(-analyticalLogLikelihoodPerSite(generated_sequences, emp_freq, w, site, q=21))
        limitL[-1]=np.array(limitL[-1]).flatten()
    #Plotting them
    print(f"Size of exactL: {np.array(exactL).flatten().size}")
    print(f"Size of limitL: {np.array(limitL).flatten().size}")
    print(f"Size of limitL: {np.array(limitL).shape}")


    # Indexes for the x-axis (0, 1, 2, ...)
    #return limitL,exactL
    x = range(len(exactL)*len(exactL[0]))

    # Create the scatter plot
    plt.scatter(x, np.array(exactL).flatten(), color='blue', label='Exact neg log likelihood')
    #plt.scatter(x, [l*1.8 for l in exactL], color='green', label='Exact likelihood *1.8')

    x2 = range(len(np.array(limitL).flatten()))

    plt.scatter(x2, np.array(limitL).flatten(), color='red', label='Analytical neg log likelihood limit')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Negative log-likelihood')
    plt.title('Checking convergence of the likelihood for mu='+str(mu_test)+" on a tree of "+str(num_leaves)+" leaves")

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()

    plt.scatter(np.array(exactL).flatten(), np.array(limitL).flatten())
    plt.xlabel('Exact negative log-likelihood')
    plt.ylabel('Analytical negative log-likelihood')
    print(f'Pearson correlation is {pearsonr(np.array(exactL).flatten(), np.array(limitL).flatten())}')
    plt.show()

    """Use example:
    root_sequence=read_fasta1('alignments_etc/PF00014/iidSample1_SparsePF14_bmDCA_fasta.txt')[10]
    path_to_tree='tree_structures/PF00014/PF00014_no_dupli_Alya_tree_short_100'
    limitL,exactL=plotLikelihoodConvergence(root_sequence, path_to_tree, mu_test=1000, w_nb_to_test=5)
    """


    return limitL,exactL

def neg_log_likelihood(t, mu, w_site, site):
    #return -jnp.log(FelsensteinNoNormalization(t, mu, w_site, site))
    #return -jnp.log(FelsensteinWithExponent(t, mu, w_site, site))
    return -FelsensteinWithExponentLogLikelihood(t, mu, w_site, site)

def plotPearsonCorrsWithReplicates(dict_dirs: list, title='Correlation analysis with replicates'):
    """_summary_

    Args:
        dict_dirs (list): _description_
        title (str, optional): _description_. Defaults to 'Correlation analysis with replicates'.

    Example usage:
        mu_interval=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,5,10,20,50,100]

        for n in range(3):
            pearsonCorrsH(MSA[0], path_to_tree, MSA, mu_interval, 'dict_n='+str(n), 'dict_folder_f_uni/', 'freqs_folder_f_uni/', error=0, start_from_emp=False, lr=0.01, num_steps=500)
            pearsonCorrsH(MSA[0], path_to_tree, MSA, mu_interval, 'dict_n='+str(n), 'dict_folder_f_emp/', 'freqs_folder_f_emp/', error=0, start_from_emp=True, lr=0.01, num_steps=500)
    """
    if not all(os.path.isdir(d) for d in dict_dirs):
        print("Error: One or more provided paths are not valid directories.")
        return
    
    # Initialize a dictionary to aggregate data
    aggregated_data = {}
    
    # Process each directory
    for dict_dir in dict_dirs:
        for filename in os.listdir(dict_dir):
            file_path = os.path.join(dict_dir, filename)
            
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'rb') as file:
                        dict_pearson = pickle.load(file)
                    
                    for mu, values in dict_pearson.items():
                        if mu not in aggregated_data:
                            aggregated_data[mu] = {
                                "inf_corr": [],
                                "emp_corr": [],
                                "emp_corr_rw": [],
                            }
                        aggregated_data[mu]["inf_corr"].append(values[0])
                        aggregated_data[mu]["emp_corr"].append(values[1])
                        aggregated_data[mu]["emp_corr_rw"].append(values[2])
                
                except (pickle.UnpicklingError, Exception) as e:
                    print(f"Error processing file {filename} in {dict_dir}: {e}")
    
    # Prepare data for plotting
    mus = sorted(aggregated_data.keys())
    inf_corr_means = []
    emp_corr_means = []
    emp_corr_rw_means = []
    inf_corr_stds = []
    emp_corr_stds = []
    emp_corr_rw_stds = []
    
    for mu in mus:
        inf_corr_means.append(np.mean(aggregated_data[mu]["inf_corr"]))
        emp_corr_means.append(np.mean(aggregated_data[mu]["emp_corr"]))
        emp_corr_rw_means.append(np.mean(aggregated_data[mu]["emp_corr_rw"]))
        
        inf_corr_stds.append(np.std(aggregated_data[mu]["inf_corr"]))
        emp_corr_stds.append(np.std(aggregated_data[mu]["emp_corr"]))
        emp_corr_rw_stds.append(np.std(aggregated_data[mu]["emp_corr_rw"]))
    
    # Plot with error bars
    plt.figure(figsize=(10, 8))
    plt.errorbar(mus, inf_corr_means, yerr=inf_corr_stds, fmt='o', color='blue', label='Inferred correlation')
    plt.errorbar(mus, emp_corr_means, yerr=emp_corr_stds, fmt='o', color='orange', label='Empirical correlation')
    plt.errorbar(mus, emp_corr_rw_means, yerr=emp_corr_rw_stds, fmt='o', color='black', label='Empirical correlation w reweighting')
    
    plt.xlabel('Values of mu')
    plt.ylabel('Pearson correlation with GT')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_felsenstein_inference(data_folder, save_folder):
    """
    Generate scatter plots comparing empirical and inferred frequencies to ground truth frequencies.
    
    Parameters:
    - data_folder (str): Path to the folder containing frequency files.
    - save_folder (str): Path to the folder to save plots.
    
    Assumes files are named like:
        - `inferred_freq301_leaves_mu=5_error=0`
        - `GTFelsenstein_301_leaves_mu=0.5`
        - `empfreqsFelsenstein_301_leaves_mu=0.3`
    
    Example usage:
        pearsonCorrsH(MSA[0], path_to_tree, MSA, [100], 'test_dict', 'test_folder_dict/', 'test_folder_freqs/', error=0, start_from_emp=True, lr=0.01, num_steps=500)
        plot_felsenstein_inference('test_folder_freqs', 'test_felsie')
    """
    createFolder(save_folder)
    # Get all files in the folder
    files = os.listdir(data_folder)
    
    # Filter and group files based on the naming convention
    inferred_files = [f for f in files if f.startswith("inferred_freq")]
    gt_files = [f for f in files if f.startswith("GTFelsenstein")]
    emp_files = [f for f in files if f.startswith("empfreqsFelsenstein")]

    for inferred_file in inferred_files:
        # Extract mu value from the inferred file name
        mu = inferred_file.split("_mu=")[-1].split("_")[0]

        # Find corresponding ground truth and empirical files
        gt_file = next((f for f in gt_files if f"_mu={mu}" in f), None)
        emp_file = next((f for f in emp_files if f"_mu={mu}" in f), None)
        
        if not gt_file or not emp_file:
            print(f"Missing ground truth or empirical file for mu={mu}. Skipping...")
            continue
        
        # Load the arrays using np.loadtxt
        inferred = np.loadtxt(os.path.join(data_folder, inferred_file)).flatten()
        ground_truth = np.loadtxt(os.path.join(data_folder, gt_file)).flatten()
        empirical = np.loadtxt(os.path.join(data_folder, emp_file)).flatten()

        # Create scatter plots
        plt.scatter(empirical, ground_truth, label='Empirical frequencies')
        plt.scatter(inferred, ground_truth, label='Inferred frequencies')

        # Fit and plot regression lines
        emp_slope, emp_intercept, _, _, _ = linregress(empirical, ground_truth)
        inf_slope, inf_intercept, _, _, _ = linregress(inferred, ground_truth)

        plt.plot(empirical, emp_slope * empirical + emp_intercept, color='red',
                 label=f'Empirical fitted line (slope={emp_slope:.2f})')
        plt.plot(inferred, inf_slope * inferred + inf_intercept, color='blue',
                 label=f'Inferred fitted line (slope={inf_slope:.2f})')

        # Calculate Pearson correlation coefficients
        emp_corr = np.corrcoef(empirical, ground_truth)[0, 1]
        inf_corr = np.corrcoef(inferred, ground_truth)[0, 1]

        # Add labels and title
        plt.xlabel('Empirical/Inferred frequencies')
        plt.ylabel('Ground truth frequencies')
        plt.title(f'Felsenstein inference for mu={mu}')
        plt.legend()

        # Save the plot
        save_path = os.path.join(save_folder, f'Felsenstein_inference_mu={mu}.png')
        plt.savefig(save_path)
        plt.close()

        # Print the results
        print(f"Results for mu={mu}:")
        print(f"  Empirical slope: {emp_slope}")
        print(f"  Inferred slope: {inf_slope}")
        print(f"  Pearson correlation (empirical): {emp_corr}")
        print(f"  Pearson correlation (inferred): {inf_corr}")

@partial(jit,static_argnames=('t'))
def FelsensteinAncestral(t, mu, w_site, site): #let us try to return a 1D array of length q for a given site and a given node

    """Returns L(s)*w[s]/sum_i(L(s_i)*w[s_i]) for the chosen node, as the probability distribution.
     P.S. Here we only consider w for one site (1D of size q, not 2D of size L*q) """
     

    def felsenstein_single(node, mu, w_site, site):
        """
        Function to run the Felsenstein algorithm on a single site.

        Returns:
            NodeProb: Probability distribution over amino acids for the node
        """
        log_node_prob = jnp.zeros(21, dtype=jnp.float64)

        for child in node.children:
            if child.is_leaf():
                child_prob = jnp.equal(jnp.arange(21), jnp.take(child.sequence, site)).astype(jnp.float64) #returns an array of booleans, length is 21
            else:
                child_prob = felsenstein_single(child, mu, w_site, site)  # Recursive call

            W = jnp.array(jnp.exp(-mu * child.dist), dtype=jnp.float64)
            #print("Shape of w:", w_site.shape)
            #print("Shape of W:", W.shape)
            #print("Shape of outer product:", jnp.outer(w_site, jnp.ones(21)).shape)
            log_node_prob += jnp.log(jnp.matmul(
                jnp.outer(
                    jnp.ones(21, dtype=jnp.float64), w_site) * (jnp.array(1,dtype=jnp.float64) - W) + W * jnp.eye(21, dtype=jnp.float64
                                                                                                                  ),
                child_prob))

        return jnp.exp(log_node_prob)/jnp.sum(jnp.exp(log_node_prob))


    site_distribution =felsenstein_single(t.get_tree_root(), mu, w_site, site)*w_site
    #print(site_distribution.shape)#felsenstein_single(t.get_tree_root(), mu, w)*w #jax.vmap(lambda a: likelihoodAncestral(chosen_node, mu, h, a, site)*jnp.exp(-h[a]))(a_range) #here the /jnp.sum(jnp.exp(-h)) simplifies so it is unnecessary
    return site_distribution/jnp.sum(site_distribution) # A 1D array of length q. Represents the probability distribution for the site 'site'.

def samplingAndLikelihood(path_to_tree='tree_structures/PF00014/PF00014_tree_alya_short_100',
                          path_to_alignment='alignments_etc/PF00014/iidSample1_SparsePF14_bmDCA_fasta.txt', mu=jnp.array(200)):
    
    """Samples on a tree with the Felsenstein propagator and a random w, and returns the computed likelihood.
    A useful tool to test the code for likelihood computation."""
    
    t=Tree(path_to_tree)
    root_sequence=read_fasta1(path_to_alignment)[0]
    num_leaves=len([leaf for leaf in t.iter_leaves()])

    L=len(root_sequence)
    q=21
    w_GT=jnp.array(randomW(L,q))

    #initializing the root sequence
    t.get_tree_root().add_features(sequence=np.array(root_sequence))

    #sample on the tree with felsenstein
    mutate_felsenstein(t, mu, w_GT)

    site_range=jnp.arange(L)

    likelihood1=jax.vmap(lambda site: FelsensteinWithExponent(t.get_tree_root(), mu, w_GT[site], site))(site_range)
    #likelihood2=jax.vmap(lambda site: FelsensteinNoNormalization(t.get_tree_root(), mu, w_GT[site], site))(site_range)
    return likelihood1

def softmax_projection(params):
    #"""Transforms the parameters to lie on the simplex (i.e., sum to 1)."""
    return jax.nn.softmax(params)

@partial(jax.jit, static_argnames=('t',))
def objective(params, t, mu, site):
    # Apply the softmax transformation to ensure weights sum to 1
    w = softmax_projection(params)
    
    return -FelsensteinWithExponentLogLikelihood(t, mu, w, site)

createFolder('gradient_figs/')
def optimize_h(mu, h, t, site, lr=0.01, num_steps=500):
    @partial(jax.jit, static_argnames=('t',))
    def step(mu, h, t, site, lr):
        # Compute gradients
        grad_h = jax.grad(objective, argnums=0)(h, t.get_tree_root(), mu, site)
        
        # Update h
        h = h - lr * grad_h
        
        return h, grad_h

    evo_likelihood = jnp.zeros(num_steps)
    evo_grad = jnp.zeros(num_steps)

    for i in range(num_steps):
        h, grad_h = step(mu, h, t, site, lr)

        # Track progress
        evo_grad = evo_grad.at[i].set(jnp.linalg.norm(grad_h))
        evo_likelihood = evo_likelihood.at[i].set(objective(h, t.get_tree_root(), mu, site))
    w=softmax_projection(h)
    
    return w, evo_grad, evo_likelihood



def inferring_w_from_MSA(sequences_path: str, path_to_tree: str, save_folder: str, save_name: str, lr=0.1, num_steps=500, start_from_emp=True, save_gradients=True, format=0)-> tuple:
    
    """_summary_

    Returns:
        _type_: _description_

    Example usage:
        base_directory = "alignments_etc/3_seq_DBDtree_collapsed_noonlychild_midpointrooted_prunedsubtree301"
        filenames=get_all_file_paths(base_directory)
        path_to_tree='tree_structures/DBDtree_collapsed_noonlychild_midpointrooted_prunedsubtree301.nwk'

        clean_filenames=[]
        for file in filenames:
            if not '.ipynb_checkpoints' in file:
                if '.fa' in file:
                    clean_filenames.append(file)
        for file in clean_filenames:
            name=remove_charsequence(file.split('/')[-1], '.fa')
            print(name)
            freq, reweighted_freq, w_ = inferring_w_from_MSA(file, path_to_tree, 'inferred_w_for_leo_seqs/', lr=0.05, num_steps=500, start_from_emp=True)#(file, 'tree_structures/DBDtree_collapsed_noonlychild_midpointrooted_prunedsubtree301.nwk', name, lr=0.0001, num_steps=2000, save_folder='inferred_w/')

    """
    
    
    # Reading the parameters and the tree
    t=Tree(path_to_tree, format=format)
    MSA, name_to_index, names=read_fasta2(sequences_path)
    num_leaves=len([leaf for leaf in t.iter_leaves()])
    L=len(MSA[0])
    q=21
    createFolder(save_folder)

    # Putting the sequences on the leaves 
    for leaf in t.iter_leaves():
        leaf.add_features(sequence=np.array(MSA[name_to_index[leaf.name]]))

    mu=fit_mu(t, sequences_path)
    print(mu)
    
    #compute the empirical frequencies    
    freq=getFrequencyDistribution(MSA) #empirical frequencies from the MSA #redo it with the weighted frequencies
    reweighted_freq=getReweightedDistribution(MSA,save_folder+'/reweighted_freqs_'+str(num_leaves)+"_leaves_inferred_mu="+str(mu)+save_name)
    
    fname=save_folder+"empfreqsFelsenstein_"+str(num_leaves)+"_leaves_inferred_mu="+str(mu)+save_name;np.savetxt(fname, np.array(freq), fmt='%.18e', delimiter=' ', newline='\n', comments='# ') #np.save(fname+'.npy', np.array(freq))#
    print('mu', mu, 'empirical frequencies', freq)

    # Defining the initial conditions
    if start_from_emp:
        pseudocount=10**(-5)
        w_init=normalizePerRow(jnp.array(freq+pseudocount))
    else:   
        w_init=jnp.ones((L,q))/q

    h_init=jnp.log(w_init)
        
    # Compute the argmax for the likelihood (inferred frequencies) by gradient descent on the likelihood
    vmap_function = jax.vmap(optimize_h, in_axes=(None, 0, None, 0, None, None))

    w_, evo_grad_, evo_likelihood_ = vmap_function(mu, h_init, t.get_tree_root(), jnp.arange(L), lr, num_steps)

    if save_gradients:
        createFolder('gradent_figs/')

        epochs=np.arange(num_steps)

        for site in range(L):
            plt.scatter(epochs, evo_grad_[site], label=f'Gradient site {site}')
            plt.scatter(epochs, evo_likelihood_[site], label=f'Negative log-likelihood site {site}')
            plt.xlabel('epochs')
            plt.legend()
            plt.savefig('gradient_figs/'+save_name+'_site'+str(site)+'.png')
            plt.show()

    # Saving w_inferred and freq and the ground truth
    fname=save_folder+"inferred_freq"+str(num_leaves)+"_leaves_inferred_mu="+str(mu)+save_name;np.savetxt(fname,  w_, fmt='%.18e', delimiter=' ', newline='\n', comments='# ')#np.save(fname+'.npy', np.array(w_))#

    return freq, reweighted_freq, w_


def inferring_h_with_sampling(root_sequence, path_to_tree, save_folder, w_GT=None, mu=600, error=0, lr=0.01, num_steps=500, start_from_emp=True, show_gradients=True):
    t=Tree(path_to_tree)
    num_leaves=len([leaf for leaf in t.iter_leaves()])

    L=len(root_sequence)
    q=21
    createFolder(save_folder)
    if w_GT is None:
        w_GT=randomW(L,q)
    fname=save_folder+"GTFelsenstein_"+str(num_leaves)+"_leaves_mu="+str(mu);np.savetxt(fname, np.array(w_GT), fmt='%.18e', delimiter=' ', newline='\n', comments='# ') #np.save(fname+'.npy', np.array(w_GT))#

    #initializing the root sequence
    t.get_tree_root().add_features(sequence=root_sequence)

    #sample on the tree with felsenstein
    mutate_felsenstein(t, mu, w_GT)

    #compute the empirical frequencies
    #fname="generated_MSAs_2025/sampledMSAFelsenstein_"+str(num_leaves)+"_leaves_mu="+str(mu)
    MSA=MSAFromLeafSequences(t, save_path="generated_MSAs_2025/sampledMSAFelsenstein_"+str(num_leaves)+"_leaves_mu="+str(mu))#;np.savetxt(fname, np.array(MSA, dtype=int), fmt='%.18e', delimiter=' ', newline='\n', comments='# ')
    
    freq=getFrequencyDistribution(MSA) #empirical frequencies from the MSA #redo it with the weighted frequencies
    reweighted_freq=getReweightedDistribution(MSA,save_folder+'/reweighted_freqs_'+str(num_leaves)+"_leaves_mu="+str(mu))
    
    fname=save_folder+"empfreqsFelsenstein_"+str(num_leaves)+"_leaves_mu="+str(mu);np.savetxt(fname, np.array(freq), fmt='%.18e', delimiter=' ', newline='\n', comments='# ') #np.save(fname+'.npy', np.array(freq))#
    print('mu', mu, 'empirical frequencies', freq)

    # Defining the initial conditions
    if start_from_emp:
        pseudocount=10**(-5)
        w_init=normalizePerRow(jnp.array(freq+pseudocount))
    else:   
        w_init=jnp.ones((L,q))/q

    h_init=jnp.log(w_init)
        
    # Compute the argmax for the likelihood (inferred frequencies) by gradient descent on the likelihood
    vmap_function = jax.vmap(optimize_h, in_axes=(None, 0, None, 0, None, None))

    w_, evo_grad, evo_likelihood = vmap_function(mu+error, h_init, t.get_tree_root(), jnp.arange(L), lr, num_steps)

    # Saving w_inferred and freq and the ground truth
    fname=save_folder+"inferred_freq"+str(num_leaves)+"_leaves_mu="+str(mu)+"_error="+str(error);np.savetxt(fname,  w_, fmt='%.18e', delimiter=' ', newline='\n', comments='# ')#np.save(fname+'.npy', np.array(w_))#
    
    if show_gradients:
        for site in range(L):
            plt.scatter([i for i in range(len(evo_likelihood[site]))],evo_likelihood[site], color='red', label=f'Neg log likelihood')
            plt.scatter([i for i in range(len(evo_grad[site]))],evo_grad[site], color='blue', label=f'Gradient')

            plt.xlabel('epochs')
            plt.legend()
            plt.title(f'site {site}')
            plt.show()

    return w_GT, freq, reweighted_freq, w_

def pearsonCorrsH(root_sequence, path_to_tree, sequences, mu_interval, dict_name, save_folder_dict, save_folder_freqs, lr=0.01, num_steps=500, error=0, start_from_emp=True):
    createFolder(save_folder_dict) 
    w_GT=getFrequencyDistribution(sequences)
    w_GT_reweighted=getReweightedDistribution(sequences)
    L=len(sequences[0])
    print('Sequence length is', L)

    dict_pearson={}
    for mu in mu_interval:
        w_GT, freq, reweighted_freq, w_= inferring_h_with_sampling(root_sequence, path_to_tree, save_folder_freqs, w_GT=w_GT, mu=mu, error=error, start_from_emp=start_from_emp, lr=lr, num_steps=num_steps)
        inferred_pearson_corr = np.corrcoef(w_GT.flatten(), w_.flatten())[0, 1]
        empirical_pearson_corr = np.corrcoef(w_GT.flatten(), freq.flatten())[0, 1]
        reweighted_empirical_pearson_corr = np.corrcoef(w_GT_reweighted.flatten(), reweighted_freq.flatten())[0, 1]

        dict_pearson[mu]=(inferred_pearson_corr, empirical_pearson_corr, reweighted_empirical_pearson_corr)

    # Save the dictionary
    with open(save_folder_dict+dict_name, 'wb') as file:
        pickle.dump(dict_pearson, file)
    print(f'Dictionary saved to {save_folder_dict+dict_name}')
    return dict_pearson

def launchFelsensteinAncestral(path_to_tree: str, path_to_leaves: str, save_name_anc: str, save_name_w: str, save_folder_for_ancestor: str, chosen_index=0, pseudocount=10**(-5), infer_w=False, lr=0.01, num_steps=500):
    """_summary_

    Args:
        path_to_tree (str): _description_
        path_to_leaves (str): _description_
        save_name_anc (str): _description_
        save_name_w (str): _description_
        save_folder_for_ancestor (str): _description_
        chosen_index (int, optional): _description_. Defaults to 0.
        pseudocount (_type_, optional): _description_. Defaults to 10**(-5).
        infer_w (bool, optional): _description_. Defaults to False.
        lr (float, optional): _description_. Defaults to 0.01.
        num_steps (int, optional): _description_. Defaults to 500.

    Returns:
        _type_: _description_
    
    Example usage:
        sequences_path='seq1_mu42.88.fa'
        path_to_tree='DBDtree_collapsed_noonlychild_midpointrooted_prunedsubtree301.nwk'

        proba_emp, ML_seq_emp=launchFelsensteinAncestral(path_to_tree, sequences_path, 'test_fels_prob_emp', root_dir, chosen_index=0, pseudocount=10**(-5), infer_w=False)

    On a complete folder of generated data:
        base_directory='3_seq_DBDtree_collapsed_noonlychild_midpointrooted_prunedsubtree301/'#'leo_data'
        filenames=get_all_file_paths(base_directory)
        path_to_tree='tree_structures/DBDtree_collapsed_noonlychild_midpointrooted_prunedsubtree301.nwk'

        clean_filenames=[]
        for file in filenames:
            if not '.ipynb_checkpoints' in file:
                if '.fa' in file:
                    clean_filenames.append(file)
        for file in clean_filenames:
            name=remove_charsequence(file.split('/')[-1], '.fa')
            print(name)
            #launchFelsensteinAncestral(path_to_tree, file, name+'_prob_emp', 'prob_distributions_leo_301/', chosen_index=0, pseudocount=10**(-5), infer_w=False)
            launchFelsensteinAncestral(path_to_tree, file, name+'_prob_w_inference', name, 'prob_distributions_leo_301/', chosen_index=0, pseudocount=10**(-5), infer_w=True)

    """
    
    # 1. Read francesco fasta path to leaves
    chosen_leaves_MSA, name_to_index_chosen, _ = read_fasta2(path_to_leaves)

    # 2. Read the tree
    t=Tree(path_to_tree)
    
    # 3. Fit mu from the data
    mu=fit_mu(t, path_to_leaves)
    print('Fitted mu is', mu)
    
    # 4. Reroot the tree if necessary
    if chosen_index!=0:
        path_to_new_tree=reroot(path_to_tree, chosen_index)
        t_outgroup=Tree(path_to_new_tree)

    else:
        t_outgroup=Tree(path_to_tree)


    # 5. Add the sequences to the leaves of the new tree
    for leaf in t_outgroup.iter_leaves():
        seq=chosen_leaves_MSA[name_to_index_chosen[leaf.name]]
        leaf.add_features(sequence=np.array(seq))

    # 6. Define w as the empirical frequencies of the leaves
    emp_freqs=getFrequencyDistribution(chosen_leaves_MSA) #using empirical frequencies as equilibriium distribution for now
    w=jnp.array(normalizePerRow(emp_freqs+pseudocount))
    print('Total sum of w ajusted with the pseudocount', jnp.sum(w))

    if infer_w:
        createFolder('inferred_w_for_ancestral_figs/')
        w=inferring_w_from_MSA(path_to_leaves, path_to_tree, 'inferred_w_for_leo_seqs/', save_name_w, lr=lr, num_steps=num_steps, start_from_emp=True)[2]#(path_to_leaves, path_to_tree, 'inferred_w'+path_to_leaves, save_folder='inferred_w_for_ancestral/')[2]

        #print('Inferred w versus empirical frequencies')
        plt.scatter(emp_freqs, w)
        plt.xlabel('Empirical frequencies')
        plt.ylabel('Inferred w')
        plt.title('Inferred w versus empirical frequencies')
        plt.savefig('inferred_w_for_ancestral_figs/'+save_name_w+'_scatter_plot.png')
        plt.show()

    # 7. Run Felsenstein
    L=len(chosen_leaves_MSA[0])
    site_range=jnp.arange(L)
    felsenstein_proba_distribution=jax.vmap(lambda site : FelsensteinAncestral(t_outgroup.get_tree_root(),  mu, w[site], site))(site_range)
    #check for NaNs

    # 8. Save it
    createFolder(save_folder_for_ancestor)
    fname=save_name_anc+'_felsenstein_proba_distribution_index_'+str(chosen_index)
    np.savetxt(save_folder_for_ancestor+fname, np.array(felsenstein_proba_distribution))

    # 9. Get the Max Likelihood Sequence
    ML_seq=get_maximum_likelihood_sequence(felsenstein_proba_distribution)
    print('Max likelihood sequence is', ML_seq)

    #save the Gibbs GT
    #fname='Gibbs_GT_index_'+str(chosen_index)
    #np.savetxt('felsenstein_data/'+fname, np.array(chosen_gibbs_seq))

    #save the leaves MSA GT
    #fname='MSA_GT_index_'+str(chosen_index)
    #np.savetxt('felsenstein_data/'+fname, np.array(chosen_leaves_MSA))
    return felsenstein_proba_distribution, ML_seq

def samplingFelsenstein(probabilities, T, M=1000):
    """
    Samples M sequences from a probability distribution using Boltzmann weighting.
    
    Parameters:
    probabilities (dict): A dictionary mapping sequences to their probabilities.
    T (float): Temperature parameter for Boltzmann weighting.
    M (int): Number of sequences to sample.
    
    Returns:
    list: A list of sampled sequences forming the MSA.
    """
    
    # Convert to energy: E = -log(P)
    E = -np.log(probabilities + 1e-10)  # Add small constant to avoid log(0)
    
    # Compute Boltzmann weights: exp(-E/T)
    boltzmann_weights = np.exp(-E / T)
    
    # Sample M sequences
    MSA = np.zeros((M,probabilities.shape[0]), dtype=int)
    for k in range(MSA.shape[1]):
        weights = to_numpy(boltzmann_weights[k])
        weights /= weights.sum()
        MSA[:, k] = np.random.choice(np.arange(21), size=M, p=weights)

        MSA[:,k] = np.random.choice(np.arange(21), size = M, p = boltzmann_weights[k]/np.sum(boltzmann_weights[k]))
    
    return MSA


def plot_felsenstein_sampling_with_temperature(probability_directory, T_range, GT_seqs:dict, Potts_parameters_path:str)->None:
    """
    Plots the energy vs. Hamming distance for Felsenstein sampling with different temperatures.
    
    Parameters:
    probability_directory (str): Directory containing probability distribution files.
    T_range (list): List of temperatures to sample.
    GT_seqs (dict): Dictionary mapping sequence names to their ground truth sequences.
    Keys should be 'GT_seq1', 'GT_seq2', 'GT_seq3', etc. and correspond the the GT sequences named in the probability directory. 
    Potts_parameters_path (str): Path to the Potts model parameters.
    
    Returns:
    None

    Plots the cloud of points for each temperature and saves the figures.
    """
    
    # Create a directory for saving figures
    createFolder('figures/')

    # Read Potts model parameters
    couplings, fields_ = readParametersFraZ(Potts_parameters_path)
    
    # Get all filenames in the specified directory
    filenames=get_all_file_paths(probability_directory)
    for file in filenames:
        for T in T_range:
            name=file.split('/')[-1].split('_')[0]+'_'+file.split('/')[-1].split('_')[1]
            print(name)
            anc_prob=np.loadtxt(file)
            ML_seq=get_maximum_likelihood_sequence(anc_prob)
            # Define ground truth sequence
            GT = eval(f'GT_{name.split("_")[0]}')  # Equivalent to GT_seq1, GT_seq2, GT_seq3
            # Generate MSA 
            MSA = create_MSA_profile(anc_prob, cardinal=301)
            # Sample M sequences with temperature T
            sampled_MSA = samplingFelsenstein(anc_prob, T=T)
            L=len(MSA[0])
            # Compute distances and energies
            fels_dists = [calculate_hamming_distance(s, GT)/L for s in MSA]
            sampled_dists = [calculate_hamming_distance(s, GT)/L for s in sampled_MSA]
            energy_fels = [energy(s, couplings, fields_) for s in MSA]
            energy_sampled = [energy(s, couplings, fields_) for s in sampled_MSA]
            ML_energy = energy(ML_seq, couplings, fields_)
            ML_dist = calculate_hamming_distance(ML_seq, GT)/L
            GT_energy = energy(GT, couplings, fields_)
            # Plot in the correct subplot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(energy_fels, fels_dists, color='blue', alpha=0.5, label='MSA')
            ax.scatter(energy_sampled, sampled_dists, color='red', alpha=0.5, label=f'Sampled MSA with T={T}')
            ax.scatter(ML_energy, ML_dist, color='black', marker='x', s=100, label='ML Sequence')
            # Add a line for the GT sequence
            ax.axvline(x=GT_energy, color='green', linestyle='--', label='GT Sequence')
            # Titles and labels
            ax.set_title(f'{name} | T={T}')
            ax.set_xlabel('Energy')
            ax.set_ylabel('Hamming Distance')
            ax.legend()
            plt.tight_layout()
            plt.savefig(f'figures/energy_hamming_{name}_T={T}.png')
            plt.show()


"""

def optimize_h(mu, h, t, site, lr=0.1, num_steps=500):
    @partial(jax.jit, static_argnames=('t',))
    def step(mu, h, t, site, lr):
        # Compute gradients
        grad_h = jax.grad(objective, argnums=0)(h, t.get_tree_root(), mu, site)
        
        # Update h
        h = h - lr * grad_h
        
        return h 

    #evo_likelihood = jnp.zeros(num_steps)
    #evo_grad = jnp.zeros(num_steps)

    for i in range(num_steps):
        h = step(mu, h, t, site, lr)

        # Track progress
        #evo_grad = evo_grad.at[i].set(jnp.linalg.norm(jax.grad(neg_log_likelihood_h, argnums=2)(t.get_tree_root(), mu, h, site)))
        #evo_likelihood = evo_likelihood.at[i].set(jnp.linalg.norm(neg_log_likelihood_h(t.get_tree_root(), mu, h, site)))
    w=softmax_projection(h)
    return w#, evo_likelihood, evo_grad
    

def inferring_w_from_MSA(sequences_path, path_to_tree, save_folder, save_name, lr=0.1, num_steps=500, start_from_emp=True):
    # Reading the parameters and the tree
    t=Tree(path_to_tree)
    MSA, name_to_index=read_fasta2(sequences_path)
    num_leaves=len([leaf for leaf in t.iter_leaves()])
    L=len(MSA[0])
    q=21
    createFolder(save_folder)

    # Putting the sequences on the leaves 
    for leaf in t.iter_leaves():
        leaf.add_features(sequence=np.array(MSA[name_to_index[leaf.name]]))

    mu=fit_mu(t, sequences_path)
    print(mu)
    
    #compute the empirical frequencies    
    freq=getFrequencyDistribution(MSA) #empirical frequencies from the MSA #redo it with the weighted frequencies
    reweighted_freq=getReweightedDistribution(MSA,save_folder+'/reweighted_freqs_'+str(num_leaves)+"_leaves_inferred_mu="+str(mu)+save_name)
    
    fname=save_folder+"empfreqsFelsenstein_"+str(num_leaves)+"_leaves_inferred_mu="+str(mu)+save_name;np.savetxt(fname, np.array(freq), fmt='%.18e', delimiter=' ', newline='\n', comments='# ') #np.save(fname+'.npy', np.array(freq))#
    print('mu', mu, 'empirical frequencies', freq)

    # Defining the initial conditions
    if start_from_emp:
        pseudocount=10**(-5)
        w_init=normalizePerRow(jnp.array(freq+pseudocount))
    else:   
        w_init=jnp.ones((L,q))/q

    h_init=jnp.log(w_init)
        
    # Compute the argmax for the likelihood (inferred frequencies) by gradient descent on the likelihood
    vmap_function = jax.vmap(optimize_h, in_axes=(None, 0, None, 0, None, None))

    #h_, evo_likelihood, evo_grad=vmap_function(mu+error, h_init, t.get_tree_root(), jnp.arange(L))
    w_=vmap_function(mu, h_init, t.get_tree_root(), jnp.arange(L), lr, num_steps)

    # Get w_ from h_
    #w_=softmax_projection(h_)
    # Saving w_inferred and freq and the ground truth
    fname=save_folder+"inferred_freq"+str(num_leaves)+"_leaves_inferred_mu="+str(mu)+save_name;np.savetxt(fname,  w_, fmt='%.18e', delimiter=' ', newline='\n', comments='# ')#np.save(fname+'.npy', np.array(w_))#

    return freq, reweighted_freq, w_
"""
