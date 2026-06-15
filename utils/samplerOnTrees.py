import numpy as np
from ete3 import Tree
from utils.PottsEnergies import energy_site_MCMC, energy_site_gibbs
from utils.utils import createFolder

def mutate_felsenstein(tt: Tree, mu: float, w: np.array)->None: #Check thisss
    """
    The Felsenstein propagator on a tree. Modifies the tree tt given as input.
    Inputs: 
        tt: tree
        mu: mutation rate
        w: profile, size L*21
    Output: 
        None.
    """
    L = len(tt.sequence)
    if not tt.is_root():
        seq = np.copy(tt.sequence)
        dist = tt.dist
        #print(dist)
        factor_mutations = np.exp(-mu*dist)
        for i in range(L):
            #we sample each site independently
            p = factor_mutations*(np.eye(21)[seq[i]])+(1-factor_mutations)*w[i]
            #print(p.sum())
            p = np.asarray(p).astype('float64')
            p = p / np.sum(p) 
            seq[i] = np.random.choice(21,p = p/p.sum())

        tt.add_features(sequence=seq)
    for child in tt.children:
        child.sequence = tt.sequence
        mutate_felsenstein(child, mu, w)

def mutate_metropolis(tt: Tree, mu: float, fields_:dict, couplings:dict, q=21)->None:
    """
    Samples sequences on the tree using Metropolis MCMC, going from the root and down. Modifies the tree tt given as input.

    Inputs: 
    tt: tree
    mu: mutation rate
    fields_, couplings: dictionaries of parameters
    Output: None
    """
    L = len(tt.sequence)
    if not tt.is_root():
        seq = np.copy(tt.sequence)
        dist = tt.dist
        factor_nb_mutations = round(L * mu * dist)+np.random.randint(0,2)
        for _ in range(factor_nb_mutations):
            ra = int(np.random.randint(L))
            ra2 = np.random.randint(q)

            if np.exp(energy_site_MCMC(seq,ra,ra2,couplings,fields_)) > np.random.random():
                seq[ra] = ra2
        tt.add_features(sequence=seq)
    for child in tt.children:
        child.sequence = tt.sequence
        mutate_metropolis(child, mu, fields_, couplings)

def mutate_gibbs(tt: Tree, mu: float, fields_: dict, couplings: dict, q=21)->None:
    """
    Samples sequences on the tree using Gibbs MCMC, going from the root and down. Modifies the tree tt given as input.
    """
    L = len(tt.sequence)
    if not tt.is_root():
        seq = np.copy(tt.sequence)
        dist = tt.dist
        #factor_nb_mutations = round(L * mu * dist)+np.random.randint(0,2)
        p = L * mu * dist-round(L * mu * dist) 
        factor_nb_mutations = round(L * mu * dist)+(np.random.rand()<p)        
        for _ in range(factor_nb_mutations):
            ra = int(np.random.randint(L))
            seq_copy = np.copy(seq)
            ener_ = energy_site_gibbs(seq_copy,ra, couplings, fields_)
            if hasattr(ener_, "detach"):
                ener_ = ener_.detach().cpu().numpy()
            else:
                ener_ = np.asarray(ener_)
            ener_softmax= np.exp(-ener_)/np.exp(-ener_).sum()
            selected_mutation = np.random.choice(q,p =ener_softmax )
            seq[ra] = selected_mutation

        tt.add_features(sequence=seq)

    for child in tt.children:
        child.sequence = tt.sequence
        mutate_gibbs(child, mu, fields_, couplings)

def get_gibbs_sampled_MSA_node(path_to_tree: str, #### maybe useless
                               mu: float,
                               node_name: str,
                               root_sequence: np.array, 
                               fields: dict, couplings: dict, 
                               chosen_index=5,
                               size_ = 10, 
                               save=True
                               ) -> np.array:
  """Inputs: a tree and a chosen node index, names the node 'chosen_node'. 
  With the parameters on the Potts model and the given root_sequence, samples on the tree, stores the sequence generated on this node.
  Does this size_ times. 
  Returns the MSA of all the sampled sequences."""
  seqs_ = []
  for j in range(size_):
    t=Tree(path_to_tree)
    #Pick a node indexed as node 5 and name it 'chosen_node'
    index=0
    for node in t.traverse("levelorder"): 
      if index==chosen_index:
        node.name='chosen_node'
        #print(node.name)
      index+=1
    t.add_features(sequence= root_sequence)
    mutate_gibbs(t, mu, fields, couplings)
   #print(t_new.search_nodes(name=node_name)[0].sequence)
    seqs_.append(t.search_nodes(name=node_name)[0].sequence)
    if j%100 == 0:
      print(j)
      print(seqs_[j])

  #save the tree with the new name
  path_to_new_tree='named_nodes_'+path_to_tree
  t.write(outfile=path_to_new_tree, format=1)
  if save:
      # Save the array to a text file
      createFolder('FelsensteinMSAs/')
      np.savetxt('FelsensteinMSAs/MSA_chosen_index_'+str(chosen_index)+'_'+path_to_tree+'.txt', np.array(seqs_), fmt='%d')
  return seqs_, path_to_new_tree