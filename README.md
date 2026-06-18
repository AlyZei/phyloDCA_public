# phyloDCA_public

Public repository for *"Towards coevolution-aware ancestral sequence reconstruction"*: https://www.biorxiv.org/content/10.64898/2026.06.08.731024v1

## Abstract

Ancestral sequence reconstruction (ASR) is a powerful approach for studying molecular evolution and the emergence of protein function. Yet most ASR methods assume that sites evolve independently, neglecting the epistatic constraints that shape protein structure, stability, and function. This simplification affects both ancestral inference and its evaluation: maximum-a-posteriori reconstructions may over-concentrate probability into a single over-idealized sequence, whereas independent posterior sampling can generate implausible or poorly functional ancestors. Here, we introduce a coevolution-aware ASR framework that combines standard phylogenetic inference with Direct Coupling Analysis (DCA), thereby preserving site-wise ancestral uncertainty while enforcing residue–residue constraints learned from extant protein families. To benchmark the method, we develop a controlled forward-evolution framework based on a DCA evolutionary sampler, allowing reconstructed ancestors to be compared with known ground-truth sequences generated under realistic epistatic constraints. Applied to β-lactamases and DNA-binding domains, the approach improves reconstruction when ancestral states are epistatically constrained, and yields ensembles of candidate ancestors that are both phylogenetically consistent and statistically compatible with natural protein families. This framework bridges the gap between single-sequence MAP reconstruction and unconstrained posterior sampling, providing a practical route toward ancestral reconstructions that better reflect the coupled nature of protein evolution.

## Getting started

This repository contains a complete, launchable example of the coevolution-aware ASR workflow described in the paper. The method combines phylogenetic ancestral sequence reconstruction with Direct Coupling Analysis (DCA), so ancestral candidates preserve site-wise ASR uncertainty while being reshuffled and ranked under residue-residue coevolutionary constraints learned from extant sequences.

Everything required to run the example is included:

- [utils/](utils/) contains the helper functions for loading data, running ASR, sampling posterior sequences, applying DCA-guided MCMC reshuffling, scoring candidates, and exporting FASTA files.
- [data_betaLac/](data_betaLac/) contains the bundled beta-lactamase data, including extant alignments, DCA/Potts parameters, and simulated sequences used by the example workflow.
- [example_usage_simple.ipynb](example_usage_simple.ipynb) shows how to launch the full workflow from the provided utilities and data.

No external data download is required: clone the repository, open the example notebook, and run the cells to generate coevolution-aware ancestral candidate sequences.

See [DOCUMENTATION.md](DOCUMENTATION.md) for full API reference of the helper functions used in the example notebook.

## Building an alignment from UniProt with HMMER

For a new protein family, start from a small trusted seed alignment and use HMMER to find homologs in UniProt. A typical workflow is:

```bash
# 1. Convert the seed alignment from FASTA to Stockholm.
esl-reformat stockholm family_seed.fasta > family_seed.sto

# 2. Build a full-length profile HMM from the seed alignment.
# --symfrac 0 makes every seed-alignment column part of the match model.
hmmbuild --amino --symfrac 0 seed_model_complete.hmm family_seed.sto

# 3. Search UniProt with the complete seed model.
# The -A output stores the aligned hits in Stockholm format.
hmmsearch \
  --cpu 8 \
  -A results_all_uniprot.sto \
  -E 1e-5 \
  --tblout family_hits.tbl \
  --domtblout family_domains.tbl \
  seed_model_complete.hmm \
  uniprot.fasta

# 4. Convert the Stockholm hit alignment to FASTA.
awk 'NF==2{
    split($0,v,"");
    if (v[1]!="#") {
        n[$1]++;
        seq[$1,n[$1]]=$2
    }
}
END{
    for (i in n) {
        printf(">%s\n", i);
        for (j=1; j<=n[i]; j++) printf("%s", seq[i,j]);
        printf("\n")
    }
}' results_all_uniprot.sto > family_alignment.fasta

# 5. Remove HMMER insertion states.
# This keeps gaps and uppercase match-state residues, and drops lowercase inserts.
awk '{
    split($0,v,"");
    if (v[1]==">") {
        if (NR>1) printf("\n");
        printf("%s\n", $0);
    }
    else {
        for (i=1; i<=length($0); i++)
            if (v[i]=="-" || v[i] != tolower(v[i]))
                printf("%c", v[i]);
    }
}
END{printf("\n")}' family_alignment.fasta > family_alignment_noinsert.fasta

# 6. Remove duplicate aligned sequences, keeping the first occurrence.
awk 'BEGIN { count = 0 }
     NR % 2 { name = $1 }
     !(NR % 2) {
        if (!printed[$1]) {
           printed[$1] = 1;
           print name;
           print $1;
        }
     }' family_alignment_noinsert.fasta > family_alignment_nodup.fasta

# 7. Remove sequences with at least 20% gaps or X characters.
awk 'NR%2{ name = $0 }
     !(NR%2){
         split($1,v,"");
         ngap = 0;
         for (i=1; i<=length($1); i++) ngap += (v[i]=="-" || v[i]=="X");
         if (ngap < 0.2*length($1)){
            print name;
            print $0;
         }
      }' family_alignment_nodup.fasta > family_alignment_nodup_max20gaps.fasta
```

The resulting `family_alignment_nodup_max20gaps.fasta` can be used as the extant MSA for phyloDCA. For large searches, UniRef90/UniRef50 or a taxonomically filtered UniProt FASTA is often easier to handle than the full UniProtKB database. Keep the same sequence headers when building the phylogenetic tree, because the ASR utilities expect tree leaf names to match the MSA headers.

## Inferring DCA parameters with adabmDCA

The DCA/Potts parameters used by phyloDCA can be inferred from the family alignment with [spqb/adabmDCA](https://github.com/spqb/adabmDCA). The adabmDCA project provides a flexible Direct Coupling Analysis package, links to a Colab tutorial notebook, and supports several training routines, including fully connected `bmDCA`, sparse `eaDCA`, and pruning-based `edDCA`.

A practical workflow is:

1. Open the adabmDCA tutorial notebook from the [spqb/adabmDCA](https://github.com/spqb/adabmDCA) README, or install the Python implementation locally with `pip install adabmDCA`.
2. Load the cleaned family MSA produced above, preferably after removing highly gapped columns/sequences and near-duplicate sequences.
3. Train a Potts/DCA model on that MSA. For phyloDCA, the standard choice is a fully connected `bmDCA` model unless you specifically want a sparse model.
4. Save/export the learned fields and couplings with the adabmDCA parameter writer.
5. Place the exported parameter file in your dataset folder, for example `data_myFamily/Parameters.dat`, and load it with `utils.PottsEnergies.read_potts_parameters_proteins()`.

## Given the MSA and DCA parameters

Once you have generated your own MSA and DCA parameters, simply follow the workflow of [example_usage_simple.ipynb](example_usage_simple.ipynb). It shows the user how to launch the full workflow from the provided utilities, given any MSA and DCA parameters. Simply replace the file paths to the betalactamase MSA and DCA parameters by those of your own family of interest, and run the notebook! 

## Citation

If you use phyloDCA, please cite our preprint: https://www.biorxiv.org/content/10.64898/2026.06.08.731024v1
