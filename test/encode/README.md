This example compares the proposed methods with HIPPO and AEML for data simulated based on real data from the 1000 Genomes Project. The data consists of [haplotype samples from 95 unrelated individuals of the CEU population](https://www.internationalgenome.org/data-portal/population/CEU) (Utah residents with ancestry from Northern and Western Europe) for the region ENm010 on chromosome 7. Based on this data, we then simulate 100 datasets of allele counts of 8 markers at a time. Since the number of possible haplotypes is large (256), we use partition ligation to specify a list of input haplotypes for all methods except HIPPO. MCMC-Approx is used as the frequency estimation subroutine to partition ligation. For HIPPO, this list is instead used as initialisation for the list of input haplotypes, as this list will be subsequently sampled during the MCMC procedure. Note that MCMC-Exact is not used as the number of possible latent counts is in some cases too large to enumerate within reasonable time.

## Data files
The alleles of the 95 individuals are recorded in `7_26917902-27417901.ped`. Each row corresponds to one individuals, with the first six entries corresponding to non-genetic data, and the remaining entries are pairs of alleles (as humans are diploid) at various markers. This collection is considered as 190 haplotypes to sample from to simulate the datasets. The genomic location of the markers recorded in `7_26917902-27417901.info`.

## Steps to reproduce results from the paper
1. Create the directory `data/encode/` from the root directory of this repository.
2. From this directory, run the following lines in Python to simulate data:
```python
from sim_data import gen_sim_data
gen_sim_data(n_pools=20, n_markers=8, pool_size=50, n_datasets=100, data_dir='../../data/encode/')
```
3. Run the partition ligation script `PL_mn_approx.py` to determine lists of input haplotypes.
4. Perform inference with HIPPO, AEML, MCMC-Approx, and LC-Sampling with the scripts `inference_hippo.py`, `inference_aeml.py`, `inference_mn_approx.py`, and `inference_cgibbs.py` respectively.
5. The results are plotted in `results.ipynb`.

To avoid a long runtime, you may wish to change the number of datasets throughout all scripts to a small number (e.g. 5).