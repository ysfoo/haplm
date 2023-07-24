This example compares the proposed methods with HIPPO and AEML, with data simulated based on real data from the 1000 Genomes Project. The data consists of 190 haplotype samples of the CEU population (Utah residents with ancestry from Northern and Western Europe) for the region ENm010 on chromosome 7. Based on this data, we then simulate datasets of allele counts of 8 markers at a time. Since the number of possible haplotypes is large (256), we use partition ligation to specify a list of input haplotypes, which is used by all methods except HIPPO. This is because the key feature of HIPPO is that it samples the list of input haplotypes as part of its MCMC procedure. 

Steps to reproduce results:
1. Create the directory `data/encode/` from the root directory of this repository.
2. From this directory, run the following lines in Python:
```python
from sim_data import gen_sim_data
gen_sim_data(n_pools=20, n_markers=8, pool_size=50, n_datasets=100, data_dir='../../data/encode/')
```
3. more lines