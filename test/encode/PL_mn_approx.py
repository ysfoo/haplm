"""
Perform partition ligation for the 100 datasets simulated based on the 1000 Genomes Project.
LC-Sampling is used as the frequency estimation subroutine for partition ligation.

Input files required:
- Pool size with observed allele counts for each pool
  @ '../../data/encode/psize{pool_size}_m{n_markers}_id{ds_idx}.data'

Output files produced:
- Output of partition ligation; each row consists of integers that are binary encodings of the
  resulting haplotypes for one dataset
  @ '../../data/encode/PL_mn_approx.txt'
"""

import numpy as np
from haplm.hap_util import PL_mn_approx
from haplm.lm_dist import find_4ti2_prefix
from sim_data import parse_sim_data
import pulp
import numpyro

# init for other libraries
solver = pulp.apis.SCIP_CMD(msg=False)
prefix_4ti2 = find_4ti2_prefix()

# MCMC parameters
chains = 5
cores = 5
n_burnin = 200
n_sample = 300
numpyro.set_host_device_count(cores)

# data parameters
n_datasets = 100
n_pools = 20
n_markers = 8
H = 2**n_markers # number of haplotypes
pool_size = 50

trials = 5 # number of random initialisations for AEML

# wildtype and haplotypes with only minor allele at one marker only are included by default
def inithaps_fn(n_markers):
    return []
    #return np.vstack([np.zeros((1, n_markers), dtype=int), np.eye(n_markers, dtype=int)])

failed = [] # 1-based indices of datasets for which PL-AEML failed
with open('../../data/encode/PL_mn_approx.txt', 'w') as fp:
    for ds_idx in range(1, n_datasets+1):
        fn_prefix = f'../../data/encode/psize{pool_size}_m{n_markers}_id{ds_idx}'
        ns, ys = parse_sim_data(fn_prefix+'.data')

        print(f'PL for dataset {ds_idx}')
        haps = PL_mn_approx(ns, ys, n_markers, 
                            n_sample, n_burnin, chains,
                            solver, prefix_4ti2,
                            0.005, 40,
                            inithaps_fn=inithaps_fn)
        if haps is None:
            failed.append(ds_idx)
            fp.write('\n')
            continue

        fp.write(' '.join([str(sum(1 << i for i, h in enumerate(hap) if h)) for hap in haps]))
        fp.write('\n')
        fp.flush()

print('Datasets that failed:', failed)