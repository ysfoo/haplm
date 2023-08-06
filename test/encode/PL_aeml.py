"""
Perform partition ligation for the 100 datasets simulated based on the 1000 Genomes Project. AEML is
used as the frequency estimation subroutine for partition ligation.

Input files required:
- Pool size with observed allele counts for each pool
  @ '../../data/encode/psize{pool_size}_m{n_markers}_id{ds_idx}.data'

Output files produced:
- Output of partition ligation; each row consists of integers that are binary encodings of the
  resulting haplotypes for one dataset
  @ '../../data/encode/PL_aeml.txt'
"""

import numpy as np
from haplm.hap_util import PL_aeml
from sim_data import parse_sim_data
import pulp


# location of AEML program
aeml_dir = '../../hippo_aeml/'

solver = pulp.apis.SCIP_CMD(msg=False)

# data parameters
n_datasets = 100
n_pools = 20
n_markers = 8
H = 2**n_markers # number of haplotypes
pool_size = 50

trials = 5 # number of random initialisations for AEML

# wildtype and haplotypes with only minor allele at one marker only are included by default
def inithaps_fn(n_markers):
    return np.vstack([np.zeros((1, n_markers), dtype=int), np.eye(n_markers, dtype=int)])

failed = [] # 1-based indices of datasets for which PL-AEML failed
with open('../../data/encode/PL_aeml.txt', 'w') as fp:
    for ds_idx in range(1, n_datasets+1):
        fn_prefix = f'../../data/encode/psize{pool_size}_m{n_markers}_id{ds_idx}'
        ns, ys = parse_sim_data(fn_prefix+'.data')

        print(f'PL for dataset {ds_idx}')
        haps, convg = PL_aeml(ns, ys, n_markers, 
                              hap_fn='encode_haplist.txt', aeml_dir=aeml_dir, 
                              solver=solver, thres=0.005, maxhaps=40, 
                              trials=trials, inithaps_fn=inithaps_fn)
        if haps is None:
            failed.append(ds_idx)
            fp.write('\n')
            continue
        print(f'{len(haps)} haplotypes,', 'AEML converged' if convg else 'AEML did not converge')

        fp.write(' '.join([str(sum(1 << i for i, h in enumerate(hap) if h)) for hap in haps]))
        fp.write('\n')

print('Datasets that failed:', failed)