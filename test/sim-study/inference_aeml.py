"""
Perform AEML on synthetic datasets.

Input files required:
- Pool size with observed allele counts for each pool
  @ '../../data/sim-study/psize{pool_size}_m{n_markers}_id{ds_idx}.data'

Output files produced:
- Dictionary output of AEML 
  @ '../../data/sim-study/psize{pool_size}_m{n_markers}_id{ds_idx}_aeml.pkl'
"""

import numpy as np

import pickle as pkl

from haplm.hap_util import mat_by_marker
from haplm.hippo_aeml import run_AEML
from sim_data import parse_sim_data

# location of AEML program
aeml_dir = '../../hippo_aeml/'

# data parameters
n_datasets = 5
n_pools = 20
n_markers = 3
H = 2**n_markers # number of haplotypes
pool_sizes = range(20, 101, 20)

trials = 10 # number of random initialisations for AEML

for pool_size in pool_sizes:
	print(f'Pool size = {pool_size}')
	print('-'*15)
	for ds_idx in range(1, n_datasets+1):
		fn_prefix = f'../../data/sim-study/psize{pool_size}_m{n_markers}_id{ds_idx}'

		ns, ys = parse_sim_data(fn_prefix+'.data')

		print(f'AEML for set {ds_idx}: ', end='')
		aeml_out = run_AEML(ns, [y[:4] for y in ys], aeml_dir, seed=ds_idx^pool_size, stab=1e-9)
		assert aeml_out is not None

		with open(f'{fn_prefix}_aeml.pkl', 'wb') as fp:
			pkl.dump(aeml_out, fp)
	print()





