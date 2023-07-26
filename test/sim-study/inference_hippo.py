"""
Perform HIPPO on synthetic datasets.

Input files required:
- Pool size with observed allele counts for each pool
  @ '../../data/sim-study/psize{pool_size}_m{n_markers}_id{ds_idx}.data'

Output files produced:
- InferenceData output of HIPPO
  @ '../../data/sim-study/psize{pool_size}_m{n_markers}_id{ds_idx}_hippo.netcdf'
"""

import numpy as np

from time import time

from haplm.hap_util import mat_by_marker
from haplm.hippo_aeml import run_hippo
from sim_data import parse_sim_data

# location of HIPPO program
hippo_dir = '../../hippo_aeml/'

# data parameters
n_datasets = 5
n_pools = 20
n_markers = 3
H = 2**n_markers # number of haplotypes
pool_sizes = range(20, 101, 20)

# MCMC parameters
chains = 5
n_burnin = 50000
n_sample = 450000
thin = 900

for pool_size in pool_sizes:
	print(f'Pool size = {pool_size}')
	print('-'*15)
	for ds_idx in range(1, n_datasets+1):
		fn_prefix = f'../../data/sim-study/psize{pool_size}_m{n_markers}_id{ds_idx}'

		ns, ys = parse_sim_data(fn_prefix+'.data')
		print(f'MCMC for set {ds_idx}: ', end='')
		t = time()
		idata = run_hippo(ns, ys, n_sample, n_burnin, hippo_dir, 
		                  thin=thin, chains=chains, seed=ds_idx^pool_size)
		mcmc_time = time() - t
		idata.attrs['mcmc_walltime'] = mcmc_time
		idata.to_netcdf(fn_prefix+'_hippo.netcdf')
	print()