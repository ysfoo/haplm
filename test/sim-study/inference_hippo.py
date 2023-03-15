import numpy as np
import arviz as az
import xarray

from time import time
import os
import subprocess as sp

from haplm.hap_utils import mat_by_marker
from haplm.hippo_aeml import run_hippo
from sim_data import gen_sim_data, parse_sim_data

# location of hippo_aeml
hippo_dir = '../../hippo_aeml/'

# parameters for data simulation
n_datasets = 5
n_pools = 20
n_markers = 3
H = 2**n_markers # number of haplotypes
pool_sizes = np.arange(20, 101, 20)
alphas = np.ones(H)*0.4

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

		# if data has not been generated
		if not os.path.isfile(fn_prefix+'.data'):
			gen_sim_data(n_pools, n_markers, pool_size, alphas, ds_idx, fn_prefix)

		ns, ys = parse_sim_data(fn_prefix+'.data')
		print(f'MCMC for set {ds_idx}: ', end='')
		idata = run_hippo(ns, ys, n_sample, n_burnin, hippo_dir, thin=thin, chains=chains)
		idata.to_netcdf(fn_prefix+'_hippo.netcdf')
	print()