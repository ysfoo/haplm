"""
Perform MCMC-Exact on synthetic datasets.

Input files required:
- Pool size with observed allele counts for each pool
  @ '../../data/sim-study/psize{pool_size}_m{n_markers}_id{ds_idx}.data'

Output files produced:
- InferenceData output of MCMC-Exact
  @ '../../data/sim-study/psize{pool_size}_m{n_markers}_id{ds_idx}_exact.netcdf'
"""

from time import time

from haplm.lm_dist import LatentMult, find_4ti2_prefix
from haplm.hap_util import mat_by_marker
from haplm.lm_inference import latent_mult_mcmc
from sim_data import parse_sim_data

import numpy as np
import pulp
import numpyro
# import arviz as az

# init for other libraries
solver = pulp.apis.SCIP_CMD(msg=False)
prefix_4ti2 = find_4ti2_prefix()

# data parameters
n_datasets = 5
n_pools = 20
n_markers = 3
H = 2**n_markers # number of haplotypes
pool_sizes = range(20, 101, 20)

# MCMC parameters
chains = 5
cores = 5
n_burnin = 500
n_sample = 500
numpyro.set_host_device_count(cores)

amat = mat_by_marker(n_markers)

for pool_size in pool_sizes:
	print(f'Pool size = {pool_size}')
	print('-'*15)
	for ds_idx in range(1, n_datasets+1):
		fn_prefix = f'../../data/sim-study/psize{pool_size}_m{n_markers}_id{ds_idx}'

		ns, ys = parse_sim_data(fn_prefix+'.data')

		print(f'Pre-processing set {ds_idx}', flush=True)
		lm_list = []
		t = time()
		for n, y in zip(ns, ys):			
			lm = LatentMult(amat, y, n, '../../4ti2-files/sim-study-exact',
				            solver, prefix_4ti2, enum_sols=True)
			lm_list.append(lm)
		pre_time = time() - t
		print(pre_time, flush=True)

		print(f'MCMC for set {ds_idx}')
		t = time()
		idata = latent_mult_mcmc(lm_list, H, n_sample, n_burnin, ['exact']*n_pools,
			                     jaxify=False, chains=chains, random_seed=ds_idx^pool_size)
		mcmc_time = time() - t

		idata.attrs['preprocess_time'] = pre_time
		idata.attrs['mcmc_walltime'] = mcmc_time

		# ess = az.ess(idata, var_names=['p'])['p'].values
		# print(ess.min())

		idata.to_netcdf(fn_prefix+'_exact.netcdf')
	print()






