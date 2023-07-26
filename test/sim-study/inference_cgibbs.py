"""
Perform LC-Sampling on synthetic datasets.

Input files required:
- Pool size with observed allele counts for each pool
  @ '../../data/sim-study/psize{pool_size}_m{n_markers}_id{ds_idx}.data'

Output files produced:
- InferenceData output of LC-Sampling
  @ '../../data/sim-study/psize{pool_size}_m{n_markers}_id{ds_idx}_cgibbs.netcdf'
"""

import numpy as np
# import arviz as az
import pulp

from time import time
# import pickle as pkl

from haplm.lm_dist import LatentMult, find_4ti2_prefix
from haplm.hap_util import mat_by_marker
from haplm.lm_inference import latent_mult_mcmc_cgibbs
from sim_data import parse_sim_data

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

amat = mat_by_marker(n_markers)

for pool_size in pool_sizes:
	print(f'Pool size = {pool_size}')
	print('-'*15)
	for ds_idx in range(1, n_datasets+1):
		fn_prefix = f'../../data/sim-study/psize{pool_size}_m{n_markers}_id{ds_idx}'

		ns, ys = parse_sim_data(fn_prefix+'.data')

		print(f'Pre-processing set {ds_idx}')
		lm_list = []
		t = time()
		for n, y in zip(ns, ys):			
			lm = LatentMult(amat, y, n, '../../4ti2-files/sim-study-cgibbs',
				            solver, prefix_4ti2, num_pts=5, walk_len=500)
			lm.compute_basis(markov=True)
			lm_list.append(lm)
		n_sum = sum(lm.n_var for lm in lm_list if lm.idx_var.size)
		pre_time = time() - t

		t = time()
		idata = latent_mult_mcmc_cgibbs(lm_list, H, n_sample, n_burnin, chains,
		                                cyc_len=5*n_sum, alphas=np.ones(H),
		                                cores=cores, random_seed=ds_idx^pool_size)
		mcmc_time = time() - t

		idata.attrs['preprocess_time'] = pre_time
		idata.attrs['mcmc_walltime'] = mcmc_time

		idata.to_netcdf(fn_prefix+'_cgibbs.netcdf')

		# print(min(az.ess(idata, var_names=['p'])['p'].values))

		# with open(fn_prefix+'_lm_list.pkl', 'wb') as fp:
			# pkl.dump(lm_list, fp)
	print()








