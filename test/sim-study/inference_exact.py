import numpy as np
import pulp

from time import time
import os

from haplm.lm_dist import LatentMult, find_4ti2_prefix, mat_by_marker
from haplm.lm_inference import latent_mult_mcmc
from sim_data import gen_sim_data, parse_sim_data

# init for other libraries
solver = pulp.apis.SCIP_CMD(msg=False)
prefix_4ti2 = find_4ti2_prefix()

# parameters for data simulation
n_datasets = 5
n_pools = 20
n_markers = 3
H = 2**n_markers # number of haplotypes
pool_sizes = np.arange(10, 31, 5)
alphas = np.ones(H)*0.4

amat = mat_by_marker(n_markers)

# MCMC parameters
cores = 5
chains = 5
n_burnin = 1000
n_sample = 5000

for pool_size in pool_sizes:
	print(f'Pool size = {pool_size}')
	print('-'*15)
	for ds_idx in range(1, n_datasets+1):
		fn_prefix = f'../../data/sim-study/psize{pool_size}_m{n_markers}_id{ds_idx}'

		# if data has not been generated
		if not os.path.isfile(fn_prefix+'.data'):
			gen_sim_data(n_pools, n_markers, pool_size, alphas, ds_idx, fn_prefix)

		ns, ys = parse_sim_data(fn_prefix+'.data')

		print(f'Pre-processing set {ds_idx}')
		lm_list = []
		t = time()
		for n, y in zip(ns, ys):			
			lm = LatentMult(amat, y, n, '../../4ti2-files/sim-study-markov',
				            solver, prefix_4ti2, merge=False, walk_len=500, num_pts=5)
			lm.enum_sols('../../4ti2-files/sim-study-zsolve')
			lm_list.append(lm)
		pre_time = time() - t

		print(f'MCMC for set {ds_idx}')
		t = time()
		idata = latent_mult_mcmc(lm_list, H, n_sample, n_burnin, ['exact']*n_pools,
			                           chains=chains, cores=cores, seeds=range(chains))
		mcmc_time = time() - t

		idata.sample_stats.attrs['preprocess_time'] = pre_time
		idata.sample_stats.attrs['mcmc_walltime'] = mcmc_time

		idata.to_netcdf(fn_prefix+'_exact.netcdf')
	print()






