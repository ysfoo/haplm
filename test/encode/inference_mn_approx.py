"""
Perform MCMC-Approx on datasets based on the 1000 Genomes Project.

Input files required:
- Output of partition ligation
  @ '../../data/encode/PL.txt'
- Pool size with observed allele counts for each pool
  @ '../../data/encode/psize{pool_size}_m{n_markers}_id{ds_idx}.data'

Output files produced:
- InferenceData output of MCMC-Approx
  @ '../../data/encode/psize{pool_size}_m{n_markers}_id{ds_idx}_mn_approx.netcdf'
"""

import numpy as np
import pulp
# import arviz as az

from time import time
import traceback

from haplm.lm_dist import LatentMult, find_4ti2_prefix
from haplm.lm_inference import latent_mult_mcmc
from sim_data import parse_sim_data

import jax.numpy as jnp
import numpyro

# init for other libraries
solver = pulp.apis.SCIP_CMD(msg=False)
prefix_4ti2 = find_4ti2_prefix()

# data parameters
n_datasets = 100
n_pools = 20
n_markers = 8
pool_size = 50

# MCMC parameters
chains = 5
cores = 5
n_burnin = 500
n_sample = 500
numpyro.set_host_device_count(cores)

hap_lists = []
amats = []
with open('../../data/encode/PL_mn_approx.txt') as fp:
    for line in fp:
        hap_list = [int(x) for x in line.split()]
        hap_lists.append(hap_list)
        amats.append(np.array([[(hnum >> m) & 1 for hnum in hap_list] for m in range(n_markers)]))

# print(np.mean([len(l) for l in hap_lists]))

# miness_lst = [[] for _ in range(max(amat.shape[1] for amat in amats))]

failed = []
for ds_idx in range(1, n_datasets+1):
    try:
        i = ds_idx - 1
        fn_prefix = f'../../data/encode/psize{pool_size}_m{n_markers}_id{ds_idx}'

        H = amats[i].shape[1]        
        
        ns, ys = parse_sim_data(fn_prefix+'.data')             

        t = time()
        lm_list = []
        for n, y in zip(ns, ys):            
            lm = LatentMult(amats[i], y, n, '../../4ti2-files/encode-mn',
                            solver, prefix_4ti2)
            lm_list.append(lm)
            # if lm.basis is not None:
            #     print(lm.basis.shape, end=' ')
        pre_time = time() - t
    
        print(f'MCMC for set {ds_idx}: {H} haplotypes')

        t = time()
        idata = latent_mult_mcmc(lm_list, H, n_sample, n_burnin, ['mn_approx']*n_pools,
                                 logprior_func=lambda p: (0.1-1)*jnp.sum(jnp.log(p)),
                                 jaxify=True, chains=chains, random_seed=ds_idx)
        mcmc_time = time() - t

        idata.attrs['preprocess_time'] = pre_time
        idata.attrs['mcmc_walltime'] = mcmc_time

        idata.to_netcdf(fn_prefix+'_mn_approx.netcdf')

        # miness_lst[H-1].append(az.ess(idata, var_names=['p'])['p'].values.min())
        # print(miness_lst[H-1][-1])

    except Exception as e:
        print(traceback.format_exc())
        failed.append(ds_idx)

# for h, m in enumerate(miness_lst):
#     if m:
#         print(h+1, np.min(m))
#     else:
#         print(h+1)

print('Datasets that failed:', failed)






