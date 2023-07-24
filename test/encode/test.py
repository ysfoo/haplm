import numpy as np
import pandas as pd
import pulp
import arviz as az

from time import time
import os
import pickle as pkl
import traceback

from haplm.lm_dist import LatentMult, find_4ti2_prefix
from haplm.hap_util import mat_by_marker, str_to_num, num_to_str
from haplm.lm_inference import latent_mult_mcmc, latent_mult_numpyro
from sim_data import gen_sim_data, parse_sim_data

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

# init for other libraries
solver = pulp.apis.SCIP_CMD(msg=False)
prefix_4ti2 = find_4ti2_prefix()

# parameters for data simulation
n_datasets = 100
n_pools = 20
n_markers = 8
pool_size = 50

# MCMC parameters
cores = 2
chains = 2
n_burnin = 5
n_sample = 5
numpyro.set_host_device_count(cores)
numpyro.enable_x64()

hap_lists = []
amats = []
with open('../../data/encode/PL.txt') as fp:
    for line in fp:
        hap_list = [int(x) for x in line.split()]
        hap_lists.append(hap_list)
        amats.append(np.array([[(hnum >> m) & 1 for hnum in hap_list] for m in range(n_markers)]))

print(np.mean([len(l) for l in hap_lists]))

miness_lst = [[] for _ in range(max(amat.shape[1] for amat in amats))]

failed = []
# for ds_idx in [12, 43, 48, 54, 62, 74, 77, 81, 82, 84, 85, 90, 100]:
for ds_idx in [100]:
    try:
        i = ds_idx - 1
        fn_prefix = f'../../data/encode/psize{pool_size}_m{n_markers}_id{ds_idx}'

        H = amats[i].shape[1]
        
        
        ns, ys = parse_sim_data(fn_prefix+'.data')             

        t = time()
        lm_list = []
        for n, y in zip(ns, ys):            
            lm = LatentMult(amats[i], y, n, '../../4ti2-files/encode-mn-markov',
                            solver, prefix_4ti2)
            lm_list.append(lm)
        pre_time = time() - t
    
        print(f'MCMC for set {ds_idx}')

        t = time()
        # mcmc, idata = latent_mult_numpyro(lm_list, H, n_sample, n_burnin, ['mn_approx']*n_pools,
        #                             UniformDirichlet(H),
        #                             chains=chains, cores=cores,
        #                             seed=ds_idx^pool_size)
        idata = latent_mult_mcmc(lm_list, H, n_sample, n_burnin, ['mn_approx']*n_pools,
                                    chains=chains, cores=cores,
                                    seed=ds_idx^pool_size)
        mcmc_time = time() - t

        idata.sample_stats.attrs['preprocess_time'] = pre_time
        idata.sample_stats.attrs['mcmc_walltime'] = mcmc_time

        #idata.to_netcdf(fn_prefix+'_mn_approx.netcdf')
        idata.to_netcdf('tmp.netcdf')

        miness_lst[H-1].append(az.ess(idata, var_names=['p'])['p'].values.min())
        print(miness_lst[H-1][-1])

    except Exception as e:
        print(traceback.format_exc())
        failed.append(ds_idx)

# for h, m in enumerate(miness_lst):
#     if m:
#         print(h+1, np.median(m))
#     else:
#         print(h+1)

# print('Datasets that failed:', failed)






