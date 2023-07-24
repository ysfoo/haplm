import numpy as np
import pandas as pd
import pulp
import arviz as az

from time import time
import os
import pickle as pkl
import traceback

from haplm.lm_dist import LatentMult, find_4ti2_prefix
from haplm.lm_inference import latent_mult_mcmc
from haplm.hap_util import mat_by_marker, str_to_num, num_to_str
from sim_data import parse_sim_data

# init for other libraries
solver = pulp.apis.SCIP_CMD(msg=False)
prefix_4ti2 = find_4ti2_prefix()

# parameters for data simulation
n_datasets = 100
n_pools = 20
n_markers = 8
pool_size = 50

# MCMC parameters
cores = 5
chains = 5
n_burnin = 500
n_sample = 500

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
for ds_idx in range(9, 10):
    try:
        i = ds_idx - 1
        fn_prefix = f'../../data/encode/psize{pool_size}_m{n_markers}_id{ds_idx}'
        ns, ys = parse_sim_data(fn_prefix+'.data')  
        H = amats[i].shape[1]

        t = time()
        lm_list = []
        for n, y in zip(ns, ys):            
            lm = LatentMult(amats[i], y, n, '../../4ti2-files/encode-markov',
                            solver, prefix_4ti2, enum_sols=True)
            lm_list.append(lm)
        pre_time = time() - t

        print(f'MCMC for set {ds_idx}')
        t = time()
        idata = latent_mult_mcmc(lm_list, H, n_sample, n_burnin, ['exact']*n_pools,
                                 chains=chains, cores=cores, seeds=list(range(chains)))
        mcmc_time = time() - t

        idata.sample_stats.attrs['preprocess_time'] = pre_time
        idata.sample_stats.attrs['mcmc_walltime'] = mcmc_time

        idata.to_netcdf(fn_prefix+'_exact.netcdf')

        ess = az.ess(idata, var_names=['p'])['p'].values
        miness_lst[H-1].append(ess.min())
        print(ess.min())

    except Exception as e:
        print(traceback.format_exc())
        failed.append(ds_idx)

for h, m in enumerate(miness_lst):
    if m:
        print(h+1, np.median(m))
    else:
        print(h+1)

print('Datasets that failed:', failed)

