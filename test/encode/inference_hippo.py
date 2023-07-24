import numpy as np
import arviz as az
import xarray

from time import time
import os
import subprocess as sp
import traceback

from haplm.hap_util import mat_by_marker, str_to_num, num_to_str
from haplm.hippo_aeml import run_hippo
from sim_data import gen_sim_data, parse_sim_data

# location of hippo_aeml
hippo_dir = '../../hippo_aeml/'

# parameters for data simulation
n_datasets = 100
n_pools = 20
n_markers = 8
pool_size = 50

# MCMC parameters
chains = 5
n_burnin = 50000*5
n_sample = 450000*5
thin = 900*5

hap_lists = []
with open('../../data/encode/PL.txt') as fp:
    for line in fp:
        hap_lists.append([int(x) for x in line.split()])

miness_lst = [[] for _ in range(max(len(hap_list) for hap_list in hap_lists))]

failed = []
for ds_idx in range(1, n_datasets+1):
	try:
		hap_fn = 'encode_haplist.txt'
		with open(hap_fn, 'w') as fp:
			hap_list = [num_to_str(hnum, n_markers) for hnum in hap_lists[ds_idx-1]]
			fp.write(f'{len(hap_list)}\n')
			for hstr in hap_list:
				fp.write(' '.join(list(hstr)))
				fp.write('\n')

		fn_prefix = f'../../data/encode/psize{pool_size}_m{n_markers}_id{ds_idx}'
		ns, ys = parse_sim_data(fn_prefix+'.data')
		H = len(hap_list)

		print(f'MCMC for set {ds_idx}: ', end='')
		idata = run_hippo(ns, ys, n_sample, n_burnin, hippo_dir,
			              thin=thin, chains=chains, hap_fn=hap_fn)
		idata.to_netcdf(fn_prefix+'_hippo.netcdf')

		miness_lst[H-1].append(az.ess(idata, var_names=['p'])['p'].values.min())
		print(miness_lst[H-1][-1])
		
	except Exception as e:
		print(traceback.format_exc())
		failed.append(ds_idx)
            
for h, m in enumerate(miness_lst):
    if m:
        print(h+1, np.median(m))
    else:
        print(h+1)

print('Datasets that failed:', failed)
