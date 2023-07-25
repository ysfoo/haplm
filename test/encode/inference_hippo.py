"""
Perform HIPPO on datasets based on the 1000 Genomes Project.

Input files required:
- Output of partition ligation for initialisation 
  @ '../../data/encode/PL.txt'
- Pool size with observed allele counts for each pool
  @ '../../data/encode/psize{pool_size}_m{n_markers}_id{ds_idx}.data'

Output files produced:
- InferenceData output of HIPPO
  @ '../../data/encode/psize{pool_size}_m{n_markers}_id{ds_idx}_hippo.netcdf'
"""

import numpy as np
# import arviz as az

from time import time
import traceback

from haplm.hap_util import mat_by_marker, str_to_num, num_to_str
from haplm.hippo_aeml import run_hippo
from sim_data import parse_sim_data

# location of HIPPO program
hippo_dir = '../../hippo_aeml/'

# data parameters
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

# miness_lst = [[] for _ in range(max(len(hap_list) for hap_list in hap_lists))]

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
			                seed=pool_size^ds_idx, thin=thin, chains=chains, stab=1e-9, hap_fn=hap_fn)
		idata.to_netcdf(fn_prefix+'_hippo.netcdf')

		# miness_lst[H-1].append(az.ess(idata, var_names=['p'])['p'].values.min())
		# print(miness_lst[H-1][-1])
		
	except Exception as e:
		print(traceback.format_exc())
		failed.append(ds_idx)
            
# for h, m in enumerate(miness_lst):
#     if m:
#         print(h+1, np.median(m))
#     else:
#         print(h+1)

print('Datasets that failed:', failed)
