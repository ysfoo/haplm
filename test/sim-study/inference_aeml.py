import numpy as np

import os
import pickle as pkl

from haplm.hap_utils import mat_by_marker
from haplm.hippo_aeml import run_AEML
from sim_data import gen_sim_data, parse_sim_data

# location of hippo_aeml
aeml_dir = '../../hippo_aeml/'

# parameters for data simulation
n_datasets = 5
n_pools = 20
n_markers = 3
H = 2**n_markers # number of haplotypes
pool_sizes = np.arange(20, 101, 20)
alphas = np.ones(H)*0.4

amat = mat_by_marker(n_markers)

trials = 10 # number of random initialisations

for pool_size in pool_sizes:
	print(f'Pool size = {pool_size}')
	print('-'*15)
	for ds_idx in range(1, n_datasets+1):
		fn_prefix = f'../../data/sim-study/psize{pool_size}_m{n_markers}_id{ds_idx}'

		# if data has not been generated
		if not os.path.isfile(fn_prefix+'.data'):
			gen_sim_data(n_pools, n_markers, pool_size, alphas, ds_idx, fn_prefix)

		ns, ys = parse_sim_data(fn_prefix+'.data')

		print(f'AEML for set {ds_idx}: ', end='')
		aeml_out = run_AEML(ns, [y[:4] for y in ys], aeml_dir)
		assert aeml_out is not None

		with open(f'{fn_prefix}_aeml.pkl', 'wb') as fp:
			pkl.dump(aeml_out, fp)
	print()





