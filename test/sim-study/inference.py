import os

from gen_sim_data import gen_sim_data


n_datasets = 20
n_pools = 30
n_markers = 4
pool_sizes = [10]
alphas = np.ones(2**n_markers)*0.5

for pool_size in pool_sizes:
	for ds_idx in range(n_datasets):
		fn_prefix = f'../../data/sim-study/psize{pool_size}_id{ds_idx}'

		# if data has not been generated
		if not (os.path.isfile(fn_prefix+'.data') and os.path.isfile(fn_prefix+'.prob')):
			gen_data(n_pools, n_markers, pool_size, alphas, ds_idx, fn_prefix)
