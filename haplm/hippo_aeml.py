import numpy as np
from scipy.stats import entropy

from time import time
import os
import pickle as pkl
import subprocess as sp
import arviz as az
import xarray

def run_AEML(ns, ys, aeml_dir, trials=10,
	         print_trial=True, seed=0, n_iterations=10**4, stab=None,
	         use_ent=False, hap_fn=None):
	n_pools = len(ys)
	n_markers = len(ys[0])

	write_data(ns, ys)

	maxval = -np.inf
	pest = None
	convg = False
	if print_trial:
		print('Trial', end='')

	t0 = time()
	found_convg = False
	for trial in range(trials):
		print(f' {trial+1}', end='', flush=True)
		trial += 1
		with open('AEML_seed', 'w') as fp:
			fp.write(str(seed ^ trial + 1))

		par_fn = 'aeml.par'
		with open(par_fn, 'w') as fp:
			fp.write(f'data_file hippo_aeml.data\n')
			fp.write(f'n_loci {n_markers}\n')
			fp.write(f'n_pools {n_pools}\n')            
			fp.write(f'random_init {min(1, trial)}\n') # first attempt is from all equal init
			fp.write(f'n_iterations {n_iterations}\n')
			if stab is not None:
				fp.write(f'stab {stab}\n')
			if hap_fn is not None:
				fp.write(f'hap_file {hap_fn}\n')

		if hap_fn is not None:
			with open(hap_fn) as fp:
				H = int(next(fp))
				hap_dict = {''.join(line.split()): i for i, line in enumerate(fp)}
		else:
			H = 2**n_markers

		with open('AEML.log', 'w') as fp:
			sp.run([f'{aeml_dir}AEML', par_fn], stdout=fp, stderr=fp)

		with open('AEML.log') as fp:
			for line in fp:
				pass
			if 'Exits' not in line:
				assert "Covariance matrix is zero matrix!" in line
				continue # did not terminate properly	

		with open('AEML_monitor.out') as fp:
			lines = 0
			for line in fp:
				lines += 1
			loglike = float(line.strip())
		convg = lines < n_iterations

		pest = np.zeros(H)
		with open('AEML.out') as fp:
			for line in fp:
				hstr, pstr = line.split()
				if hap_fn is None:
					h = sum(1 << i for i, b in enumerate(hstr) if b=='1')
				else:
					h = hap_dict[hstr]
				pest[h] = float(pstr)

		if use_ent:
			currval = entropy(pest)
		else:
			currval = loglike

		if currval <= maxval:
			continue

		if convg:
			found_convg = True
		maxval = currval
		selected = pest.copy()

	if print_trial:
		print()

	if not maxval > -np.inf:
		if stab is None:
			raise RuntimeError("Singular covariance, need to include stabilising constant")

		if print_trial:
			print('Increase stabilising constant')
		stab *= 10

		return run_AEML(ns, ys, aeml_dir, trials,
		         		print_trial, seed, n_iterations, stab,
		         		use_ent, hap_fn)

	return {'pest': selected, 'time': time() - t0, 'convg': found_convg}


def run_hippo(ns, ys, n_sample, n_burnin, hippo_dir, 
			  thin=1, chains=5, alpha=None, gamma=None, stab=None,
			  hap_fn=None, print_chain=True, seed=0):
	n_pools = len(ys)
	n_markers = len(ys[0])
	H = 2**n_markers

	write_data(ns, ys)
	par_fn = 'hippo.par'
	with open(par_fn, 'w') as fp:
		fp.write(f'data_file hippo_aeml.data\n')
		fp.write(f'n_loci {n_markers}\n')
		fp.write(f'n_pools {n_pools}\n')            
		fp.write(f'n_iterations {n_sample+n_burnin}\n')
		fp.write(f'n_burnin {n_burnin}\n')
		fp.write('tol 0\n')
		fp.write('variable_list 2\n')
		fp.write('write_trace 1\n')
		fp.write(f'thin {thin}\n')
		if alpha is not None:
			fp.write(f'alpha {alpha}\n')
		if gamma is not None:
			fp.write(f'gamma {gamma}\n')
		if stab is not None:
			fp.write(f'stab {stab}\n')
		if hap_fn is not None:
			fp.write(f'hap_file {hap_fn}\n')
	
	if print_chain:
		print('Chain', end='')

	trace = []
	times_excl_tune = []
	times_incl_tune = []
	modes = []
	avg_logposts = []
	max_loglike = -np.inf
	for chain in range(chains):
		print(f' {chain+1}', end='', flush=True)
		t = time()
		with open('seed', 'w') as fp:
			fp.write(str(seed^chain+1))
		with open('hippo.log', 'w') as fp:
			sp.run([f'{hippo_dir}hippo', par_fn], stdout=fp, stderr=fp)
		times_incl_tune.append(time() - t)

		with open('trace.out') as fp:
			trace.append([parse_trace_line(line, H) for line in fp])

		logpost = []
		with open('monitor.out') as fp:
			for i, line in enumerate(fp):
				if i*100 < n_burnin:
					continue
				logpost.append(float(line.split()[0]))
		avg_logposts.append(np.mean(logpost))

		with open('hippo.log') as fp:
			for line in fp:
				pass
			assert 'Time after burn-in is' in line
			times_excl_tune.append(float(line.split()[-1]))	

		with open('MAP.out') as fp:
			loglike = float(next(fp).strip())
			if loglike > max_loglike:
				pmode = [0]*H
				for line in fp:
					hstr, pstr = line.split()
					h = sum(1 << i for i, hchar in enumerate(hstr) if hchar == '1')
					pmode[h] = float(pstr)

	if print_chain:
		print()

	posterior = {'p': (['chain', 'draw', 'p_dim'], np.array(trace))}
	sample_stats = {'time_incl_tune': (['chain'], np.array(times_incl_tune)),
	                'time_excl_tune': (['chain'], np.array(times_excl_tune)),
	                'pmode': (['p_dim'], np.array(pmode)),
	                'avg_logpost': np.array(avg_logposts),
	                }

	return az.InferenceData(posterior=xarray.Dataset(posterior),
							sample_stats=xarray.Dataset(sample_stats))

def write_data(ns, ys):
	# write data for AEML
	with open('hippo_aeml.data', 'w') as fp_out:
		for n, y in zip(ns, ys):
			tokens = [str(n)] + [str(yval) for yval in y]
			fp_out.write(' '.join(tokens))
			fp_out.write('\n')

def parse_trace_line(line, H):
	p = [0]*H
	tokens = line.split()
	for i in range(0, len(tokens), 2):
		p[int(tokens[i])] = float(tokens[i+1])
	return p
