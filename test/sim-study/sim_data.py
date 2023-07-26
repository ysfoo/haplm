"""
Simulates synthetic latent multinomial data with a Dirichlet prior.
"""

import numpy as np
from haplm.hap_util import mat_by_marker


def gen_sim_data(n_pools, n_markers, pool_size, alphas, seed, fname):
	"""
	Generates latent multinomial data for the simulation study. Each dataset consists of `n_pools` 
	pools of `n_markers` allele counts. For each dataset, multinomial probabilities are sampled for 
	`2^n_markers` haplotypes from a Dirichlet distribution.

    The pool size and allele counts are written into a space separated file `<fname>.data`. Each 
    line begins with the pool size (all equal in this case) followed by the allele counts of each 
    marker. The multinomial probabilities are written into a space separated file `<fname>.prob`.

    Parameters
    ----------
    n_pools : int > 0
    	Nnumber of pools to simulate.
    n_markers : int > 0
    	Number of markers.
    pool_size : int > 0
    	Number of samples in each pool.
    alphas : list[float > 0], optional
    	Concentration parameters of the Dirichlet distribution.
    seed : int
    	Seed for reproducibility.
    fname : string
		Prefix for data filenames.

	Returns
	-------
	None
	"""
	seed = seed ^ n_markers
	np.random.seed(seed)

	H = 2**n_markers
	ptrue = np.random.dirichlet(alphas)

	with open(f'{fname}.prob', 'w') as fp:
		fp.write('\n'.join([str(p) for p in ptrue]))

	amat = mat_by_marker(n_markers)
	#present = np.zeros(H)
	with open(f'{fname}.data', 'w') as fp:
		for _ in range(n_pools):
			zvec = np.random.multinomial(pool_size, ptrue)
			#present[zvec>0] = 1
			yvec = np.dot(amat, zvec).astype(int)

			fp.write(' '.join([str(pool_size)] + [str(y) for y in yvec]))
			fp.write('\n')
	#print(present.sum())


def parse_sim_data(fname):
	"""
	Parse data produced by `gen_sim_data`.

	Parameters
    ----------
    fname : string
		Filename containing pool size and observed counts.

	Returns
    -------
    tuple (list[int], list[list[int]])
        2-tuple consisting of (i) a list of pool sizes, and (ii) a list of observed allele counts.
	"""
	ns = []
	ys = []

	with open(fname) as fp:
		for line in fp:
			tokens = [int(x) for x in line.split()]
			ns.append(tokens[0])
			ys.append(tokens[1:])

	return np.array(ns), np.array(ys)
