import numpy as np
from haplm.hap_utils import mat_by_marker

def gen_sim_data(n_pools, n_markers, pool_size,
			     alphas, seed, fn):
	"""
	Generates latent multinomial data for the simulation study. Each 
    dataset consists of N pools of G allele counts. For each dataset, multinomial
    probabilities are sampled for 2^G haplotypes from a Dirichlet distribution.

    The pool size and allele counts are written into a space separated file
	`<fn>.data`. Each line begins with the pool size (all equal in this case) 
	followed by the allele counts of each marker.
	The multinomial probabilities are written into a space separated file
	`<fn>.prob`.

    Parameters
    ----------
    n_pools : int
    	Nnumber of datasets to generate.
    n_markers : int
    	Number of markers.
    pool_size : int
    	Number of samples in each pool.
    alphas : 1D-array
    	Concentration parameters of the Dirichlet distribution.
    seed : int
    	Seed for reproducibility.
    fn : string
		Prefix for data filenames.
	"""
	seed = seed ^ n_markers
	np.random.seed(seed)

	H = 2**n_markers
	ptrue = np.random.dirichlet(alphas)

	with open(f'{fn}.prob', 'w') as fp:
		fp.write('\n'.join([str(p) for p in ptrue]))

	amat = mat_by_marker(n_markers)
	#present = np.zeros(H)
	with open(f'{fn}.data', 'w') as fp:
		for _ in range(n_pools):
			zvec = np.random.multinomial(pool_size, ptrue)
			#present[zvec>0] = 1
			yvec = np.dot(amat, zvec).astype(int)

			fp.write(' '.join([str(pool_size)] + [str(y) for y in yvec]))
			fp.write('\n')
	#print(present.sum())

def parse_sim_data(fn):
	"""
	Parse data produced by `gen_sim_data`.

	Parameters
    ----------
    fn : string
		Filename containing pool size and observed counts.

	Returns
	-------
	tuple
		1) Pool sizes.
		2) Observed allele counts.
	"""
	ns = []
	ys = []

	with open(fn) as fp:
		for line in fp:
			tokens = [int(x) for x in line.split()]
			ns.append(tokens[0])
			ys.append(tokens[1:])

	return np.array(ns), np.array(ys)
