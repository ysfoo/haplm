'''
Runs MCMC inference for latent multinomial model applied to WWARN molecular marker data under a 
cross validation framework. Run the script with two command line arguments - the total number of 
folds (F), and the number of the specific fold (between 1 and F) to perform inference for.

The script `inference.py` must be run beforehand for the processed input file `processed.pkl`.
'''

import numpy as np
import scipy
import collections
import pickle as pkl
import sys
from time import time

import pymc as pm
import pymc.sampling_jax
import jax.numpy as jnp
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify
from haplm.gp_util import GP, SphereGneiting
import numpyro

# from haplm.numpyro_util import sample_numpyro_nuts

cv_rng = np.random.default_rng(7337) # seed for random subsetting for cross validation

draws = 500
tune = 500
chains = 5
numpyro.set_host_device_count(chains)

cv_tot, cv_fold = map(int, sys.argv[1:])
print(f'Performing inference for CV fold {cv_fold} out of {cv_tot}')

with open('processed.pkl', 'rb') as fp:
    data = pkl.load(fp)
    ys = data['ys']
    ns = data['ns']
    amats = data['amats']
    codes = data['codes']
    infos = data['infos']
    X = data['X']
    lm_list = data['lm_list']

# handle CV split
N_tot = len(ys)
test_start = int(N_tot * ((cv_fold - 1) / cv_tot))
test_end = int(N_tot * (cv_fold / cv_tot))
rand_idx = np.arange(N_tot)
cv_rng.shuffle(rand_idx)
train_idx = rand_idx[list(range(test_start)) + list(range(test_end, N_tot))]
test_idx = rand_idx[list(range(test_start, test_end))]

# save data for comparison later
N_test = len(test_idx)
ys_test = [ys[i] for i in test_idx]
ns_test = [ns[i] for i in test_idx]
amats_test = [amats[i] for i in test_idx]
codes_test = [codes[i] for i in test_idx]
X_test = X[test_idx]

# overwrite full lists to just the training set
N = len(train_idx)
ys = [ys[i] for i in train_idx]
ns = [ns[i] for i in train_idx]
amats = [amats[i] for i in train_idx]
infos = [infos[i] for i in train_idx]
X = X[train_idx]
lm_list = [lm_list[i] for i in train_idx]

print(f'Using {N} data points for inference '
      f'while withholding {N_test} data points for validation')


# dict that maps boolean tuple of whether pool has info about each marker
# to list of pool indices
# if pool has info on all markers, the index is separately stored in idx_all_haps
idx_by_info = collections.defaultdict(list)
idx_all_haps = []
for idx, info in enumerate(infos):
    if info == (True, True, True):
        idx_all_haps.append(idx)
    else:
        idx_by_info[info].append(idx)
N = len(ys)
H = 8
G = 3



# return list of list of full haplotypes for each collapsed haplotype,
# given boolean tuple of whether there is info about each marker
# collapsed haplotypes omit markers for which there is no info
def cols_given_info(has_info):
    G = len(has_info)
    H = 1 << G
    G_sub = sum(has_info)
    H_sub = 1 << G_sub
    cols = [[] for _ in range(H_sub)]
    
    # marker indices that have info
    test_gs = [g for g in range(G) if has_info[g]]
    # for each full haplotype, find the corresponding collapsed haplotype
    for h in range(H):
        cols[(sum(((h >> g) & 1) << g_sub for g_sub, g in enumerate(test_gs)))].append(h)
    return cols

idx_allhaps = [i for i, info in enumerate(infos) if info == (True, True, True)]
infos_excl_allhaps, idx_excl_allhaps = zip(*[(info, i) for i, info in enumerate(infos) if info != (True, True, True)])

pmat_dict = {}
for info in set(infos):
    cols_list = cols_given_info(info)
    pmat = np.zeros((len(cols_list), H), int)
    for i, cols in enumerate(cols_list):
        pmat[i, cols] = 1
    pmat_dict[info] = jnp.array(pmat)

logfacts = scipy.special.gammaln(np.arange(1, max(lm.n_var for lm in lm_list)+2))


# same approach as haplm.lm_inference.hier_latent_mult_mcmc with jaxify=True and methods=['exact']*N
def loglike_fn(p):    
    return (
            # deal with pools where there is info about all markers
            sum(lm_list[i].loglike_exact_jax(p[i], logfacts) for i in idx_allhaps) +
            # deal with other pools, as multinomial probabilities should be summed for each collapsed haplotype
            sum(lm_list[i].loglike_exact_jax(jnp.dot(pmat_dict[info], p[i]), logfacts)
                for i, info in zip(idx_excl_allhaps, infos_excl_allhaps)))

class LoglikeOp(Op):
    default_output = 0

    def make_node(self, p,):
        inputs = [p]
        outputs = [pt.dscalar()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        outputs[0][0] = np.asarray(loglike_fn(inputs[0]), dtype=node.outputs[0].dtype)

loglike_op = LoglikeOp()

@jax_funcify.register(LoglikeOp)
def loglike_dispatch(op, **kwargs):
    return loglike_fn


with pm.Model() as model:
    sigma = pm.InverseGamma('sigma', alpha=3, beta=1)
    alpha = pm.InverseGamma('alpha', alpha=3, beta=3, shape=H)
    ls_t = pm.InverseGamma('ls_t', alpha=3, beta=5, shape=H)
    ls_s = pm.InverseGamma('ls_s', alpha=3, beta=7, shape=H)    

    mean = pm.ZeroSumNormal('mu', sigma=2, shape=H)
    beta = pm.ZeroSumNormal('beta', sigma=2, shape=H)
    gps = [GP(cov_func=SphereGneiting(ls_s[h], ls_t[h]))
           for h in range(H)]
    ps = pm.Deterministic('p', pm.math.softmax(pt.stack([gp.prior(f'f{h}', X=X[:,:3])
                                                         for h, gp in enumerate(gps)], axis=1)
                                               *alpha + mean + pt.outer(pt.as_tensor_variable(X[:,3]), beta)
                                               + sigma*pm.Normal('noise', shape=(N,H)),
                                               axis=-1))
    loglike = pm.Potential('loglike', loglike_op(ps))

# run MCMC
t = time()
with model:
    idata = pm.sampling_jax.sample_numpyro_nuts(draws=draws, tune=tune, chains=chains, target_accept=0.9,
                                                random_seed=2023, postprocessing_chunks=25)
mcmc_time = time() - t

idata.sample_stats.attrs['mcmc_walltime'] = mcmc_time

# save trace
idata.posterior = idata.posterior.drop_vars('noise')
idata.to_netcdf(f'../../data/dhps/exact_cv{cv_fold}.netcdf')


# sample posterior predictive
with model:
    fs_pred = [gps[h].marg_cond(f'f_pred{h}', Xnew=X_test[:,:3]) for h in range(H)]
    ps_pred = pm.Deterministic(f'p_pred', 
                               pm.math.softmax(pt.stack(fs_pred, axis=1)
                                               *alpha + mean + pt.outer(pt.as_tensor_variable(X_test[:,3]), beta)
                                               + sigma*pm.Normal('noise_pred', shape=(N_test,H)),
                                               axis=-1))
    pred_idata = pm.sample_posterior_predictive(idata, var_names=['p_pred'])

pred_samples = np.vstack(pred_idata.posterior_predictive.p_pred)
np.save(f'../../data/dhps/exact_pred_samples_cv{cv_fold}.npy', pred_samples)