"""
Perform MCMC-Approx on time series example.

Input files required:
- Observation time, pool size, and observed allele counts for each pool
  @ '../../data/time-series/time_series.data'

Output files produced:
- InferenceData output of MCMC-Approx
  @ '../../data/time-series/psize{pool_size}_m{n_markers}_id{ds_idx}_mn_approx.netcdf'
- Posterior predictive samples as a numpy.darray of dimensions (chains*N) x H
  @ '../../data/time-series/psize{pool_size}_m{n_markers}_id{ds_idx}_mn_approx_pred_samples.npy'
- Summary statistics of posterior predictive distribution as a dictionary
  @ '../../data/time-series/psize{pool_size}_m{n_markers}_id{ds_idx}_mn_approx_sumstats.pkl'
"""

import numpy as np
import pickle as pkl
import pulp
from time import time

import pymc as pm
# import arviz as az
import pytensor.tensor as pt

from haplm.lm_dist import LatentMult, find_4ti2_prefix
from haplm.hap_util import mat_by_marker
from haplm.lm_inference import hier_latent_mult_mcmc
from haplm.gp_util import GP

import numpyro

# init for other libraries
solver = pulp.apis.SCIP_CMD(msg=False)
prefix_4ti2 = find_4ti2_prefix()

# data parameters
H = 8
N = 30 # number of data points
pool_size = 50
n_markers = 3

# MCMC parameters
chains = 5
cores = 5
n_burnin = 1000
n_sample = 1000
numpyro.set_host_device_count(cores)

amat = mat_by_marker(n_markers)

# each row of input file consists of observation time, pool size, allele counts of each marker
t_obs = []
ns = []
ys = []
with open('../../data/time-series/time_series.data') as fp:
    for line in fp:
        tokens = line.split()
        t_obs.append(float(tokens[0]))
        ns.append(int(tokens[1]))
        ys.append(np.array([int(x) for x in tokens[2:]]))
t_obs = np.array(t_obs)

# preprocessing
t = time()
lm_list = []
for i, (n, y) in enumerate(zip(ns, ys)):    
    lm = LatentMult(amat, y, n, '../../4ti2-files/tseries-mn_approx',
                    solver, prefix_4ti2, enum_sols=True)
    lm_list.append(lm)
pre_time = time() - t
# print(pre_time, flush=True)

# define GP prior
with pm.Model() as model:
    sigma = pm.InverseGamma('sigma', alpha=3, beta=1)
    alpha = pm.InverseGamma('alpha', alpha=3, beta=3, shape=H)
    ls_t = pm.InverseGamma('ls_t', alpha=3, beta=5, shape=H)

    mean = pm.ZeroSumNormal('mu', sigma=2, shape=H)
    gps = [GP(cov_func=pm.gp.cov.RatQuad(1, 1, ls_t[h]))
           for h in range(H)]
    ps = pm.Deterministic('p', pm.math.softmax(pt.stack([gp.prior(f'f{h}', X=t_obs[:,None])
                                                         for h, gp in enumerate(gps)], axis=1)
                                               *alpha + mean
                                               + sigma*pm.Normal('noise', shape=(N,H)),
                                               axis=-1))

t = time()
with model:
    idata = hier_latent_mult_mcmc(ps, lm_list, H, n_sample, n_burnin, ['mn_approx']*N, jaxify=True, 
                                  chains=chains, random_seed=2023, postprocessing_chunks=25)
mcmc_time = time() - t

idata.sample_stats.attrs['preprocess_time'] = pre_time
idata.sample_stats.attrs['mcmc_walltime'] = mcmc_time

# ess = az.ess(idata, var_names=['p'])['p'].values
# print(ess.min())

idata.posterior = idata.posterior.drop_vars('noise')
idata.to_netcdf(f'../../data/time-series/psize50_m3_mn_approx.netcdf')

# simulate predictive distribution
t_pred = np.arange(0, 20.001, 0.01)
N_pred = len(t_pred)
with model:
    fs_pred = [gps[h].marg_cond(f'f_pred{h}', Xnew=t_pred[:,None]) for h in range(H)]
    ps_pred = pm.Deterministic(f'p_pred', pm.math.softmax(pt.stack(fs_pred, axis=1)*alpha + mean
                                                          + sigma*pm.Normal('noise_pred', shape=(N_pred,H)),
                                                          axis=-1))
    pred_idata = pm.sample_posterior_predictive(idata, var_names=['p_pred'])

pred_samples = np.vstack(pred_idata.posterior_predictive.p_pred)
np.save(f'../../data/time-series/psize50_m3_mn_approx_pred_samples.npy', pred_samples)
# np.save('../../data/time-series/psize50_m3_mn_approx_pred_samples_tint.npy', pred_samples[:,::100])

# pred_samples = np.load(f'../../data/time-series/psize50_m3_mn_approx_pred_samples.npy')

# summary statistics of predictive distribution
sumstats = {}
sumstats['mean'] = pred_samples.mean(axis=0)
sumstats['sd'] = pred_samples.std(axis=0)
sumstats['median'] = np.median(pred_samples, axis=0)
sumstats['mad'] = np.median(np.abs(pred_samples - sumstats['median'][None,:,:]), axis=0)
sumstats['quantiles'] = np.quantile(pred_samples, np.arange(0, 1.01, 0.05), axis=0)

with open(f'../../data/time-series/psize50_m3_mn_approx_sumstats.pkl', 'wb') as fp:
    pkl.dump(sumstats, fp)