#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import pickle as pkl
import pulp

import pymc as pm
import arviz as az
import aesara
import aesara.tensor as at


# In[2]:


from haplm.lm_dist import LatentMult, find_4ti2_prefix
from haplm.hap_utils import num_to_str, str_to_num, mat_by_marker
from haplm.gp_tools import GP


# In[3]:


# init for other libraries
solver = pulp.apis.SCIP_CMD(msg=False)
prefix_4ti2 = find_4ti2_prefix()


# In[4]:


H = 8
N = 30 # number of data points
pool_size = 50
n_markers = 3
amat = mat_by_marker(n_markers)


# In[5]:


with open('../../data/time-series/sim_data.pkl', 'rb') as fp:
    tmp = pkl.load(fp)
    p_true = tmp['p'] # H x T
    zs_true = tmp['z'] # N x H


# In[6]:


t_obs = []
ns = []
ys = []
with open('../../data/time-series/ts.data') as fp:
    for line in fp:
        tokens = line.split()
        t_obs.append(float(tokens[0]))
        ns.append(int(tokens[1]))
        ys.append(np.array([int(x) for x in tokens[2:]]))
t_obs = np.array(t_obs)


# In[7]:


lm_list = []
for n, y in zip(ns, ys):	
    # t = time()
    lm = LatentMult(amat, y, n, '../../4ti2-files/tseries-exact',
                    solver, prefix_4ti2, enum_sols=True)
    lm_list.append(lm)
    # pre_time = time() - t
    # print(pre_time, flush=True)


# In[10]:


def latent_mult_pymc(lm_list, ps, H, n_sample, n_burnin, methods,
                     target_accept=0.9,
                     chains=5, cores=1, seeds=None):   
    if seeds is None:
        seeds = [None]*chains
    else:
        seeds = list(seeds)
    assert len(seeds) == chains
    
    # check if no need for approximation nor discrete sampling
    for i, lm in enumerate(lm_list):
        if not lm.idx_var or not lm.amat_var.size:
            methods[i] = 'exact'
    
    exact_is = [i for i, method in enumerate(methods) if method=='exact']
    basis_is = [i for i, method in enumerate(methods) if method=='basis']
    mn_is = [i for i, method in enumerate(methods) if method=='mn_approx']
    assert len(exact_is)+len(basis_is)+len(mn_is) == len(lm_list)

    def make_logp(lm, method):
        if method == 'exact':
            return lambda z, p: lm.loglike_exact(p)
        elif method == 'mn_approx':
            return lambda z, p: lm.loglike_mn(p)

    with model:        
        ys_mn = [pm.DensityDist('y'+str(i), ps[i],
                                logp=make_logp(lm_list[i], 'mn_approx'),
                                observed=np.zeros(1), dtype='int64')
                 for i in mn_is]
        ys_exact = [pm.DensityDist('y'+str(i), ps[i],
                                logp=make_logp(lm_list[i], 'exact'),
                                observed=np.zeros(1), dtype='int64')
                 for i in exact_is]
               
    if basis_is:
        raise NotImplementedError
    
    print('call pm.sample')
    with model:
        idata = pm.sample(draws=n_sample, tune=n_burnin, target_accept=target_accept,
                          chains=chains, cores=cores, random_seed=seeds,
                          #nuts_sampler='numpyro',
                          compute_convergence_checks=False)

    return idata


# In[9]:


idx = 0
# eps = 1e-8/(1+H*1e-8)
bmat = np.array([[1]*i+[-i]+[0]*(H-i-1) for i in range(1, H)])
bmat = bmat / np.linalg.norm(bmat, axis=1)[:,None]

with pm.Model() as model:
    alpha = pm.HalfNormal('alpha', sigma=2, shape=H)
    sigma = pm.HalfNormal('sigma')
    mrot = pm.Normal('mu_rotated', sigma=2, shape=H-1)
    mean = pm.Deterministic('mu', at.dot(mrot, bmat))
    ls_t = pm.Lognormal('ls_t', mu=1, sigma=0.5, shape=H)
    gps = [GP(cov_func=pm.gp.cov.RatQuad(1, 1, ls_t[i]))
           for i in range(H)]
    ps = pm.math.softmax(at.stack([gp.prior("f"+str(i), X=t_obs[:,None])
                                  for i, gp in enumerate(gps)], axis=1)
                         *alpha + mean + sigma*pm.Normal('noise', shape=(N,H)),
                         axis=-1)


# In[11]:

from time import time
t = time()

with model:
    y_rv = pm.Binomial('y', p=at.dot(ps, amat.T), n=pool_size, observed=np.array(ys, int))
    print('call pm.sample')
    idata = pm.sample()

# idata = latent_mult_pymc(lm_list, ps, H, 500, 500, ['exact']*N,
#                          target_accept=0.9, chains=4, cores=4, seeds=np.arange(4))
print(time() - t)

idata.to_netcdf('playground.netcdf') 