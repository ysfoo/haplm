import numpy as np
import scipy
import sympy
from sympy.polys.domains import ZZ
from sympy.polys.matrices import DM
import pickle as pkl
import pulp
from time import time

import pymc as pm
import arviz as az
import xarray

import jax
from jax import jit
import jax.numpy as jnp
import numpyro
import pytensor.tensor as pt

chains = 5
draws = 20000
tune = 2000
thinning = 20
numpyro.set_host_device_count(chains)

from haplm.lm_dist import LatentMult, find_4ti2_prefix
from haplm.hap_util import num_to_str, str_to_num, mat_by_marker
from haplm.lm_inference import latent_mult_mcmc
from haplm.gp_util import GP
from haplm.numpyro_util import sample_numpyro_nuts_gibbs

# init for other libraries
solver = pulp.apis.SCIP_CMD(msg=False)
prefix_4ti2 = find_4ti2_prefix()

H = 8
N = 30 # number of data points
pool_size = 50
n_markers = 3
amat = mat_by_marker(n_markers)

t_obs = []
ns = []
ys = []
with open('../../data/time-series/time-series.data') as fp:
    for line in fp:
        tokens = line.split()
        t_obs.append(float(tokens[0]))
        ns.append(int(tokens[1]))
        ys.append(np.array([int(x) for x in tokens[2:]]))
t_obs = np.array(t_obs)

t = time()
lm_list = []
# ys[5][2] = 0
for n, y in zip(ns, ys):    
    # t = time()
    lm = LatentMult(amat, y, n, '../../4ti2-files/tseries-mbasis',
                    solver, prefix_4ti2, walk_len=500, num_pts=chains)
    lm.compute_basis(markov=True)
    lm_list.append(lm)
pre_time = time() - t

jnp_logfacts = jnp.array(scipy.special.gammaln(np.arange(1, max(lm.n_var for lm in lm_list)+2)))
temp_init = 1000
temp_mult = 1000 / tune

def make_site_fn(lm_list, i):
    lm = lm_list[i]
    idx_var = jnp.array(lm.idx_var)    
    bvecs = jnp.array(np.vstack([lm.basis, -lm.basis]))
    
    @jit
    def site_fn(hmc_i, z_full, logp, key):
        logp_var = logp[idx_var]
        z0 = z_full[idx_var]
        temp = jnp.maximum(1, temp_init/(1+temp_mult*hmc_i))
        def fn_to_rep(i, state):
            z0, opts0, ws0, key = state
            opt_key, mh_key, next_key = jax.random.split(key, 3)
            z = jax.random.choice(opt_key, opts0, p=ws0)
            opts = z + bvecs
            ws = jnp.where((opts >= 0).all(axis=1), jnp.exp(jnp.dot(opts, logp_var) - jnp_logfacts[opts].sum(axis=1)), 0)
            return_new = jnp.log(jax.random.uniform(mh_key)) < jnp.log(ws0.sum()/ws.sum()) / temp
            #jax.debug.print('z {z}\nopts0 {opts0}\nws0 {ws0}', opts0=opts0, z=z, ws0=ws0)
            return jax.lax.cond(return_new,
                                lambda x: (z, opts, ws, next_key),
                                lambda x: (z0, opts0, ws0, next_key),
                                None)
        
        opts0 = z0 + bvecs
        ws0 = jnp.where((opts0 >= 0).all(axis=1), jnp.exp(jnp.dot(opts0, logp_var) - jnp_logfacts[opts0].sum(axis=1)), 0)
        return z_full.at[idx_var].set(jax.lax.fori_loop(0, 10*lm.n_var, fn_to_rep, (z0, opts0, ws0, key))[0])
        
    return site_fn

def make_gibbs_fn(lm_list):
    gibbs_fns = []
    for i, lm in enumerate(lm_list):
        if not lm.idx_var.size:
            gibbs_fns.append(lambda z, logp, key: z)
        else:
            gibbs_fns.append(make_site_fn(lm_list, i))
    
    def tmp_gibbs_fn(hmc_i, rng_key, gibbs_sites, hmc_sites):
        keys = jax.random.split(rng_key, N)
        log_ps = jnp.log(hmc_sites['p'])
        new_z = {'z': jnp.array([gibbs_fns[i](hmc_i, gibbs_sites['z'][i], log_ps[i], keys[i]).astype('int64') for i in range(N)])} 
        # jax.debug.print("{new_z}", new_z=new_z)
        return new_z
    
    return tmp_gibbs_fn

def update_gibbs_fn(samples, tune, gibbs_sites, gibbs_idxs):
    samples_idx = gibbs_idxs[gibbs_sites.index('z')]
    posterior = {'z': (('chain', 'draw', 'z_dim_0', 'z_dim_1'), samples[samples_idx])}
    ess = az.ess(xarray.Dataset(posterior).sel(draw=np.arange(tune // 2, tune)))['z'].values
    for i, lm in enumerate(lm_list):
        if not lm.idx_var.size:
            continue
        nz = len(lm.idx_var)
        zess = ess[i, lm.idx_var]
        order = list(np.argsort(zess))
        dist_best = 0.2*np.sum(np.abs(zess-np.median(zess)))
        lowess_list = order
        for div in range(1, len(order)):
            dist = sum(np.sum(np.abs(vals-np.median(vals)))
                       for vals in (zess[order[:div]], zess[order[div:]]))
            if dist < dist_best:
                dist_best = dist
                lowess_list = order[:div]

        amat = sympy.Matrix(np.vstack([np.ones(nz, int), lm.amat_var.astype(int)]))
        nspace = []
        for vec in amat[:,lowess_list].nullspace():
            denoms = [x.q for x in vec if type(x) == sympy.Rational]
            if len(denoms) == 0:
                arr = np.array(vec, int)
            elif len(denoms) == 1:
                arr = np.array(vec*denoms[0], int)
            else:
                arr = np.array(vec*sympy.ilcm(*denoms), int)
            nspace.append(arr.T[0])
        nspace = np.vstack(nspace)

        reduced = DM(nspace, ZZ).lll()
        extras = np.array(reduced.to_Matrix(), int)
        div = len(lowess_list)
        for tmp in extras:
            if np.linalg.norm(tmp) > 5:
                continue
            extra = np.zeros(nz, int)
            extra[lowess_list] = tmp
            assert extra.sum() == 0 and (lm.amat_var.dot(extra) == 0).all()
            if {tuple(row) for row in lm.basis}.isdisjoint({tuple(extra), tuple(-extra)}):
                # vector not already in Markov basis
                print(f'add basis vector {extra} for data point {i}')
                lm.basis = np.vstack([lm.basis, extra])
    return make_gibbs_fn(lm_list)

gibbs_fn = make_gibbs_fn(lm_list)
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
    zs = pm.Multinomial('z', n=ns, p=ps)

t = time()
idata, mcmc = sample_numpyro_nuts_gibbs(gibbs_fn, ['z'], ['p'],
                                        draws=draws, tune=tune, thinning=thinning, chains=chains, target_accept=0.95,
                                        random_seed=2023, model=model,
                                        initvals=[{'z': np.array([lm.inits[c] for lm in lm_list])} for c in range(chains)],
                                        postprocessing_chunks=25,
                                        update_gibbs_fn=update_gibbs_fn)
mcmc_time = time() - t

idata.sample_stats.attrs['preprocess_time'] = pre_time
idata.sample_stats.attrs['mcmc_walltime'] = mcmc_time

ess = az.ess(idata, var_names=['p'])['p'].values
print(ess.min())

idata.posterior = idata.posterior.drop_vars('noise')
idata.to_netcdf(f'../../data/time-series/psize50_m3_gibbs.netcdf')

t_pred = np.arange(0, 20.001, 0.01)
N_pred = len(t_pred)
with model:
    fs_pred = [gps[h].marg_cond(f'f_pred{h}', Xnew=t_pred[:,None]) for h in range(H)]
    ps_pred = pm.Deterministic(f'p_pred', pm.math.softmax(pt.stack(fs_pred, axis=1)
                                               *alpha + mean
                                               + sigma*pm.Normal('noise_pred', shape=(N_pred,H)),
                                               axis=-1))
    pred_idata = pm.sample_posterior_predictive(idata, var_names=['p_pred'])

pred_samples = np.vstack(pred_idata.posterior_predictive.p_pred)
np.save(f'../../data/time-series/psize50_m3_gibbs_pred_samples.npy', pred_samples)
np.save('../../data/time-series/psize50_m3_gibbs_pred_samples_tint.npy',
pred_samples[:,::100])

# pred_samples = np.load(f'../../data/time-series/psize50_m3_gibbs_pred_samples.npy')

sumstats = {}
sumstats['mean'] = pred_samples.mean(axis=0)
sumstats['sd'] = pred_samples.std(axis=0)
sumstats['median'] = np.median(pred_samples, axis=0)
sumstats['mad'] = np.median(np.abs(pred_samples - sumstats['median'][None,:,:]), axis=0)
sumstats['quantiles'] = np.quantile(pred_samples, np.arange(0, 1.01, 0.05), axis=0)

with open(f'../../data/time-series/psize50_m3_gibbs_sumstats.pkl', 'wb') as fp:
    pkl.dump(sumstats, fp)