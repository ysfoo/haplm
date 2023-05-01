from collections import Counter, defaultdict as dd

import numpy as np
import scipy.stats
from scipy.special import softmax, gammaln, logsumexp
import pulp
import sympy
import math
from time import time
import xarray

import pymc as pm
from pymc.distributions.dist_math import check_parameters, factln, logpow
from pymc.blocking import RaveledVars
from pymc.math import logdet
from pymc.step_methods.arraystep import metrop_select, ArrayStepShared
from pymc.step_methods.metropolis import delta_logp
import arviz as az
from arviz.data.base import make_attrs
from pymc.backends.arviz import find_constants, find_observations

import pytensor
from pytensor.tensor.random.op import RandomVariable, default_supp_shape_from_params
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, HMC, NUTS, HMCGibbs
from numpyro.infer.hmc import hmc
from numpyro.infer.hmc_gibbs import HMCGibbsState

from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import sys
import copy

from atpbar import atpbar, register_reporter, find_reporter, flush
import multiprocessing


def latent_mult_chain(c, s):
    return latent_mult_mcmc.mcmc(c, s)

def latent_mult_cgibbs_chain(c, s):
    return latent_mult_mcmc_cgibbs.mcmc(c, s)


class UniformDirichlet(dist.Distribution):
    support = dist.constraints.simplex

    def __init__(self, size):
        if isinstance(size, int):
            size = (size,)
        self.concentration = jnp.ones(size)
        super().__init__(batch_shape=size[:-1], event_shape=size[-1:])

    def sample(self, key, sample_shape=()):
        assert dist.util.is_prng_key(key)
        shape = sample_shape + self.batch_shape
        samples = jax.random.dirichlet(key, self.concentration, shape=shape)
        return jnp.clip(
            samples, a_min=jnp.finfo(samples).tiny, a_max=1 - jnp.finfo(samples).eps
        )

    def log_prob(self, value):
        return 0


class LatentMultApprox(dist.Distribution):
    support = dist.constraints.nonnegative_integer

    def __init__(self, p, lm, mn_stab=1e-9):
        self.p = p
        self.all_unique = lm.idx_var.size == 0
        self.no_info = lm.amat_var.size == 0
        self.idx_var = jnp.array(lm.idx_var)
        self.idx_fix = jnp.array(lm.idx_fix)
        self.z_fix = jnp.array(lm.z_fix)
        self.amat_var = jnp.array(lm.amat_var)
        self.n_var = lm.n_var
        self.y_var = jnp.array(lm.y_var)
        self.mn_stab = mn_stab
        super().__init__(batch_shape=(len(lm.y),), event_shape=())

    def sample(self, key, sample_shape=()):
        raise NotImplementedError

    def log_prob(self, value):
        # return self.lm.loglike_mn_jax(self.p, self.mn_stab)
        if self.all_unique: # all counts can be uniquely determined
            return jnp.sum(jsp.special.xlogy(self.z_fix, self.p))

        qsum = jnp.sum(self.p[self.idx_var])

        if self.no_info: # no info on counts that cannot be uniquely determined
            return (jnp.sum(jsp.special.xlogy(self.z_fix, self.p[self.idx_fix]))
                    + jsp.special.xlogy(self.n_var, qsum))

        q = self.p[self.idx_var]/qsum
        aq = jnp.dot(self.amat_var, q)
        delta = self.y_var - self.n_var*aq # observed - mean
        chol = jsp.linalg.cholesky(self.n_var*(jnp.dot(self.amat_var*q, self.amat_var.T) 
                                   - jnp.outer(aq, aq) + jnp.eye(len(self.y_var))*self.mn_stab),
                                   lower=True)
        chol_inv_delta = jsp.linalg.solve_triangular(chol, delta, lower=True)
        quadform = jnp.dot(chol_inv_delta, chol_inv_delta)
        logdet = jnp.sum(jnp.log(jnp.diag(chol)))

        log_prob = (jnp.sum(jsp.special.xlogy(self.z_fix, self.p[self.idx_fix]))
                    + jsp.special.xlogy(self.n_var, qsum) - 0.5*quadform - logdet)

        # jax.debug.print('{lp} {q} {ld} {pmin}',
        #                 lp=log_prob, q=quadform, ld=logdet, pmin=jnp.min(self.p))

        return log_prob


class LatentMultExact(dist.Distribution):
    support = dist.constraints.nonnegative_integer

    def __init__(self, p, lm):
        self.p = p
        self.no_info = lm.amat_var.size == 0
        self.idx_var = jnp.array(lm.idx_var)
        self.idx_fix = jnp.array(lm.idx_fix)
        self.z_fix = jnp.array(lm.z_fix)
        self.n_var = lm.n_var
        self.sols = jnp.array(lm.sols)
        self.logfacts = jsp.special.gammaln(jnp.arange(1, lm.n_var + 2))
        super().__init__(batch_shape=(len(lm.y),), event_shape=())

    def sample(self, key, sample_shape=()):
        raise NotImplementedError

    def log_prob(self, value):
        if self.no_info: # no info on counts that cannot be uniquely determined
            return (jnp.sum(jsp.special.xlogy(self.z_fix, self.p[self.idx_fix]))
                    + jsp.special.xlogy(self.n_var, jnp.sum(self.p[self.idx_var])))

        return jsp.special.logsumexp((jsp.special.xlogy(self.sols, self.p)-self.logfacts[self.sols]).sum(axis=-1))


def latent_mult_numpyro(lm_list, H, n_sample, n_burnin, methods,
                        p_dist, chains=5, cores=1, seed=None):  
    # check if no need for approximation nor discrete sampling
    for i, lm in enumerate(lm_list):
        if not lm.idx_var.size or not lm.amat_var.size:
            methods[i] = 'exact'

    exact_is = [i for i, method in enumerate(methods) if method=='exact']
    basis_is = [i for i, method in enumerate(methods) if method=='basis']
    mn_is = [i for i, method in enumerate(methods) if method=='mn_approx']
    assert len(exact_is)+len(basis_is)+len(mn_is) == len(lm_list)

    def model():
        p = numpyro.sample('p', p_dist)
        for i in exact_is:
            y = numpyro.sample(f"y{i}", LatentMultExact(p, lm_list[i]), obs=0)
        for i in mn_is:
            y = numpyro.sample(f"y{i}", LatentMultApprox(p, lm_list[i]), obs=0)

    nuts_kernel = NUTS(model,
                       target_accept_prob=0.9,)
                       #init_strategy=numpyro.infer.init_to_median)
    print(numpyro.infer.util._get_model_transforms(model))
    mcmc = MCMC(nuts_kernel, num_warmup=n_burnin, num_samples=n_sample, num_chains=chains)
    mcmc.run(jax.random.PRNGKey(seed), init_params=init_params, extra_fields=(
                "num_steps",
                "potential_energy",
                "energy",
                "adapt_state.step_size",
                "accept_prob",
                "mean_accept_prob",
                "diverging",
            ),
    )    

    idata = az.from_numpyro(mcmc)

    return mcmc, idata


def latent_mult_mcmc(lm_list, H, n_sample, n_burnin, methods,
                     logprior_func=None,
                     chains=5, cores=1, seed=None):   
    if logprior_func is None:
        logprior_func = lambda p: 0 # pt.as_tensor_variable(0)
    
    if logprior_func is None:
        logprior_func = lambda p: 0 # pt.as_tensor_variable(0)
    
    # check if no need for approximation nor discrete sampling
    for i, lm in enumerate(lm_list):
        if not lm.idx_var.size or not lm.amat_var.size:
            methods[i] = 'exact'
    
    exact_is = [i for i, method in enumerate(methods) if method=='exact']
    basis_is = [i for i, method in enumerate(methods) if method=='basis']
    mn_is = [i for i, method in enumerate(methods) if method=='mn_approx']
    assert len(exact_is)+len(basis_is)+len(mn_is) == len(lm_list)

    class CustomRV(RandomVariable):
        name = "custom"
        ndim_supp = 1
        ndims_params = [1]
        dtype = "floatX"
        _print_name = ("Custom", "\\operatorname{Custom}")

        def __call__(self, param, size=None, **kwargs):
            return super().__call__(param, size=size, **kwargs)

    custom = CustomRV()

    class Custom(pm.distributions.multivariate.SimplexContinuous):
        rv_op = custom

        @classmethod
        def dist(cls, param, **kwargs):
            return super().dist([param], **kwargs)

        def moment(rv, size, param):
            return pt.ones(H)/H

        def logp(value, param):
            return logprior_func(value)

    logfacts = scipy.special.gammaln(np.arange(1, max(lm.n_var for lm in lm_list)+2))
    logp = lambda p: (sum(lm_list[i].loglike_mn_jax(p) for i in mn_is)
                      + sum(lm_list[i].loglike_exact_jax(p, logfacts) for i in exact_is))
    val_and_grad = jax.value_and_grad(logp)
    class LogpOp(Op):
        default_output = 0

        def make_node(self, *inputs):
            inputs = [pt.as_tensor_variable(inputs[0])]
            outputs = [pt.dscalar()] + [inp.type() for inp in inputs]
            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            result, grad = val_and_grad(inputs[0])
            outputs[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)
            outputs[1][0] = np.asarray(grad, dtype=node.outputs[1].dtype)

        def grad(self, inputs, output_gradients):
            value = self(*inputs)
            gradients = value.owner.outputs[1:]
            assert all(
                isinstance(g.type, pytensor.gradient.DisconnectedType) for g in output_gradients[1:]
            )
            return [output_gradients[0] * grad for grad in gradients]

    logp_op = LogpOp()

    @jax_funcify.register(LogpOp)
    def logp_dispatch(op, **kwargs):
        return val_and_grad

    # logp = lambda p: (sum(lm_list[i].loglike_mn(p) for i in mn_is)
    #                   + sum(lm_list[i].loglike_exact(p, logfacts) for i in exact_is))

    with pm.Model() as model:
        #p = pm.Dirichlet('p', np.ones(H))
        p = Custom('p', np.ones(H), shape=(H,))
        loglike = pm.Potential('loglike', logp_op(p))
        #loglike = pm.Potential('loglike', logp(p))
               
    if basis_is:
        raise NotImplementedError
    
    else:
        with model:
            idata = pm.sample(draws=n_sample, tune=n_burnin,
                              step=[pm.NUTS(target_accept=0.9)],
                              nuts_sampler='numpyro',
                              chains=chains, cores=cores, random_seed=seed,
                              compute_convergence_checks=False)

    return idata  


def latent_mult_mcmc_cgibbs(lm_list, H, n_sample, n_burnin, alphas=None, 
                            cycles=5, chains=5, cores=1, seed=None): 
                            #adapt=False, tune_itv=50, chains=5, cores=1, seeds=None):
    if alphas is None:
        alphas = np.ones(H)
    else:
        alphas = np.array(alphas)

    seeds = np.random.default_rng(seed).integers(2**30, dtype=np.int64, size=chains)
    # assert len(seeds) == chains
    assert n_burnin >= 200
    # latent_mult_mcmc_cgibbs.mcmc = lambda c, s: latent_mult_sample_cgibbs(lm_list, H, n_sample,
    #                                                                       n_burnin, alphas,
    #                                                                       chain=c, seed=s) 

    # find low ESS haplotypes  
    latent_mult_mcmc_cgibbs.mcmc = lambda c, s: latent_mult_sample_cgibbs_alt(lm_list, H, 100,
                                                                              n_burnin - 100, alphas, cycles,
                                                                              chain=c, seed=s)         
    if cores==1:
        outputs_burnin = [latent_mult_mcmc_cgibbs.mcmc(c, s) for c, s in zip(range(chains), seeds)]    
    else:        
        reporter = find_reporter()
        with multiprocessing.Pool(cores, register_reporter, [reporter]) as p:
            outputs_burnin = p.starmap(latent_mult_cgibbs_chain, list(zip(range(chains), seeds)))
            flush()

    posterior = {'p': (('chain', 'draw', 'p_dim'),
                       np.stack([output['trace_p'] for output in outputs_burnin]))}

    ess = az.ess(xarray.Dataset(posterior))['p'].values
    hsorted = np.argsort(ess)
    n_add = 0

    for lm in lm_list:
        nz = len(lm.idx_var)
        if nz == 0:
            continue
        amat = sympy.Matrix(np.vstack([np.ones(nz, int), lm.amat_var.astype(int)]))
        cols = []
        found = False
        # start from haplotype with lowest ESS
        for h in hsorted:
            # look through haplotype list
            for col, idx in enumerate(lm.idx_var):
                if h == idx:
                    cols.append(col)
                    if len(cols) >= 3:
                        # check nullspace
                        nulls = amat[:,cols].nullspace()
                        if nulls:
                            # found valid basis vector
                            found = True
                            # scale basis vector to be integer
                            vec = nulls[0]
                            denoms = [x.q for x in vec if type(x) == sympy.Rational]
                            if len(denoms) == 0:
                                arr = np.array(vec, int)
                            elif len(denoms) == 1:
                                arr = np.array(vec*denoms[0], int)
                            else:
                                arr = np.array(vec*sympy.ilcm(*denoms), int)
                            extra = np.zeros(nz, int)
                            extra[cols] = arr.T[0]
                            if extra.sum() != 0 or not (lm.amat_var.dot(extra) == 0).all():
                                print(cols)
                                print(amat[:,cols])
                                print(arr.T)
                                print(extra)
                                print(lm.idx_var)
                                print(lm.amat_var)
                                print(lm.amat_var.dot(extra))
                            assert extra.sum() == 0 and (lm.amat_var.dot(extra) == 0).all()
                            if {tuple(row) for row in lm.basis}.isdisjoint({tuple(extra), tuple(-extra)}):
                                # vector not already in Markov basis
                                n_add += 1
                                lm.basis = np.vstack([lm.basis, extra])
                            # haplotype column found
                            break
            if found:
                # found valid basis vector, but may or may not already be in Markov basis
                break
    print(f'Added extra basis vector with low ESS haplotypes for {n_add} data points')

    # set starting points
    for idx, lm in enumerate(lm_list):
        if not lm.idx_var.size:
            continue
        lm.inits = [output[f'trace_z{idx}'][-1] for output in outputs_burnin]

    # inference run
    latent_mult_mcmc_cgibbs.mcmc = lambda c, s: latent_mult_sample_cgibbs_alt(lm_list, H, n_sample, 0,
                                                                              alphas, cycles,
                                                                              chain=c, seed=s)         
    if cores==1:
        outputs = [latent_mult_mcmc_cgibbs.mcmc(c, s) for c, s in zip(range(chains), seeds)]    
    else:        
        reporter = find_reporter()
        with multiprocessing.Pool(cores, register_reporter, [reporter]) as p:
            outputs = p.starmap(latent_mult_cgibbs_chain, list(zip(range(chains), seeds)))
            flush()
        
    posterior = {}
    sample_stats = {}
    for key in outputs[0]:
        stacked = np.stack([output[key] for output in outputs])
        ndims = len(stacked.shape)
        if ndims > 1 and stacked.shape[1] == n_sample:
            dims = ['chain', 'draw', f'{key[6:]}_dim'][:ndims]
        else:
            dims = ['chain', 'znum'][:ndims]        
        
        if key[:6] == 'trace_':
            posterior[key[6:]] = (dims, stacked)
        else:
            sample_stats[key] = (dims, stacked)
    sample_stats['time_incl_tune'] = (('chain',), sample_stats['time_excl_tune'][1] +
                                      np.array([output['time_incl_tune'] for output in outputs_burnin]))

    return az.InferenceData(posterior=xarray.Dataset(posterior),
                            sample_stats=xarray.Dataset(sample_stats))
    

# def latent_mult_sample_cgibbs(lm_list, H, n_sample, n_burnin, alphas,
#                               chain=0, seed=None): 
#     t = time()
    
#     basis_is = [i for i, lm in enumerate(lm_list) if lm.idx_var]
#     logfacts = gammaln(np.arange(1,max(lm_list[i].n_var for i in basis_is)+2))
#     rng = np.random.default_rng(seed)
#     random.seed(seed)

#     zs = [lm_list[i].inits[chain] for i in basis_is]
    
#     post_alp = alphas.copy()
#     for lm in lm_list:
#         post_alp[lm.idx_fix] += lm.z_fix
#     for z, i in zip(zs, basis_is):
#         post_alp[lm_list[i].idx_var] += z
#     # init p
#     p0 = rng.dirichlet(post_alp)

#     nz = len(basis_is)

#     bsizes = [len(lm_list[i].basis) for i in basis_is]
#     ncols = [len(lm_list[i].idx_var) for i in basis_is]
    
#     trace_p = []
#     trace_zs = [[] for _ in basis_is]

#     simul = 5
#     thin = int(5*sum(lm_list[i].n_var for i in basis_is) / simul)

#     acc = 0
#     for i in atpbar(range(n_sample + n_burnin),
#                     time_track=True,
#                     name=f'chain {chain}'):
#         if i == n_burnin:
#             post_burnin = time()

#         for t in range(thin):
#             # update z
#             idxs = random.sample(list(range(nz)), simul)
#             step_rands = rng.random(simul)
#             deltas = []
#             delta_sum = np.zeros(H)
#             logfact_sum = 0
#             hratio = 0

#             for rand, idx in zip(step_rands, idxs):
#                 lm = lm_list[basis_is[idx]]
#                 basis = lm.basis
#                 nbors = np.vstack((zs[idx] - basis, zs[idx] + basis))
#                 valid = np.where(np.all(nbors >= 0, axis=1))[0]

#                 num = valid[int(valid.size*rand)]
#                 delta = -basis[num] if num < bsizes[idx] else basis[num-bsizes[idx]]            
#                 deltas.append(delta)
#                 delta_sum[lm.idx_var] += delta

#                 logfact_sum += logfacts[nbors[num]].sum()
#                 hratio += math.log(valid.size / np.all(nbors + delta >= 0, axis=1).sum())

#             lp_q0 = gammaln(post_alp).sum() - sum(logfacts[zs[idx]].sum() for idx in idxs)
#             lp_q = gammaln(post_alp + delta_sum).sum() - logfact_sum
#             if random.random() < np.exp(lp_q - lp_q0 + hratio):
#                 for idx, delta in zip(idxs, deltas):
#                     zs[idx] += delta
#                 acc += 1
#                 post_alp += delta_sum

#         if i >= n_burnin:
#             # update p
#             p0 = rng.dirichlet(post_alp)
#             trace_p.append(p0)

#             for idx in range(nz):
#                 trace_zs[idx].append(zs[idx].copy())

#         if i % 100 == 99:
#             # print(i+1, acc/100, file=sys.stderr)
#             acc = 0

#     output = {'trace_p': np.array(trace_p),
#               'time_excl_tune': time()-post_burnin,
#               'time_incl_tune': time()-t}
#     output.update({f'trace_z{basis_is[idx]}': np.array(trace_zs[idx]) for idx in range(nz)})

#     #print(sorted(Counter([tuple(arr) for arr in trace_zs[basis_is.index(3)]]).items(), key=lambda t: -t[1]))
#     #print(sorted(Counter([tuple(arr) for arr in acc_moves_arr[basis_is.index(3)]]).items(), key=lambda t: -t[1]))

#     return output


def nbor_ps(z0, basis, a0, logfacts):
    nbors = np.vstack((z0 - basis, z0 + basis))
    nbors = nbors[np.all(nbors >= 0, axis=1)]
    return nbors, logfacts[nbors].sum(axis=1)


def latent_mult_sample_cgibbs_alt(lm_list, H, n_sample, n_burnin, alphas,
                                  cycles=5, chain=0, seed=None): 
    t = time()
    
    basis_is = [i for i, lm in enumerate(lm_list) if lm.idx_var]
    logfacts = gammaln(np.arange(1,max(lm_list[i].n_var for i in basis_is)+2))
    rng = np.random.default_rng(seed)

    zs = [lm_list[i].inits[chain] for i in basis_is]
    bsizes = [len(lm_list[i].basis) for i in basis_is]
    psizes = np.array([lm_list[i].n_var for i in basis_is])
    ncols = [len(lm_list[i].idx_var) for i in basis_is]

    # sum latent counts
    post_alp = alphas.copy()
    for lm in lm_list:
        post_alp[lm.idx_fix] += lm.z_fix
    for z, i in zip(zs, basis_is):
        post_alp[lm_list[i].idx_var] += z

    # initialise neighbour info
    nbor_arr = [None for _ in basis_is]
    nbps_arr = [None for _ in basis_is]
    for idx, z in enumerate(zs):
        j = basis_is[idx]
        nbor_arr[idx], nbps_arr[idx] = nbor_ps(z, lm_list[j].basis,
                                               post_alp[lm_list[j].idx_var] - z,
                                               logfacts) 

    # init p
    p0 = rng.dirichlet(post_alp)

    nz = len(basis_is)
    idxs = list(range(nz))
    
    trg_moves = cycles
    n_sims = trg_moves*nz

    zacc_arr = np.zeros(nz, int)
    moves_arr = np.zeros(nz, int)
    
    trace_p = []
    trace_zs = [[] for _ in basis_is]
    for i in atpbar(range(n_sample + n_burnin),
                    time_track=True,
                    name=f'chain {chain}'):
        if i == n_burnin:
            zacc_arr = np.zeros(nz, int)   
            moves_arr = np.zeros(nz, int)
            post_burnin = time()

        # update z     
        rand_idxs = [int(r*nz) for r in rng.random(size=n_sims)]
        mh_rands = rng.random(size=n_sims)
        delta_rands = np.log(-np.log(rng.random(size=(n_sims, 2*max(bsizes)))))

        for idx, delta_rand, mh_rand in zip(rand_idxs, delta_rands, mh_rands): 
            j = basis_is[idx]

            a0 = post_alp[lm_list[j].idx_var] - zs[idx]
            currp_arr = gammaln(a0 + nbor_arr[idx]).sum(axis=1) - nbps_arr[idx]
            q = nbor_arr[idx][np.argmax(currp_arr - delta_rand[:len(currp_arr)])]

            q_nbors, q_nbps = nbor_ps(q, lm_list[j].basis, a0, logfacts)
            qp_arr = gammaln(a0 + q_nbors).sum(axis=1) - q_nbps
            #accept = logsumexp(currp_arr) - logsumexp(qp_arr)
            
            currmax = max(currp_arr)
            qmax = max(qp_arr)
            accept = np.exp(currp_arr-currmax).sum()/np.exp(qp_arr-qmax).sum()*np.exp(currmax-qmax)

            moves_arr[idx] += 1
            if mh_rand < accept:     
                zacc_arr[idx] += 1           
                zs[idx] = q  
                post_alp[lm_list[j].idx_var] = a0 + q
                nbor_arr[idx] = q_nbors
                nbps_arr[idx] = q_nbps

        if i >= n_burnin:
            # update p
            p0 = rng.dirichlet(post_alp)
            trace_p.append(p0)

            for idx in range(nz):
                trace_zs[idx].append(zs[idx])

    output = {'trace_p': np.array(trace_p),
              'zacc_rate': zacc_arr/moves_arr,
              'time_excl_tune': time()-post_burnin,
              'time_incl_tune': time()-t}
    output.update({f'trace_z{basis_is[idx]}': np.array(trace_zs[idx]) for idx in range(nz)})

    #print(sorted(Counter([tuple(arr) for arr in trace_zs[basis_is.index(3)]]).items(), key=lambda t: -t[1]))
    #print(sorted(Counter([tuple(arr) for arr in acc_moves_arr[basis_is.index(3)]]).items(), key=lambda t: -t[1]))

    return output

