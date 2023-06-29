from collections import Counter, defaultdict as dd

import numpy as np
import scipy.stats
from scipy.special import softmax, gammaln, logsumexp
import pulp
import sympy
from sympy.polys.domains import ZZ
from sympy.polys.matrices import DM
import math
from time import time
import xarray

import pymc as pm
from pymc import modelcontext
from pymc.distributions.dist_math import check_parameters, factln, logpow
from pymc.blocking import RaveledVars
from pymc.math import logdet
from pymc.step_methods.arraystep import metrop_select, ArrayStepShared
from pymc.step_methods.metropolis import delta_logp
import pymc.sampling_jax
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
from numpyro.util import progress_bar_factory
import numpyro.distributions as dist
from numpyro.infer import MCMC, HMC, NUTS, HMCGibbs
from numpyro.infer.hmc import hmc
from numpyro.infer.hmc_gibbs import HMCGibbsState
from haplm.numpyro_util import sample_numpyro_nuts

from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import sys
import copy

from atpbar import atpbar, register_reporter, find_reporter, flush
import multiprocessing


def latent_mult_mcmc(lm_list, H, n_sample, n_burnin, methods,
                     logprior_func=None, target_accept=0.9, jaxify=False,
                     chains=5, seed=None):   
    # check if no need for approximation nor discrete sampling
    for i, lm in enumerate(lm_list):
        if not lm.idx_var.size or not lm.amat_var.size:
            methods[i] = 'exact'
    
    exact_is = [i for i, method in enumerate(methods) if method=='exact']
    basis_is = [i for i, method in enumerate(methods) if method=='basis']
    mn_is = [i for i, method in enumerate(methods) if method=='mn_approx']
    assert len(exact_is)+len(basis_is)+len(mn_is) == len(lm_list)

    logfacts = scipy.special.gammaln(np.arange(1, max(lm.n_var for lm in lm_list)+2))

    ### jax, with grad

    # if logprior_func is None:
    #     logjoint = lambda p: (sum(lm_list[i].loglike_mn_jax(p) for i in mn_is)
    #                       + sum(lm_list[i].loglike_exact_jax(p, logfacts) for i in exact_is))
    # else:
    #     logjoint = lambda p: (logprior_func(p)
    #                       + sum(lm_list[i].loglike_mn_jax(p) for i in mn_is)
    #                       + sum(lm_list[i].loglike_exact_jax(p, logfacts) for i in exact_is))
    # val_and_grad = jax.value_and_grad(logjoint)
    # class LogjointOp(Op):
    #     default_output = 0

    #     def make_node(self, *inputs):
    #         inputs = [pt.as_tensor_variable(inputs[0])]
    #         outputs = [pt.dscalar()] + [inp.type() for inp in inputs]
    #         return Apply(self, inputs, outputs)

    #     def perform(self, node, inputs, outputs):
    #         result, grad = val_and_grad(inputs[0])
    #         outputs[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)
    #         outputs[1][0] = np.asarray(grad, dtype=node.outputs[1].dtype)

    #     def grad(self, inputs, output_gradients):
    #         value = self(*inputs)
    #         gradients = value.owner.outputs[1:]
    #         assert all(
    #             isinstance(g.type, pytensor.gradient.DisconnectedType) for g in output_gradients[1:]
    #         )
    #         return [output_gradients[0] * grad for grad in gradients]

    # logjoint = LogjointOp()

    # @jax_funcify.register(LogjointOp)
    # def logjoint_dispatch(op, **kwargs):
    #     return val_and_grad

    if jaxify:
        if logprior_func is None:
            logjoint_fn = lambda p: (sum(lm_list[i].loglike_mn_jax(p) for i in mn_is)
                              + sum(lm_list[i].loglike_exact_jax(p, logfacts) for i in exact_is))
        else:
            logjoint_fn = lambda p: (logprior_func(p)
                              + sum(lm_list[i].loglike_mn_jax(p) for i in mn_is)
                              + sum(lm_list[i].loglike_exact_jax(p, logfacts) for i in exact_is))

        class LogjointOp(Op):
            default_output = 0

            def make_node(self, p,):
                inputs = [p]
                outputs = [pt.dscalar()]
                return Apply(self, inputs, outputs)

            def perform(self, node, inputs, outputs):
                p, = inputs
                outputs[0][0] = np.asarray(logjoint_fn(p), dtype=node.outputs[0].dtype)

        logjoint = LogjointOp()

        @jax_funcify.register(LogjointOp)
        def logjoint_dispatch(op, **kwargs):
            return logjoint_fn  

    else:
        if logprior_func is None:
            def logjoint(p):
                return (sum(lm_list[i].loglike_mn(p) for i in mn_is)
                        + sum(lm_list[i].loglike_exact(p, logfacts) for i in exact_is))
        else:
            def logjoint(p):
                return (logprior_func(p)
                        + sum(lm_list[i].loglike_mn(p) for i in mn_is)
                        + sum(lm_list[i].loglike_exact(p, logfacts) for i in exact_is))

    # inference

    if basis_is:
        raise NotImplementedError

    with pm.Model() as model:
        p = pm.Dirichlet('p', np.ones(H))
        logjoint_var = pm.Potential('logjoint', logjoint(p))
    
    with model:
        idata = pm.sampling_jax.sample_numpyro_nuts(draws=n_sample,
                                                    tune=n_burnin,
                                                    chains=chains,
                                                    target_accept=target_accept,
                                                    random_seed=seed,
                                                    postprocessing_chunks=10)

        # idata = pm.sample(draws=n_sample, tune=n_burnin,
        #                   step=[pm.NUTS(target_accept=target_accept)],
        #                   nuts_sampler='numpyro',
        #                   chains=chains, cores=chains, random_seed=seed,
        #                   compute_convergence_checks=False)
    print('', file=sys.stderr)

    return idata  


def latent_mult_mcmc_cgibbs(lm_list, H, n_sample, n_burnin, cyc_len,
                            alphas=None, chains=5, cores=1, seed=None):
    if alphas is None:
        alphas = np.ones(H)
    else:
        alphas = np.array(alphas)
    basis_is = [i for i, lm in enumerate(lm_list) if lm.idx_var.size]

    seeds = np.random.default_rng(seed).integers(2**30, dtype=np.int64, size=2*chains)

    # find low ESS haplotypes  
    n_remove = n_burnin // 2
    mcmc = partial(latent_mult_sample_cgibbs, lm_list, H, n_burnin - n_remove, n_remove, alphas, cyc_len)

    print('Burn-in phase...')
    if cores==1:
        outputs_burnin = [mcmc(c, s) for c, s in zip(range(chains), seeds[:chains])]    
    else:        
        reporter = find_reporter()
        with multiprocessing.Pool(cores, register_reporter, [reporter]) as p:
            outputs_burnin = p.starmap(mcmc, list(zip(range(chains), seeds[:chains])))
            flush()

    print('Augmenting bases...')
    t = time()
    posterior = {f'z{i}': (('chain', 'draw', 'z_dim'),
                           np.array([output[f'trace_z{i}'] for output in outputs_burnin]))
                for i in basis_is}
    ess = az.ess(xarray.Dataset(posterior))

    for i, lm in enumerate(lm_list):
        if not lm.idx_var.size:
            continue
        nz = len(lm.idx_var)
        zess = ess[f'z{i}'].values[lm.idx_var]
        # indices of zess in increasing order
        order = list(np.argsort(zess))
        # cluster indices into two groups by ESS
        dist_best = 0.2*np.sum(np.abs(zess-np.median(zess)))
        lowess_list = order # indices for the group with low ESS
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
        if not nspace:
            continue
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

    extend_t = time()-t

    # extend basis and set starting points
    for idx, lm in enumerate(lm_list):
        if not lm.idx_var.size:
            continue
        lm.inits = np.array([output[f'trace_z{idx}'][-1] for output in outputs_burnin])

    # inference run
    mcmc = partial(latent_mult_sample_cgibbs, lm_list, H, n_sample, 0, alphas, cyc_len)

    print('Inference phase...')
    if cores==1:
        outputs = [mcmc(c, s) for c, s in zip(range(chains), seeds[chains:])]    
    else:        
        reporter = find_reporter()
        with multiprocessing.Pool(cores, register_reporter, [reporter]) as p:
            outputs = p.starmap(mcmc, list(zip(range(chains), seeds[chains:])))
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

    idata = az.InferenceData(posterior=xarray.Dataset(posterior),
                             sample_stats=xarray.Dataset(sample_stats))
    idata.attrs['extend_time'] = extend_t
    return idata


def nbor_ps(z0, signed_basis, a0, logfacts):
    nbors = z0 + signed_basis
    nbors = nbors[np.all(nbors >= 0, axis=1)]
    return nbors, logfacts[nbors].sum(axis=1)


def latent_mult_sample_cgibbs(lm_list, H, n_sample, n_burnin, alphas,
                              cyc_len, chain=0, seed=None): 
    t = time()
    
    basis_is = [i for i, lm in enumerate(lm_list) if lm.idx_var.size]
    logfacts = gammaln(np.arange(1,max(lm_list[i].n_var for i in basis_is)+2))
    rng = np.random.default_rng(seed)

    zs = [lm_list[i].inits[chain,lm_list[i].idx_var] for i in basis_is]
    bsizes = [len(lm_list[i].basis) for i in basis_is]
    ncols = [len(lm_list[i].idx_var) for i in basis_is]
    signed_bases = [np.vstack([-lm_list[i].basis, lm_list[i].basis]) for i in basis_is]

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
        nbor_arr[idx], nbps_arr[idx] = nbor_ps(z, signed_bases[idx],
                                               post_alp[lm_list[j].idx_var] - z,
                                               logfacts)

    nz = len(basis_is)
    idxs = list(range(nz))
    ns_var = np.array([lm_list[i].n_var for i in basis_is])
    p_cuml = np.cumsum(ns_var)

    zacc_arr = np.zeros(nz, int)
    moves_arr = np.zeros(nz, int)
    
    trace_p = []
    trace_zs = [[] for _ in basis_is]
    temp_init = 1000
    temp_mult = temp_init / n_burnin if n_burnin else 0
    for i in atpbar(range(n_sample + n_burnin),
                    time_track=True,
                    name=f'chain {chain}'):

        temp = max(1, temp_init / (1+temp_mult*i)) if n_burnin else 1
        if i == n_burnin:
            zacc_arr = np.zeros(nz, int)   
            moves_arr = np.zeros(nz, int)
            post_burnin = time()

        # update z     
        r = p_cuml[-1] * (1 - rng.random(size=cyc_len))
        rand_idxs = np.searchsorted(p_cuml, r).astype(int)
        mh_rands = np.log(rng.random(size=cyc_len))

        for idx, mh_rand in zip(rand_idxs, mh_rands): 
            j = basis_is[idx]
            nbor = nbor_arr[idx]

            a0 = post_alp[lm_list[j].idx_var] - zs[idx]
            currp_arr = gammaln(a0 + nbor).sum(axis=1) - nbps_arr[idx]
            q = nbor[np.argmax(currp_arr - np.log(-np.log(rng.random(size=nbor.shape[0]))))]

            q_nbors, q_nbps = nbor_ps(q, signed_bases[idx], a0, logfacts)
            qp_arr = gammaln(a0 + q_nbors).sum(axis=1) - q_nbps
            #accept = logsumexp(currp_arr) - logsumexp(qp_arr)
            
            currmax = max(currp_arr)
            qmax = max(qp_arr)
            accept = np.log(np.exp(currp_arr-currmax).sum()/np.exp(qp_arr-qmax).sum()*np.exp(currmax-qmax)) / temp

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
                lm = lm_list[basis_is[idx]]
                tmp = lm.inits[chain].copy()
                tmp[lm.idx_var] = zs[idx]
                trace_zs[idx].append(tmp)

    output = {'trace_p': np.array(trace_p),
              'zacc_rate': zacc_arr/moves_arr,
              'time_excl_tune': time()-post_burnin,
              'time_incl_tune': time()-t}
    output.update({f'trace_z{basis_is[idx]}': trace_zs[idx] for idx in range(nz)})

    #print(sorted(Counter([tuple(arr) for arr in trace_zs[basis_is.index(3)]]).items(), key=lambda t: -t[1]))
    #print(sorted(Counter([tuple(arr) for arr in acc_moves_arr[basis_is.index(3)]]).items(), key=lambda t: -t[1]))

    return output


def hier_latent_mult_mcmc(p, lm_list, H, n_sample, n_burnin, methods, model=None, target_accept=0.9,
                          jaxify=False, chains=5, seed=None, postprocessing_chunks=None):  
    model = modelcontext(model)

    # check if no need for approximation nor discrete sampling
    for i, lm in enumerate(lm_list):
        if not lm.idx_var.size or not lm.amat_var.size:
            methods[i] = 'exact'
    
    exact_is = [i for i, method in enumerate(methods) if method=='exact']
    basis_is = [i for i, method in enumerate(methods) if method=='basis']
    mn_is = [i for i, method in enumerate(methods) if method=='mn_approx']
    assert len(exact_is)+len(basis_is)+len(mn_is) == len(lm_list)

    logfacts = scipy.special.gammaln(np.arange(1, max(lm.n_var for lm in lm_list)+2))

    if jaxify:
        logjoint_fn = lambda p: (sum(lm_list[i].loglike_mn_jax(p[i]) for i in mn_is)
                                 + sum(lm_list[i].loglike_exact_jax(p[i], logfacts) for i in exact_is))

        class LogjointOp(Op):
            default_output = 0

            def make_node(self, p,):
                inputs = [p]
                outputs = [pt.dscalar()]
                return Apply(self, inputs, outputs)

            def perform(self, node, inputs, outputs):
                p, = inputs
                outputs[0][0] = np.asarray(logjoint_fn(p), dtype=node.outputs[0].dtype)

        logjoint = LogjointOp()

        @jax_funcify.register(LogjointOp)
        def logjoint_dispatch(op, **kwargs):
            return logjoint_fn  

    else:
        logjoint = lambda p: (sum(lm_list[i].loglike_mn(p[i]) for i in mn_is)
                              + sum(lm_list[i].loglike_exact(p[i], logfacts) for i in exact_is))

    # inference

    if basis_is:
        raise NotImplementedError

    with model:
        logjoint_var = pm.Potential('logjoint', logjoint(p))
    
    with model:
        idata = sample_numpyro_nuts(draws=n_sample, tune=n_burnin, chains=chains, target_accept=target_accept,
                                    random_seed=seed, postprocessing_chunks=postprocessing_chunks)
    print('', file=sys.stderr)

    return idata  