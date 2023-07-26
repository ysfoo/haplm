"""
Functions for performing Bayesian inference with latent multinomial models.
"""

from collections import Counter, defaultdict as dd

import numpy as np
from scipy.special import gammaln
import pulp
import sympy
from sympy.polys.domains import ZZ
from sympy.polys.matrices import DM
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
from jax import jit
import jax.numpy as jnp

from haplm.numpyro_util import sample_numpyro_nuts, sample_numpyro_nuts_gibbs

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import sys

from atpbar import atpbar, register_reporter, find_reporter, flush
import multiprocessing


def latent_mult_mcmc(lm_list, H, n_sample, n_burnin, methods,
                     logprior_func=None, jaxify=False, **kwargs):   
    """
    Performs Bayesian inference using NUTS for latent multinomial distributions 
    which share the same multinomial probabilities. NUTS is called via 
    `pymc.sampling_jax.sample_numpyro_nuts`. The latent multinomial likelihood 
    for each data point is either calculated exactly via enumerating all 
    possible latent counts ('exact') or approximately ('mn_approx') via a 
    multinormal approximation, as specified through the `methods` argument.

    Parameters
    ----------
    lm_list : list[haplm.lm_dist.LatentMult]
        List of `LatentMult` objects storing information about the latent
        multinomial observations.
    H : int > 0
        Number of multinomial categories.
    n_sample : int > 0
        Number of inference iterations.
    n_burnin : int > 0
        Number of burn-in iterations.
    methods : list[str]
        List of strings that are either 'exact' or 'mn_approx', whose length is
        equal to that of `lm_list`, indicating whether the probability mass
        function of the corresponding latent multinomial distribution is
        calculated exactly or using a multinormal approximation.
    logprior_func : callable, optional
        Function that takes in a `pytensor.tensor.TensorVariable` vector as the
        multinomial probabilities and returns the log prior density. Defaults
        to a uniform prior if unspecified.
    jaxify : bool, default False
        Whether the log joint-likelihood density is computed through `jax` 
        functions wrapped in a `pytensor.graph.Op`. Setting this to `True` may 
        speed up compilation but slow down sampling.
    kwargs :
        Keyword arguments passed to `pymc.sampling_jax.sample_numpyro_nuts`,
        such as `chains`, `random_seed` and `target_accept`.

    Returns
    -------
    arviz.InferenceData
        An ArviZ ``InferenceData`` object that contains the posterior samples.
    """
    # if all latent counts are uniquely determined, use exact calculation
    for i, lm in enumerate(lm_list):
        if not lm.idx_var.size or not lm.amat_var.size:
            methods[i] = 'exact'
    
    exact_is = [i for i, method in enumerate(methods) if method=='exact']
    mn_is = [i for i, method in enumerate(methods) if method=='mn_approx']
    assert len(exact_is)+len(mn_is) == len(lm_list), (
        "Method names must be `exact` or `mn_approx`")

    # get log-factorial values for LatentMult with exact method
    if exact_is:
        logfacts = gammaln(np.arange(1, max(lm_list[i].n_var for i in exact_is)+2))

    # --- begin unused

    ### jax, with grad

    # import jax
    #
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

    # --- end unused

    # using JAX
    if jaxify:
        if logprior_func is None:
            logjoint_fn = lambda p: (
                              sum(lm_list[i].loglike_mn_jax(p) for i in mn_is)
                              + sum(lm_list[i].loglike_exact_jax(p, logfacts) for i in exact_is))
        else:
            logjoint_fn = lambda p: (logprior_func(p)
                              + sum(lm_list[i].loglike_mn_jax(p) for i in mn_is)
                              + sum(lm_list[i].loglike_exact_jax(p, logfacts) for i in exact_is))

        # based on https://www.pymc.io/projects/examples/en/latest/case_studies/wrapping_jax_function.html
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

    # not using JAX
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
    with pm.Model() as model:
        p = pm.Dirichlet('p', np.ones(H))
        logjoint_var = pm.Potential('logjoint', logjoint(p))
        idata = pm.sampling_jax.sample_numpyro_nuts(draws=n_sample, tune=n_burnin, **kwargs)

    print('', file=sys.stderr)

    return idata  


def latent_mult_mcmc_cgibbs(lm_list, H, n_sample, n_burnin, chains, 
                            cyc_len=None, alphas=None, cores=1, random_seed=None):
    """
    Performs Bayesian inference for latent multinomial distributions which share 
    the same multinomial probabilities using a collapsed Metropolis-within-Gibbs
    algorithm to sample the latent counts. This is an exact method that does 
    not require all possible latent counts to be enumerated. However, the prior
    must be a Dirichlet distribution.

    Between the burn-in phase and the inference phase, the basis from which 
    proposal directions are sampled from is augmented. The augmented vectors are 
    determined by the LLL algorithm performed on the columns of the 
    configuration matrix that correspond to low ESS during the second half of 
    the burn-in phase. Also, prior annealing is used for the first half of the 
    burn-in phase.

    Parameters
    ----------
    lm_list : list[haplm.lm_dist.LatentMult]
        List of `LatentMult` objects storing information about the latent
        multinomial observations.
    H : int > 0
        Number of multinomial categories.
    n_sample : int > 0
        Number of inference iterations.
    n_burnin : int > 0
        Number of burn-in iterations.
    chains : int > 0
        The number of MCMC chains to run.
    cyc_len : int > 0, optional
        Number of latent count updates per MCMC iteration. Set to 5 times the 
        sum of all pool sizes if unspecified.
    alphas : list[float > 0], optional
        List of Dirichlet concentrations for the prior to the multinomial 
        probabilities. Set to a uniform prior if unspecified.
    cores : int > 0, default 1
        The number of chains to run in parallel.
    random_seed : int, optional
        Seed for `numpy` random generation.

    Returns
    -------
    arviz.InferenceData
        An ArviZ ``InferenceData`` object that contains the posterior samples.
    """
    # indices of `lm_list` where latent counts should be sampled
    basis_is = [i for i, lm in enumerate(lm_list) if lm.idx_var.size]

    if alphas is None:
        alphas = np.ones(H)
    else:
        alphas = np.array(alphas)
    
    # default value for number of latent count updates per MCMC iteration
    if cyc_len is None:
        cyc_len = 5*sum(lm_list[i].n_var for i in basis_is)

    # seeds for each chain
    seeds = np.random.default_rng(random_seed).integers(2**30, dtype=np.int64, size=2*chains)

    print('Burn-in phase...')
    t0 = time()

    # use second half of burn-in phase to determine multinomial categories with low ESS
    n_remove = n_burnin // 2
    mcmc = partial(_latent_mult_sample_cgibbs, lm_list, H, n_burnin - n_remove, n_remove, alphas, cyc_len)

    if cores==1:
        outputs_burnin = [mcmc(c, s) for c, s in zip(range(chains), seeds[:chains])]    
    else:        
        reporter = find_reporter()
        with multiprocessing.Pool(cores, register_reporter, [reporter]) as p:
            outputs_burnin = p.starmap(mcmc, list(zip(range(chains), seeds[:chains])))
            flush()

    print('Augmenting bases...')
    t1 = time()
    posterior = {f'z{i}': (('chain', 'draw', 'z_dim'),
                           np.array([output[f'trace_z{i}'] for output in outputs_burnin]))
                for i in basis_is}
    ess = az.ess(xarray.Dataset(posterior))

    for i, lm in enumerate(lm_list):
        # skip if no latent counts samples
        if not lm.idx_var.size:
            continue

        # number of categories sampled
        nz = len(lm.idx_var)
        zess = ess[f'z{i}'].values[lm.idx_var]

        # cluster indices into two groups by ESS
        # indices of zess in increasing order
        order = list(np.argsort(zess))
        # threshold for there to be considered two clusters
        dist_best = 0.2*np.sum(np.abs(zess-np.median(zess)))
        lowess_list = order # indices for the group with low ESS
        for div in range(1, len(order)):
            dist = sum(np.sum(np.abs(vals-np.median(vals)))
                       for vals in (zess[order[:div]], zess[order[div:]]))
            if dist < dist_best:
                dist_best = dist
                lowess_list = order[:div]

        amat = sympy.Matrix(np.vstack([np.ones(nz, int), lm.amat_var.astype(int)]))
        nullspace = []
        # sympy nullspace may be rationals instead, convert to integer vectors
        for vec in amat[:,lowess_list].nullspace():
            denoms = [x.q for x in vec]
            if len(denoms) == 1:
                arr = np.array(vec*denoms[0], int)
            else:
                arr = np.array(vec*sympy.ilcm(*denoms), int)
            nullspace.append(arr.T[0])
        if not nullspace:
            continue
        nullspace = np.vstack(nullspace)

        # find short basis vectors to nullspace using LLL algorithm
        reduced = DM(nullspace, ZZ).lll()
        extras = np.array(reduced.to_Matrix(), int)
        # skip vectors that have Euclidean norm > 5 as the accept prob. would often be low
        for tmp in extras:
            if np.linalg.norm(tmp) > 5:
                continue
            extra = np.zeros(nz, int)
            extra[lowess_list] = tmp
            if not (extra.sum() == 0 and (lm.amat_var.dot(extra) == 0).all()):
                print(lowess_list)
                print(lm.amat_var)
                print(extra)
                print(extras)
                import pickle as pkl
                with open('tmp.pkl', 'wb') as fp:
                    pkl.dump(lm, fp)
            assert (extra.sum() == 0 and (lm.amat_var.dot(extra) == 0).all()), 'Augmentation gave invalid vector, please report'
            if {tuple(row) for row in lm.basis}.isdisjoint({tuple(extra), tuple(-extra)}):
                # vector not already in Markov basis
                print(f'add basis vector {extra} for data point {i}')
                lm.basis = np.vstack([lm.basis, extra])

    time_augment = time()-t1

    # set starting points of inference phase to the last point of burn-in phase
    for idx, lm in enumerate(lm_list):
        if not lm.idx_var.size:
            continue
        lm.inits = np.array([output[f'trace_z{idx}'][-1] for output in outputs_burnin])

    print('Inference phase...')

    mcmc = partial(_latent_mult_sample_cgibbs, lm_list, H, n_sample, 0, alphas, cyc_len)

    if cores==1:
        outputs = [mcmc(c, s) for c, s in zip(range(chains), seeds[chains:])]    
    else:        
        reporter = find_reporter()
        with multiprocessing.Pool(cores, register_reporter, [reporter]) as p:
            outputs = p.starmap(mcmc, list(zip(range(chains), seeds[chains:])))
            flush()

    time_total = time()-t0
        
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

    idata = az.InferenceData(posterior=xarray.Dataset(posterior),
                             sample_stats=xarray.Dataset(sample_stats))
    idata.attrs['augment_time'] = time_augment
    idata.attrs['sampling_time'] = time_total - time_augment

    return idata


def _nbor_ps(z0, signed_basis, a0, logfacts):
    """
    Helper function for getting neighbours to a latent sample vector, and the 
    sum of log-factorials of each neighbour's latent counts.
    """
    nbors = z0 + signed_basis
    nbors = nbors[np.all(nbors >= 0, axis=1)]
    return nbors, logfacts[nbors].sum(axis=1)


def _latent_mult_sample_cgibbs(lm_list, H, n_sample, n_burnin, alphas,
                               cyc_len, chain=0, seed=None): 
    """
    Helper function for running a single chain of collapsed Metropolis-within-
    Gibbs to sample latent counts.
    """
    # indices of `lm_list` where latent counts should be sampled
    basis_is = [i for i, lm in enumerate(lm_list) if lm.idx_var.size]
    nz = len(basis_is)
    idxs = list(range(nz))

    # pre-compute log-factorial values
    logfacts = gammaln(np.arange(1,max(lm_list[i].n_var for i in basis_is)+2))
    rng = np.random.default_rng(seed)

    # current latent count values
    zs = [lm_list[i].inits[chain,lm_list[i].idx_var] for i in basis_is]
    # basis vectors and their negations for each LatentMult
    signed_bases = [np.vstack([-lm_list[i].basis, lm_list[i].basis]) for i in basis_is]

    # sum latent counts for posterior Dirichlet distribution
    post_alp = alphas.copy()
    for lm in lm_list:
        post_alp[lm.idx_fix] += lm.z_fix
    for z, i in zip(zs, basis_is):
        post_alp[lm_list[i].idx_var] += z

    # initialise neighbour info for each LatentMult
    nbor_arr = [None for _ in basis_is]
    nbps_arr = [None for _ in basis_is]
    for idx, z in enumerate(zs):
        j = basis_is[idx]
        nbor_arr[idx], nbps_arr[idx] = _nbor_ps(z, signed_bases[idx],
                                               post_alp[lm_list[j].idx_var] - z,
                                               logfacts)

    # precompute values for randomly selecting a latent count vector to update
    ns_var = np.array([lm_list[i].n_var for i in basis_is])
    p_cuml = np.cumsum(ns_var)

    # number of times each latent count vector is selected for updating
    zacc_arr = np.zeros(nz, int)
    moves_arr = np.zeros(nz, int)
    
    trace_p = []
    trace_zs = [[] for _ in basis_is]

    # temperature setting for prior annealing
    temp_init = 100
    temp_mult = temp_init / n_burnin if n_burnin else 0

    for i in atpbar(range(n_sample + n_burnin),
                    time_track=True,
                    name=f'chain {chain}'):

        # linear multiplicative cooling schedule
        temp = max(1, temp_init / (1+temp_mult*i)) if n_burnin else 1

        # reset acceptance and proposal counts for each LatentMult
        if i == n_burnin:
            zacc_arr = np.zeros(nz, int)   
            moves_arr = np.zeros(nz, int)

        # select latent count vector to propose update for
        r = p_cuml[-1] * (1 - rng.random(size=cyc_len))
        rand_idxs = np.searchsorted(p_cuml, r).astype(int)

        # log-uniform random values for M-H steps
        mh_rands = np.log(rng.random(size=cyc_len)) * temp

        for idx, mh_rand in zip(rand_idxs, mh_rands): 
            j = basis_is[idx]
            nbor = nbor_arr[idx]

            a0 = post_alp[lm_list[j].idx_var] - zs[idx]
            currp_arr = gammaln(a0 + nbor).sum(axis=1) - nbps_arr[idx]
            q = nbor[np.argmax(currp_arr - 
                               np.log(-np.log(rng.random(size=nbor.shape[0]))))]

            q_nbors, q_nbps = _nbor_ps(q, signed_bases[idx], a0, logfacts)
            qp_arr = gammaln(a0 + q_nbors).sum(axis=1) - q_nbps
            accept = (np.logaddexp.reduce(currp_arr) - 
                      np.logaddexp.reduce(qp_arr))
            
            # currmax = max(currp_arr)
            # qmax = max(qp_arr)
            # accept = np.log(np.exp(currp_arr-currmax).sum() /
            #                 np.exp(qp_arr-qmax).sum() *
            #                 np.exp(currmax-qmax)) / temp

            moves_arr[idx] += 1

            # M-H step
            if mh_rand < accept:     
                zacc_arr[idx] += 1           
                zs[idx] = q  
                post_alp[lm_list[j].idx_var] = a0 + q
                nbor_arr[idx] = q_nbors
                nbps_arr[idx] = q_nbps

        if i >= n_burnin:
            # sample and store p
            p0 = rng.dirichlet(post_alp)
            trace_p.append(p0)

            # store z
            for idx in range(nz):
                lm = lm_list[basis_is[idx]]
                tmp = lm.inits[chain].copy()
                tmp[lm.idx_var] = zs[idx]
                trace_zs[idx].append(tmp)

    output = {'trace_p': np.array(trace_p),
              'zacc_rate': zacc_arr/moves_arr}
    output.update({f'trace_z{basis_is[idx]}': trace_zs[idx] for idx in range(nz)})

    #print(sorted(Counter([tuple(arr) for arr in trace_zs[basis_is.index(3)]]).items(), key=lambda t: -t[1]))
    #print(sorted(Counter([tuple(arr) for arr in acc_moves_arr[basis_is.index(3)]]).items(), key=lambda t: -t[1]))

    return output


def hier_latent_mult_mcmc(p, lm_list, H, n_sample, n_burnin, methods, model=None, jaxify=False, **kwargs):
    """
    Performs Bayesian inference using NUTS for latent multinomial distributions 
    under a hierarchical PyMC model, where the multinomial probabilities are not
    shared. NUTS is called via a custom modification of 
    `pymc.sampling_jax.sample_numpyro_nuts`, which is implemented in
    `haplm.numpyro_util.sample_numpyro_nuts`, to handle a bug due to the use of
    GPs with the argument `postprocessing_chunks` in `sample_numpyro_nuts`. 
    The latent multinomial likelihood for each data point is either calculated 
    exactly via enumerating all possible latent counts ('exact') or 
    approximately ('mn_approx') via a multinormal approximation, as specified 
    through the `methods` argument.

    Parameters
    ----------
    p : iterable[pytensor.tensor.var.TensorVariable for 1D-array]
        Array of multinomial probability vectors, one for each latent 
        multinomial observation.
    lm_list : list[haplm.lm_dist.LatentMult]
        List of `LatentMult` objects storing information about the latent
        multinomial observations.
    H : int > 0
        Number of multinomial categories.
    n_sample : int > 0
        Number of inference iterations.
    n_burnin : int > 0
        Number of burn-in iterations.
    methods : list[str]
        List of strings that are either 'exact' or 'mn_approx', whose length is
        equal to that of `lm_list`, indicating whether the probability mass
        function of the corresponding latent multinomial distribution is
        calculated exactly or using a multinormal approximation.
    model : pymc.Model, optional if in `with` context
        PyMC model where `p` is defined.
    jaxify : bool, default False
        Whether the log joint-likelihood density is computed through `jax` 
        functions wrapped in a `pytensor.graph.Op`. Setting this to `True` may 
        speed up compilation but slow down sampling.
    kwargs :
        Keyword arguments passed to `haplm.numpyro_util.sample_numpyro_nuts`,
        such as `chains`, `random_seed`, `target_accept`, and 
        `postprocessing_chunks`. The last keyword argument can be set to a 
        positive integer to reduce memory usage at the end of MCMC sampling.

    Returns
    -------
    arviz.InferenceData
        An ArviZ ``InferenceData`` object that contains the posterior samples.
    """
    model = modelcontext(model)

    # if all latent counts are uniquely determined, use exact calculation
    for i, lm in enumerate(lm_list):
        if not lm.idx_var.size or not lm.amat_var.size:
            methods[i] = 'exact'
    
    exact_is = [i for i, method in enumerate(methods) if method=='exact']
    mn_is = [i for i, method in enumerate(methods) if method=='mn_approx']
    assert len(exact_is)+len(mn_is) == len(lm_list), (
        "Method names must be `exact` or `mn_approx`")

    # get log-factorial values for LatentMult with exact method
    if exact_is:
        logfacts = gammaln(np.arange(1, max(lm_list[i].n_var for i in exact_is)+2))

    # using JAX
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

    # not using JAX
    else:
        logjoint = lambda p: (sum(lm_list[i].loglike_mn(p[i]) for i in mn_is)
                              + sum(lm_list[i].loglike_exact(p[i], logfacts) for i in exact_is))

    # inference
    with model:
        logjoint_var = pm.Potential('logjoint', logjoint(p))

        idata = sample_numpyro_nuts(draws=n_sample, tune=n_burnin, **kwargs)

    print('', file=sys.stderr)

    return idata


def hier_latent_mult_mcmc_gibbs(p, lm_list, H, n_sample, n_burnin, chains, n_reps=None,
                                thinning=1, model=None, **kwargs):
    """
    Performs Bayesian inference using Metropolis-within-NUTS for latent multinomial distributions 
    under a hierarchical PyMC model, where the multinomial probabilities are not shared. The latent 
    counts are sampled alternatingly with the continuous parameters. The sampler is called through
    `haplm.numpyro_util.sample_numpyro_nuts_gibbs`.

    Parameters
    ----------
    p : iterable[pytensor.tensor.var.TensorVariable for 1D-array]
        Array of multinomial probability vectors, one for each latent multinomial observation.
    lm_list : list[haplm.lm_dist.LatentMult]
        List of `LatentMult` objects storing information about the latent multinomial observations.
    H : int > 0
        Number of multinomial categories.
    n_sample : int > 0
        Number of inference iterations.
    n_burnin : int > 0
        Number of burn-in iterations.
    chains : int > 0
        The number of MCMC chains to run.
    n_reps : list[int > 0]
        List of number of times to update each latent count vector during a MCMC iteration. Set to 5 
        times each pool size if unspecified.
    thinning : int > 0, default 1
        Positive integer that controls the fraction of post-warmup samples that are retained.
    model : pymc.Model, optional if in `with` context
        PyMC model where `p` is defined.
    kwargs :
        Keyword arguments passed to `haplm.numpyro_util.sample_numpyro_nuts_gibbs`, such as 
        `random_seed`, `target_accept`, and `postprocessing_chunks`. The last keyword argument can 
        be set to a positive integer to reduce memory usage at the end of MCMC sampling.

    Returns
    -------
    arviz.InferenceData
        An ArviZ ``InferenceData`` object that contains the posterior samples.
    """
    model = modelcontext(model)

    with model:
        zs = pm.Multinomial('z', n=np.array([lm.n for lm in lm_list]), p=p)

    N = len(lm_list)
    logfacts = jnp.array(gammaln(np.arange(1, max(lm.n_var for lm in lm_list)+2)))
    temp_init = 100
    temp_mult = 100 / n_burnin

    if n_reps is None:
        n_reps = [5*lm.n_var for lm in lm_list]

    def make_site_fn(lm_list, i):
        '''Create function for updating one latent count vector.'''
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
                ws = jnp.where((opts >= 0).all(axis=1), jnp.exp(jnp.dot(opts, logp_var) - logfacts[opts].sum(axis=1)), 0)
                return_new = jnp.log(jax.random.uniform(mh_key)) < jnp.log(ws0.sum()/ws.sum()) / temp
                #jax.debug.print('z {z}\nopts0 {opts0}\nws0 {ws0}', opts0=opts0, z=z, ws0=ws0)
                return jax.lax.cond(return_new,
                                    lambda x: (z, opts, ws, next_key),
                                    lambda x: (z0, opts0, ws0, next_key),
                                    None)
            
            opts0 = z0 + bvecs
            ws0 = jnp.where((opts0 >= 0).all(axis=1), jnp.exp(jnp.dot(opts0, logp_var) - logfacts[opts0].sum(axis=1)), 0)
            return z_full.at[idx_var].set(jax.lax.fori_loop(0, n_reps[i], fn_to_rep, (z0, opts0, ws0, key))[0])
            
        return site_fn


    def make_gibbs_fn(lm_list):
        '''Create function for updating all latent count vectors.'''
        gibbs_fns = []
        for i, lm in enumerate(lm_list):
            if not lm.idx_var.size:
                gibbs_fns.append(lambda z, logp, key: z)
            else:
                gibbs_fns.append(make_site_fn(lm_list, i))
        
        def tmp_gibbs_fn(hmc_i, rng_key, gibbs_sites, hmc_sites):
            keys = jax.random.split(rng_key, N)
            log_ps = jnp.log(hmc_sites['p'])
            new_z = {'z': jnp.array([gibbs_fns[i](hmc_i, gibbs_sites['z'][i], log_ps[i], keys[i]).astype('int64') 
                                     for i in range(N)])} 
            # jax.debug.print("{new_z}", new_z=new_z)
            return new_z
        
        return tmp_gibbs_fn


    def update_gibbs_fn(samples, n_burnin, gibbs_sites, gibbs_idxs):
        '''Augment Markov bases and call `make_gibbs_fn` again to get `gibbs_fn`.'''
        samples_idx = gibbs_idxs[gibbs_sites.index('z')]
        posterior = {'z': (('chain', 'draw', 'z_dim_0', 'z_dim_1'), samples[samples_idx])}
        ess = az.ess(xarray.Dataset(posterior).sel(draw=np.arange(n_burnin // 2, n_burnin)))['z'].values
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
                denoms = [x.q for x in vec]
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
                if not (extra.sum() == 0 and (lm.amat_var.dot(extra) == 0).all()):
                    print(lowess_list)
                    print(lm.amat_var)
                    print(extra)
                    print(extras)
                    import pickle as pkl
                    with open('tmp.pkl', 'wb') as fp:
                        pkl.dump(lm, fp)
                assert (extra.sum() == 0 and (lm.amat_var.dot(extra) == 0).all()), (
                        'Augmentation gave invalid vector, please report')
                if {tuple(row) for row in lm.basis}.isdisjoint({tuple(extra), tuple(-extra)}):
                    # vector not already in Markov basis
                    print(f'add basis vector {extra} for data point {i}')
                    lm.basis = np.vstack([lm.basis, extra])
        return make_gibbs_fn(lm_list)


    gibbs_fn = make_gibbs_fn(lm_list)

    idata, mcmc = sample_numpyro_nuts_gibbs(gibbs_fn, ['z'], [p.name], draws=n_sample, tune=n_burnin, 
                                            chains=chains, thinning=thinning, model=model,
                                            initvals=[{'z': np.array([lm.inits[c] for lm in lm_list])} 
                                                       for c in range(chains)],
                                            update_gibbs_fn=update_gibbs_fn, **kwargs)

    return idata