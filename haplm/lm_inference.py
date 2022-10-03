from collections import Counter, defaultdict as dd

import numpy as np
import scipy.stats
from scipy.special import softmax, gammaln, logsumexp
import pulp
import sympy
import random
import math
from time import time
import xarray

from typing import Any, Callable, Dict, List, Tuple
import pymc as pm
from pymc.aesaraf import compile_pymc
from pymc.distributions.dist_math import check_parameters, factln, logpow
from pymc.blocking import RaveledVars
from pymc.math import logsumexp, logdet
from pymc.step_methods.arraystep import metrop_select, ArrayStepShared
from pymc.step_methods.metropolis import delta_logp
import arviz as az
import aesara
from aesara.tensor.random.op import RandomVariable, default_supp_shape_from_params
import aesara.tensor as at
from aesara.graph.basic import vars_between

from atpbar import atpbar, register_reporter, find_reporter, flush
import multiprocessing


def latent_mult_chain(c, s):
    return latent_mult_mcmc.mcmc(c, s)

def latent_mult_cgibbs_chain(c, s):
    return latent_mult_mcmc_cgibbs.mcmc(c, s)


def latent_mult_mcmc(lm_list, H, n_sample, n_burnin, methods, 
                     logprior_func=None,
                     tune_itv=50, p_updates=5,
                     chains=5, cores=1, seeds=None):   
    if logprior_func is None:
        logprior_func = lambda p: 0
    
    if seeds is None:
        seeds = [None]*chains
    assert len(seeds) == chains        
    
    p_tensor = at.vector('p')
    
    # check if no need for approximation nor discrete sampling
    for i, lm in enumerate(lm_list):
        if not lm.idx_var or not lm.amat_var.size:
            methods[i] = 'exact'
    
    exact_is = [i for i, method in enumerate(methods) if method=='exact']
    basis_is = [i for i, method in enumerate(methods) if method=='basis']
    mn_is = [i for i, method in enumerate(methods) if method=='mn_approx']
    assert len(exact_is)+len(basis_is)+len(mn_is) == len(lm_list)
    
    loglike_contribs = []
    for i in exact_is:
        loglike_contribs.append(lm_list[i].loglike_exact(p_tensor))
    for i in mn_is:
        loglike_contribs.append(lm_list[i].loglike_mn(p_tensor))   
    for i in basis_is:
        lm = lm_list[i]
        loglike_contribs.append(at.dot(lm.z_fix, at.log(p_tensor[lm.idx_fix])))
        
    loglike_func = aesara.function([p_tensor], at.sum(loglike_contribs))
               
    if basis_is:
        latent_mult_mcmc.mcmc = lambda c, s: latent_mult_sample_basis(lm_list, H, n_sample, n_burnin,
                                                                      loglike_func, logprior_func, basis_is,
                                                                      tune_itv, p_updates, chain=c, seed=s)
    
    else:
        latent_mult_mcmc.mcmc = lambda c, s: latent_mult_sample(lm_list, H, n_sample, n_burnin,
                                                                loglike_func, logprior_func,
                                                                tune_itv, p_updates, chain=c, seed=s)  
        
    if cores==1:
        outputs = [latent_mult_mcmc.mcmc(c, s) for c, s in zip(range(chains), seeds)]
    
    else:        
        reporter = find_reporter()
        with multiprocessing.Pool(cores, register_reporter, [reporter]) as p:
            outputs = p.starmap(latent_mult_chain, list(zip(range(chains), seeds)))
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

    return az.InferenceData(posterior=xarray.Dataset(posterior),
                            sample_stats=xarray.Dataset(sample_stats))


def latent_mult_sample_basis(lm_list, H, n_sample, n_burnin,
                             loglike_func, logprior_func, basis_is,
                             tune_itv=50, p_updates=5, chain=0, seed=None):
    t = time()
    
    logfacts = gammaln(np.arange(1,max(lm_list[i].n_var for i in basis_is)+2))
    rng = np.random.default_rng(seed)
    random.seed(seed)
    
    p0 = rng.dirichlet(np.ones(H))
    loglike0 = loglike_func(p0) + logprior_func(p0)
    
    zs = [lm_list[i].inits[chain] for i in basis_is]
    logp_zs = [z*np.log(p0[lm_list[i].idx_var])-np.sum(logfacts[z]) for i, z in zip(basis_is, zs)]   
    
    acc = 0
    stab = 0.5
    scale = 0.1
    
    nz = len(basis_is)
    n_sims_arr = 10*np.ones(nz, int)
    max_step_arr = 1*np.ones(nz, int)
    zacc_arr = np.zeros(nz, int)   
    moves_arr = np.zeros(nz, int)
    trg_moves = 10
    bsizes = [len(lm_list[i].basis) for i in basis_is]
    ncols = [len(lm_list[i].idx_var) for i in basis_is]
    
    acc_moves_arr = [[] for _ in range(nz)]
    tune_accs_arr = [0]*nz
    
    trace_p = []
    trace_zs = [[] for _ in basis_is]
    lp = []
    for i in atpbar(range(n_sample + n_burnin),
                    time_track=True,
                    name=f'chain {chain}'):
        if i == n_burnin:
            post_burnin = time()
        if i%tune_itv == 0 and 0 < i < n_burnin:
            # tune p
            scale = tune_scale(scale, acc/(tune_itv*p_updates))
            acc = 0
            
            # tune z
            for idx in range(nz):
                if moves_arr[idx]:
                    max_step_arr[idx] = tune_max_step(max_step_arr[idx], zacc_arr[idx]/moves_arr[idx])
                n_sims_arr[idx] = tune_n_sims(n_sims_arr[idx], moves_arr[idx]/tune_itv, trg_moves)
                zacc_arr[idx] = 0
                moves_arr[idx] = 0
        
        # update p
        for _ in range(p_updates):
            prop_as0 = p0/scale + stab
            p = rng.dirichlet(prop_as0)
            prop_as = p/scale + stab
            
            loglike = loglike_func(p) + logprior_func(p)
            
            if math.log(random.random()) < (loglike - loglike0
                                            + sum(np.dot(z, np.log(p[lm_list[j].idx_var])-np.log(p0[lm_list[j].idx_var]))
                                                  for j, z in zip(basis_is, zs))
                                            - (np.dot(prop_as0-1, np.log(p))-gammaln(prop_as0).sum()) 
                                            + (np.dot(prop_as-1, np.log(p0))-gammaln(prop_as).sum())):
                p0 = p
                loglike0 = loglike
                acc += 1
        curr_lp = loglike0

        # update z        
        logp0 = np.log(p0)
        prob_prev = min(0.8, i/n_burnin - 0.2) # prob of using previous move as delta

        for idx in range(nz):            
            j = basis_is[idx]
            n_sims = n_sims_arr[idx]
            
            basis = lm_list[j].basis
            bsize = bsizes[idx]
            ncol = ncols[idx]
            
            logp_var = logp0[lm_list[j].idx_var]
            q0 = zs[idx]
            lp_q0 = np.dot(q0,logp_var)-sum(logfacts[qidx] for qidx in q0)
            
            # symmetric 
            if lm_list[j].markov:                
                deltas = basis[rng.integers(bsize, size=n_sims)]
            else:
                if bsize > 1:
                    geo_denom = math.log(1 - max(0.2, 1/math.sqrt(bsize)))
                deltas = []
                for m in range(n_sims):
                    rand = random.random()
                    if rand < prob_prev and tune_accs_arr[idx]:
                        deltas.append(acc_moves_arr[idx][int(rand/prob_prev*tune_accs_arr[idx])])
                    else:
                        sgn_dict = {}
                        delta = np.zeros(ncol, int)
                        repeats = 1 if bsize == 1 else int(math.log(random.random())/geo_denom) + 1
                        for _ in range(repeats):
                            bidx = int(random.random()*bsize)
                            sgn = sgn_dict.get(bidx)
                            if sgn is None:
                                sgn = sgn_dict[bidx] = random.getrandbits(1)
                            if sgn:
                                delta += basis[bidx]
                            else:
                                delta -= basis[bidx]
                            deltas.append(delta)
                
            step_rands = rng.random(size=n_sims)
            mh_rands = np.log(rng.random(size=n_sims))
            
            for delta, step_rand, mh_rand in zip(deltas, step_rands, mh_rands):                
                neg_step, pos_step = n_steps(q0, delta, max_step_arr[idx])                
                tot_step = neg_step + pos_step
                if tot_step == 0:
                    continue                
                
                moves_arr[idx] += 1
                rand_int = int(step_rand*tot_step)
                if rand_int < neg_step:
                    q = q0 - (rand_int+1)*delta
                else:
                    q = q0 + (tot_step-rand_int)*delta
                
                s1, s2 = n_steps(q, delta, max_step_arr[idx])
                
                lp_q = np.dot(q,logp_var)-sum(logfacts[qidx] for qidx in q)
                accept = lp_q - lp_q0 + np.log(tot_step / (s1+s2))
                if mh_rand < accept:
                    q0 = q
                    zacc_arr[idx] += 1
                    lp_q0 = lp_q  
                    if 0.1 < i/n_burnin < 1 and not lm_list[j].markov:
                        tune_accs_arr[idx] += 1
                        acc_moves_arr[idx].append(delta)
                
            zs[idx] = q0     
            curr_lp += lp_q0

        if i >= n_burnin:
            trace_p.append(p0)
            for idx in range(nz):
                trace_zs[idx].append(zs[idx])
            lp.append(curr_lp)

    output = {'trace_p': np.array(trace_p),
              'lp': np.array(lp),
              'p_scale': scale,
              'p_accrate': acc/(n_sample*p_updates),
              'n_sims': n_sims_arr,
              'max_step': max_step_arr,
              'zacc_rate': zacc_arr/moves_arr,
              'moves_avg': moves_arr/n_sample,
              'time_excl_tune': time()-post_burnin,
              'time_incl_tune': time()-t}
    output.update({f'trace_z{basis_is[idx]}': np.array(trace_zs[idx]) for idx in range(nz)})

    #print(sorted(Counter([tuple(arr) for arr in trace_zs[basis_is.index(3)]]).items(), key=lambda t: -t[1]))
    #print(sorted(Counter([tuple(arr) for arr in acc_moves_arr[basis_is.index(3)]]).items(), key=lambda t: -t[1]))

    return output

        
def latent_mult_sample(lm_list, H, n_sample, n_burnin,
                       logprior_func, loglike_func,
                       tune_itv=50, p_updates=5, chain=0, seed=None):    
    t = time()
    rng = np.random.default_rng(seed)
    p0 = rng.dirichlet(np.ones(H))
    #logp0 = np.log(p0)
    loglike0 = loglike_func(p0) + logprior_func(p0)
    acc = 0
    stab = 0.5
    scale = 0.1
    
    trace_p = []
    lp = []
    for i in atpbar(range(n_sample + n_burnin),
                    time_track=True,
                    name=f'chain {chain}'):
        if i == n_burnin:
            post_burnin = time()
        if i%tune_itv == 0 and 0 < i < n_burnin:
            scale = tune_scale(scale, acc/(tune_itv*p_updates))
            acc = 0
            
        for _ in range(p_updates):
            prop_as0 = p0/scale + stab
            p = rng.dirichlet(prop_as0)
            #logp = np.log(p)
            prop_as = p/scale + stab
            
            loglike = loglike_func(p) + logprior_func(p)
            if math.log(random.random()) < (loglike - loglike0
                                            - (np.dot(prop_as0-1, np.log(p))-gammaln(prop_as0).sum()) 
                                            + (np.dot(prop_as-1, np.log(p0))-gammaln(prop_as).sum())):
                p0 = p
                #logp0 = logp
                loglike0 = loglike
                acc += 1

        if i >= n_burnin:
            trace_p.append(p0)
            lp.append(loglike0)

    return {'trace_p': np.array(trace_p),
            'lp': np.array(lp),
            'p_scale': scale,
            'p_accrate': acc/(n_sample*p_updates),
            'time_excl_tune': time()-post_burnin,
            'time_incl_tune': time()-t}


def latent_mult_mcmc_cgibbs(lm_list, H, n_sample, n_burnin, alphas=None, adapt=False,
                           tune_itv=50, chains=5, cores=1, seeds=None):     
    if alphas is None:
        alphas = np.ones(H)
    else:
        alphas = np.array(alphas)

    if seeds is None:
        seeds = [None]*chains
    assert len(seeds) == chains
    
    latent_mult_mcmc_cgibbs.mcmc = lambda c, s: latent_mult_sample_cgibbs(lm_list, H, n_sample,
                                                                          n_burnin, alphas,
                                                                          adapt=adapt, tune_itv=tune_itv,
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

    return az.InferenceData(posterior=xarray.Dataset(posterior),
                            sample_stats=xarray.Dataset(sample_stats))

def latent_mult_sample_cgibbs(lm_list, H, n_sample, n_burnin, alphas,
                              adapt=False, tune_itv=50, chain=0, seed=None): 
    t = time()
    
    basis_is = [i for i, lm in enumerate(lm_list) if lm.idx_var]
    logfacts = gammaln(np.arange(1,max(lm_list[i].n_var for i in basis_is)+2))
    rng = np.random.default_rng(seed)
    random.seed(seed)
    
    basis_is = [i for i, lm in enumerate(lm_list) if lm.idx_var]
    zs = [lm_list[i].inits[chain] for i in basis_is]
    
    post_alp = alphas.copy()
    for lm in lm_list:
        post_alp[lm.idx_fix] += lm.z_fix
    for z, i in zip(zs, basis_is):
        post_alp[lm_list[i].idx_var] += z

    nz = len(basis_is)
    idxs = list(range(nz))
    n_sims_arr = 10*np.ones(nz, int)
    max_step_arr = 1*np.ones(nz, int)
    zacc_arr = np.zeros(nz, int)   
    moves_arr = np.zeros(nz, int)
    trg_moves = 10
    bsizes = [len(lm_list[i].basis) for i in basis_is]
    ncols = [len(lm_list[i].idx_var) for i in basis_is]
    
    if adapt:
        acc_moves_arr = [[] for _ in range(nz)]
        tune_accs_arr = [0]*nz
    
    trace_p = []
    trace_zs = [[] for _ in basis_is]
    for i in atpbar(range(n_sample + n_burnin),
                    time_track=True,
                    name=f'chain {chain}'):
        if i == n_burnin:
            post_burnin = time()
        if i%tune_itv == 0 and 0 < i < n_burnin:            
            # tune z
            for idx in range(nz):
                if moves_arr[idx]:
                    max_step_arr[idx] = tune_max_step(max_step_arr[idx], zacc_arr[idx]/moves_arr[idx])
                n_sims_arr[idx] = tune_n_sims(n_sims_arr[idx], moves_arr[idx]/tune_itv, trg_moves)
                zacc_arr[idx] = 0
                moves_arr[idx] = 0
        
        # update p
        p0 = rng.dirichlet(post_alp)

        # update z        
        random.shuffle(idxs)
        prob_prev = min(0.8, i/n_burnin - 0.2) if adapt else 0 # prob of using previous move as delta

        for idx in idxs:            
            j = basis_is[idx]
            n_sims = n_sims_arr[idx]
            
            basis = lm_list[j].basis
            bsize = bsizes[idx]
            ncol = ncols[idx]
            
            q0 = zs[idx]
            a0 = post_alp[lm_list[j].idx_var] - q0
            lp_q0 = gammaln(a0 + q0).sum() - logfacts[q0].sum()
            
            # symmetric 
            if lm_list[j].markov: 
                deltas = []
                for m in range(n_sims):
                    rand = random.random()               
                    if rand < prob_prev and tune_accs_arr[idx]:
                        deltas.append(acc_moves_arr[idx][int(rand/prob_prev*tune_accs_arr[idx])])
                    else:
                        deltas.append(basis[int((1-rand)/(1-prob_prev)*bsize)])
            else:
                if bsize > 1:
                    geo_denom = math.log(1 - max(0.2, 1/math.sqrt(bsize)))
                deltas = []
                for m in range(n_sims):
                    rand = random.random()
                    if rand < prob_prev and tune_accs_arr[idx]:
                        deltas.append(acc_moves_arr[idx][int(rand/prob_prev*tune_accs_arr[idx])])
                    else:
                        sgn_dict = {}
                        delta = np.zeros(ncol, int)
                        repeats = 1 if bsize == 1 else int(math.log(random.random())/geo_denom) + 1
                        for _ in range(repeats):
                            bidx = int(random.random()*bsize)
                            sgn = sgn_dict.get(bidx)
                            if sgn is None:
                                sgn = sgn_dict[bidx] = random.getrandbits(1)
                            if sgn:
                                delta += basis[bidx]
                            else:
                                delta -= basis[bidx]
                            deltas.append(delta)
                
            step_rands = rng.random(size=n_sims)
            mh_rands = np.log(rng.random(size=n_sims))
            
            for delta, step_rand, mh_rand in zip(deltas, step_rands, mh_rands):                
                neg_step, pos_step = n_steps(q0, delta, max_step_arr[idx])                
                tot_step = neg_step + pos_step
                if tot_step == 0:
                    continue                
                
                moves_arr[idx] += 1
                rand_int = int(step_rand*tot_step)
                if rand_int < neg_step:
                    q = q0 - (rand_int+1)*delta
                else:
                    q = q0 + (tot_step-rand_int)*delta
                
                s1, s2 = n_steps(q, delta, max_step_arr[idx])
                
                lp_q = gammaln(a0 + q).sum() - logfacts[q].sum()
                accept = lp_q - lp_q0 + np.log(tot_step / (s1+s2))
                if mh_rand < accept:
                    q0 = q
                    zacc_arr[idx] += 1
                    lp_q0 = lp_q  
                    if 0.1 < i/n_burnin < 1 and adapt:
                        tune_accs_arr[idx] += 1
                        acc_moves_arr[idx].append(delta)
                
            zs[idx] = q0     
            post_alp[lm_list[j].idx_var] = a0 + q0

        if i >= n_burnin:
            trace_p.append(p0)
            for idx in range(nz):
                trace_zs[idx].append(zs[idx])

    output = {'trace_p': np.array(trace_p),
              'n_sims': n_sims_arr,
              'max_step': max_step_arr,
              'zacc_rate': zacc_arr/moves_arr,
              'moves_avg': moves_arr/n_sample,
              'time_excl_tune': time()-post_burnin,
              'time_incl_tune': time()-t}
    output.update({f'trace_z{basis_is[idx]}': np.array(trace_zs[idx]) for idx in range(nz)})

    #print(sorted(Counter([tuple(arr) for arr in trace_zs[basis_is.index(3)]]).items(), key=lambda t: -t[1]))
    #print(sorted(Counter([tuple(arr) for arr in acc_moves_arr[basis_is.index(3)]]).items(), key=lambda t: -t[1]))

    return output


class MBasisMetropolis(pm.step_methods.metropolis.Metropolis):
    """PyMC Metropolis-Hastings sampling step where discrete random walk directions are given by a Markov basis."""

    name = "markov basis metropolis"
    default_blocked = False
    generates_stats = True
    stats_dtypes = [{"moves": int,
                     "accepted": int,
                     "max_step": int,
                     "n_sims": int,
                     #"time_a": float,
                     #"time_b": float,
                     #"time_c": float,
                    }]

    # same initial values as pm.step_methods.metropolis.Metropolis except S
    def __init__(self, vars, shared, proposal_dist,                 
                 n_sims=10, max_step=1, trg_moves=10,
                 tune=True, tune_interval=50, model=None,
                 mode=None, **kwargs):
        """
        Initialises the `basisMetropolis` object.
        
        Parameters
        ----------
        vars : list
            List of one latent multinomial vector.
        shared : list
            List of Aesara tensors that the latent multnomial vector depends on, e.g. `p_simplex__` if there the multinomial probability is named `p` and follows a Dirichlet distribution.
        proposal_dist : pm.step_methods.metropolis.Proposal object
            Proposal object to simulate Metropolis-Hastings proposals.
        n_sims : int
            Number of times a proposal is simulated. Increase to reduce autocorrelation between recorded samples at the cost of increased runtime. Adjusted during tuning phase if `tune=True` (default).
        max_step : int
            Maximum step size of proposal step. Increase to encourage exploration at the cost of a lower acceptance rate. Adjusted during tuning phase if `tune=True` (default).
        trg_moves : int
            Number of valid proposal directions during an MCMC iteration to aim for during tuning phase. Only used if `tune=True`.
        tune : bool
            Whether `n_sims` and `max_step` should be adjusted during tuning phase.
        tune_interval : int
            Number of iterations for how often `n_sims` and `max_step` should be adjusted.
        model : PyMC Model
            Optional model for sampling step. Defaults to None (taken from context).
        mode : string or `Mode` instance
            Compilation mode passed to Aesara functions
        """
        self.n_sims = n_sims
        self.max_step = max_step
        self.trg_moves = trg_moves
        
        # start copy from metropolis ---
        
        model = pm.modelcontext(model)
        initial_values = model.initial_point()
        
        assert len(vars) == 1
        z = vars[0]
        vars = [model.rvs_to_values.get(var, var) for var in vars]
            
        vars = pm.inputvars(vars)
        
        #initial_values_shape = [initial_values[v.name].shape for v in vars]
        #S = np.ones(int(sum(np.prod(ivs) for ivs in initial_values_shape)))
        
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        
        self.accepted = 0 # for tuning max_step
        self.moves = 0 # for tuning n_sims
        
        # Determine type of variables
        # print(vars)
        self.discrete = np.concatenate(
            [[v.dtype in pm.discrete_types] * (initial_values[v.name].size or 1) for v in vars]
        )
        self.any_discrete = self.discrete.any()
        self.all_discrete = self.discrete.all()       
        
        assert self.all_discrete
        
        # remember initial settings before tuning so they can be reset
        self._untuned_settings = dict(
            steps_until_tune=tune_interval, accepted=0, moves=0,
        )

        self.mode = mode
        
        shared = {
            var: aesara.shared(initial_values[var.name], var.name + "_shared", broadcastable=var.broadcastable)
            for var in shared
        }
        
        # tensor_type = vars[0].type
        # inarray0 = tensor_type("inarray0")
        # inarray1 = tensor_type("inarray1")
        
        logp = pm.distributions.joint_logp(z, model.rvs_to_values[z], sum=False)[0]
        
        self.delta_logp = delta_logp(initial_values, logp, vars, shared)
        ArrayStepShared.__init__(self, vars, shared)        
        
        # --- end copy from metropolis
        
        self.proposal_dist = proposal_dist
        
    def reset_tuning(self):
        """Resets the tuned sampler parameters to their initial values."""
        for attr, initial_value in self._untuned_settings.items():
            setattr(self, attr, initial_value)
        return        

    def astep(self, q0: RaveledVars) -> Tuple[RaveledVars, List[Dict[str, Any]]]:
        """
        Execute one Metropolis-Hastings step.
        
        Parameters
        ----------
        q0 : RaveledVars
            Current state of the latent multinomial vector. See `pymc.blocking` for definition of `RaveledVars`.
            
        Returns
        -------
        tuple
            Tuple containing the next state of the latent multinomial vector as `RaveledVars`, and a dictionary consisting of the MCMC statistics for this iteration. The dictionary has the following keys:
                max_step : Maximum step size of proposal step.
                n_sims : Number of proposals simulated during this iteration.           
                accepted : Number of proposals accepted (excludes transitions that do not change the state) during this iteration out of the `n_sims` proposals.
                moves : Number of times the proposal is different from the current state out of the `n_sims` proposals.                
        """
        point_map_info = q0.point_map_info
        q0 = q0.data
        
        if not self.steps_until_tune and self.tune:
            if self.moves: # tune max step size
                self.max_step = tune_max_step(self.max_step, self.accepted / self.moves)
            
            # tune number of proposals per iteration
            self.n_sims = tune_n_sims(self.n_sims, self.moves / self.tune_interval, self.trg_moves)            
            
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0        
            self.moves = 0
            
        q0 = q0.astype("int64")
            
        # asum, bsum, csum = 0, 0, 0
        
        curr_moves = 0
        curr_accepted = 0
        for _ in range(self.n_sims):
            #t0 = time()           
            
            delta = self.proposal_dist()
            
            # sym
            #q = q0 + delta
            #accept = self.delta_logp(q, q0)
            
            # uniform
            neg_step, pos_step = n_steps(q0, delta, self.max_step)
            tot_step = neg_step + pos_step
            if tot_step == 0:
                q_new = q0
                continue
            curr_moves += 1
            rand = np.random.randint(tot_step)
            if rand < neg_step:
                q = q0 - (rand+1)*delta
            else:
                q = q0 + (tot_step-rand)*delta
            s1, s2 = n_steps(q, delta, self.max_step)
            
            #t1 = time()            
                     
            accept = self.delta_logp(q, q0) + np.log(tot_step / (s1+s2))   
            
            #print(q0, q, dlogp)
            #tmp = list(self.shared.values())[0]
            #print(tmp.get_value())
            
            # filter
            #q, dlogp = self.proposal_dist(q0)
            #accept = self.delta_logp(q, q0) + dlogp
            
            #t2 = time()
            
            q_new, accepted = metrop_select(accept, q, q0)
            curr_accepted += accepted
            q0 = q_new
            
            #t3 = time()
            #asum += t1-t0
            #bsum += t2-t1
            #csum += t3-t2
        
        self.accepted += curr_accepted
        self.moves += curr_moves
        self.steps_until_tune -= 1

        stats = {
            "accepted": curr_accepted,
            "moves": curr_moves,
            "max_step": self.max_step,
            "n_sims": self.n_sims,
            #"time_a": asum,
            #"time_b": bsum,
            #"time_c": csum,
        }

        return RaveledVars(q_new, point_map_info), [stats]   
    
class MBSymProposal(pm.step_methods.metropolis.Proposal):
    """Simulates proposal directions by uniformly sampling from a Markov basis."""
    def __init__(self, basis):
        """
        Initialises the `MBSymProposal` object.
        
        Parameters
        ----------
        basis : 2D-array
            Markov basis with basis vectors to sample from as rows.
        """
        self.basis = basis
        self.bsize = len(basis)
        
    def __call__(self):
        """Uniformly samples a basis vector from the Markov basis."""
        return self.basis[np.random.randint(self.bsize)]
    
def n_steps(q, step, max_step):
    """
    Finds the largest possible step sizes (in two opposite directions) given the current state and proposal direction.

    Parameters
    ----------
    q : 1D-array
        Current state of latent multinomial vector.
    step : 1D-array
        Proposal direction.
    max_step : int
        Maximum step size.

    Returns
    -------
    tuple
        Largest possible step size in the negative and positive directions.            
    """
    neg_step = pos_step = max_step
    for z, d in zip(q, step):
        if d > 0 and (s := z//d) < neg_step:
            neg_step = s
        if d < 0 and (s := z//(-d)) < pos_step:
            pos_step = s
    return neg_step, pos_step
            
            
    # pos_mask = step > 0
    # neg_mask = step < 0
    # return (np.min(q[pos_mask]//step[pos_mask], initial=max_step),
    #         np.min(q[neg_mask]//(-step[neg_mask]), initial=max_step))

def tune_scale(scale, acc_rate):
    """
    Adapted from pm.step_methods.metropolis.tune
    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:

    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10
    
    Parameters
    ----------
    scale : float
        Current proposal scale.
    acc_rate : float
        Current acceptance rate.

    Returns
    -------
    float
        Adjusted proposal scale.
    """
    if acc_rate < 0.001:
        return scale * 0.1
    elif acc_rate < 0.05:
        return scale * 0.5
    elif acc_rate < 0.2:
        return scale * 0.9
    elif acc_rate > 0.95:
        return scale * 10.0
    elif acc_rate > 0.75:
        return scale * 2.0
    elif acc_rate > 0.5:
        return scale * 1.1
    return scale
    
def tune_max_step(max_step, acc_rate):
    """
    Tunes the maximum step size for the proposal distribution
    according to the acceptance rate over the last tune_interval:

    Rate    Variance adaptation
    ----    -------------------
    <0.02         x 0.1
    <0.1          x 0.5
    <0.4          x 0.8
    >0.6          x 1.2
    >0.8          x 2
    >0.95         x 10

    Parameters
    ----------
    max_step : int
        Current maximum step size.
    acc_rate : float
        Current acceptance rate.

    Returns
    -------
    int
        Adjusted maximum step size, rounded to an integer.
    """
    if acc_rate < 0.02:
        max_step_float = max_step * 0.1
    elif acc_rate < 0.1:
        max_step_float = max_step * 0.5
    elif acc_rate < 0.4:
        max_step_float = max_step * 0.8
    elif acc_rate > 0.95:
        max_step_float = max_step * 10.0
    elif acc_rate > 0.8:
        max_step_float = max_step * 2.0
    elif acc_rate > 0.6:
        max_step_float = max_step * 1.2
    else:
        max_step_float = max_step
    return max(1, int(round(max_step_float)))

def tune_n_sims(n_sims, avg_moves, trg_moves):
    """TODO"""
    factor = 1
    if avg_moves < trg_moves * 0.1:
        factor = 5
    elif avg_moves < trg_moves * 0.2:
        factor = 2
    elif avg_moves < trg_moves * 0.5:
        factor = 1.5
    elif avg_moves > trg_moves * 1.5:
        factor = 0.8            
    return max(1, min(round(factor*n_sims), 500))
    
    



