from collections import Counter, defaultdict as dd

import numpy as np
import scipy.stats
from scipy.special import softmax, gammaln, logsumexp
import pulp
import sympy
import math
from time import time
import xarray

from typing import Any, Callable, Dict, List, Tuple
import pymc as pm
from pymc.distributions.dist_math import check_parameters, factln, logpow
from pymc.blocking import RaveledVars
from pymc.math import logdet
from pymc.step_methods.arraystep import metrop_select, ArrayStepShared
from pymc.step_methods.metropolis import delta_logp
import arviz as az
import aesara
from aesara.tensor.random.op import RandomVariable, default_supp_shape_from_params
import aesara.tensor as at
from aesara.graph.basic import vars_between

from atpbar import atpbar, register_reporter, find_reporter, flush
import multiprocessing
import sys


def latent_mult_chain(c, s):
    return latent_mult_mcmc.mcmc(c, s)

def latent_mult_cgibbs_chain(c, s):
    return latent_mult_mcmc_cgibbs.mcmc(c, s)

def latent_mult_mcmc(lm_list, H, n_sample, n_burnin, methods,
                     logprior_func=None,
                     chains=5, cores=1, seeds=None):   
    if logprior_func is None:
        logprior_func = lambda p: 0 # at.as_tensor_variable(0)
    
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
            return at.ones(H)/H

        def logp(value, param):
            return logprior_func(value)

    with pm.Model() as model:
        #p = pm.Dirichlet('p', np.ones(H))
        p = Custom('p', np.ones(H), shape=(H,))
        #ys_mn = pm.Potential('ys_mn', lm_list[0].loglike_exact(p))

        if mn_is:
            ys_mn = pm.Potential('ys_mn', sum(lm_list[i].loglike_mn(p) for i in mn_is))
        if exact_is:
            ys_exact = pm.Potential('ys_exact', sum(lm_list[i].loglike_exact(p) for i in exact_is))
        
        # ys_mn = [pm.DensityDist('y'+str(i), p,
        #                         logp=make_logp(lm_list[i], 'mn_approx'),
        #                         observed=np.zeros(1), dtype='int64')
        #          for i in mn_is]
        # ys_exact = [pm.DensityDist('y'+str(i), p,
        #                         logp=make_logp(lm_list[i], 'exact'),
        #                         observed=np.zeros(1), dtype='int64')
        #          for i in exact_is]
               
    if basis_is:
        raise NotImplementedError
    
    else:
        with model:
            idata = pm.sample(draws=n_sample, tune=n_burnin,
                              step=[pm.NUTS(target_accept=0.9)],
                              chains=chains, cores=cores, random_seed=seeds,
                              compute_convergence_checks=False)

    return idata  


def latent_mult_mcmc_cgibbs(lm_list, H, n_sample, n_burnin, alphas=None, 
                            cycles=5, chains=5, cores=1, seeds=None): 
                            #adapt=False, tune_itv=50, chains=5, cores=1, seeds=None):
    if alphas is None:
        alphas = np.ones(H)
    else:
        alphas = np.array(alphas)

    if seeds is None:
        seeds = [None]*chains
    assert len(seeds) == chains
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
        if not lm.idx_var:
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


class MBasisMetropolis(pm.step_methods.metropolis.Metropolis):
    """PyMC Metropolis-Hastings sampling step where discrete random walk directions are given by a Markov basis."""

    name = "markov basis metropolis"
    default_blocked = False
    generates_stats = True
    stats_dtypes = [{"acc_rate": float,
                     #"time_a": float,
                     #"time_b": float,
                     #"time_c": float,
                    }]

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
    <0.01        x 0.1
    <0.1         x 0.5
    <0.3         x 0.9
    >0.6         x 1.1
    >0.8         x 2
    >0.95        x 10
    
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
    if acc_rate < 0.01:
        return scale * 0.1
    elif acc_rate < 0.1:
        return scale * 0.5
    elif acc_rate < 0.3:
        return scale * 0.9
    elif acc_rate > 0.95:
        return scale * 10.0
    elif acc_rate > 0.8:
        return scale * 2.0
    elif acc_rate > 0.6:
        return scale * 1.1
    return scale
    
def tune_max_step(max_step, acc_rate):
    """
    Tunes the maximum step size for the proposal distribution
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
    max_step : int
        Current maximum step size.
    acc_rate : float
        Current acceptance rate.

    Returns
    -------
    int
        Adjusted maximum step size, rounded to an integer.
    """
    if acc_rate < 0.001:
        max_step_float = max_step * 0.1
    elif acc_rate < 0.05:
        max_step_float = max_step * 0.5
    elif acc_rate < 0.2:
        max_step_float = max_step * 0.8
    elif acc_rate > 0.95:
        max_step_float = max_step * 10.0
    elif acc_rate > 0.75:
        max_step_float = max_step * 2.0
    elif acc_rate > 0.5:
        max_step_float = max_step * 1.2
    else:
        max_step_float = max_step
    return max(1, int(round(max_step_float)))

def tune_n_sims(n_sims, avg_moves, trg_moves):
    """TODO"""           
    return max(1, round(trg_moves/avg_moves*n_sims))
    
    



