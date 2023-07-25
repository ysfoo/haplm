"""
Functions related to pre-processing latent multinomial distributions, along with
exact and approximate likelihood functions for latent multinomial distributions.
"""

import numpy as np
import math
# import random
import pulp
import sympy

from pymc.distributions.dist_math import logpow
# from scipy.special import loggamma
import scipy.linalg

from collections import defaultdict as dd

import os
import warnings
import shutil
import subprocess as sp
import multiprocessing
import pytensor
import pytensor.tensor as pt
import jax.numpy as jnp
import jax.scipy as jsp


def find_4ti2_prefix():
    """Detect prefix to 4ti2 commands, which varies with installation method."""
    if shutil.which('markov') is not None:
        return ''
    if shutil.which('4ti2-markov') is not None:
        return '4ti2-'
    else:
        raise RuntimeError("Cannot find 4ti2 prefix, please specify the "
                           "prefix to 4ti2's commands, which may be empty")



def encode_amat(amat):
    """
    Encodes a binary matrix into a tuple of integers.
    
    Parameters
    ----------
    amat : 2D-array[0,1]
        Matrix to encode.
        
    Returns
    -------
    tuple[int]
        A tuple consisting of the number of rows of `amat`, and each column of 
        `amat` interpreted as a binary number.
    """
    r, _ = amat.shape
    return tuple([r] + list(np.dot(1<<np.arange(r)[::-1], amat)))


def decode_amat(code):
    """
    Decodes a tuple as the result of a binary matrix encoded by `encode_amat`.
    
    Parameters
    ----------
    code : tuple[int]
        Output of `encode_amat`.
        
    Returns
    -------
    2D-array[0,1]
        Decoded binary matrix.
    """
    return np.transpose([np.fromstring(' '.join(np.binary_repr(num)
                                                  .zfill(code[0])),
                                                sep=' ', dtype=int)
                         for num in code[1:]]).astype(pytensor.config.floatX)


class LatentMult():
    """
    Stores data relevant to a latent multinomial observation. See `prep_amat` 
    for details about pre-processing details. Run the `.compute_basis()` method
    if latent count sampling methods are to be used.
    
    Parameters
    ----------
    amat : 2D-array[int]
        Configuration matrix.
    y : list[int]
        Vector of observed counts.
    n : int > 0
        Sample size.
    fname_4ti2 : str, optional
        Prefix for temporary files used by 4ti2, must not be used by any 
        concurrent instances of 4ti2.
    pulp_solver : pulp.core.LpSolver_CMD, optional
        Mixed-integer programming solver that is passed to pulp.LpProblem.solve.
        See https://coin-or.github.io/pulp/technical/solvers.html for a list of 
        possible solvers.
    prefix_4ti2 : str, default to calling `find_4ti2_prefix()`
        Prefix to 4ti2's commands depending on installation method, e.g. "" or 
        "4ti2-".
    enum_sols : bool, default False
        Whether to enumerate all latent count solutions. Enumeration is only 
        practical for problems with relatively small configuration matrices.
    find_bounds : bool, default True
        Whether to find bounds to all latent counts using integer programming. 
        Setting to False makes preprocessing faster, but does not guarantee 
        finding all latent counts that have only one solution. This may lead to 
        latent count sampling to proposing invalid directions more often, and 
        may also present more errors for multinormal approximation.
    num_pts : int >= 0, default 0
        Number of MCMC initialisation points to find for latent count vectors.
    walk_len : int >= `num_pts`, default 500
        Number of iterations for random walk when finding initial latent count
        values. The value must be larger than `num_pts`.
    timeout : float, optional
        Maximum time in seconds to spend when enumerate solutions, ignored if 
        `enum_sols` is False. No maximum is imposed if set to None.
    seed : int, optional
        Random seed for initial values to latent count sampling.
    """    
    def __init__(self, amat, y, n, fname_4ti2, pulp_solver=None, 
                 prefix_4ti2=find_4ti2_prefix(), enum_sols=False, 
                 find_bounds=True, num_pts=0, walk_len=500, timeout=None, 
                 seed=None):
        self.amat = np.array(amat)
        self.y = np.array(y)
        self.n = n
        self.fname_4ti2 = fname_4ti2
        self.pulp_solver = pulp_solver
        self.prefix_4ti2 = prefix_4ti2
        self.enum_sols = enum_sols
        self.find_bounds = find_bounds
        self.num_pts = num_pts
        self.walk_len = walk_len
        self.timeout = timeout
        self.seed = seed
        
        # self.logfacts = loggamma(np.arange(1,n+2)) if logfacts is None else logfacts
        
        # pre-processing
        self.prep_amat()
        
        # constants for logp
        # self.logbinom_fix = self.logfacts[self.n] - self.logfacts[list(self.z_fix)].sum()
        # self.logbinom_fix_nvar = self.logbinom_fix - self.logfacts[self.n_var]
        

    def compute_basis(self, markov=False):
        """
        Computes integer basis of pre-processed configuration matrix. Sets 
        `self.basis` to the result if there are latent counts whose values are
        not uniquely determined, otherwise sets `self.basis` to None.
        
        Parameters
        ----------
        markov : bool, default False
            Whether to compute a Markov basis.

        Returns
        -------
        None
        """
        self.markov = markov
        amat = self.amat_var.astype(int)
        self.basis = (
            int_basis_4ti2(np.vstack([np.ones(amat.shape[1], int), amat]), 
                           self.fname_4ti2, self.prefix_4ti2, markov) 
            if amat.shape[1] 
            else None
        )


    def enum_sols_fn(self):
        """
        Enumerates all nonnegative integer solutions to Az = y after 
        pre-processing. If there are multiple solutions, the counts that are not 
        uniquely determined have their solutions stored in `self.sols_var`. 
        Otherwise, `self.sols_var` is set to None. If the solver takes longer
        than the specified timeout duration, `self.sols_var` is set to None.
        If `self.sols_var` is not None, `self.sols` is set to `self.sols_var`
        augmented with the uniquely determined values. Otherwise, `self.sols`
        is set to None.
            
        Returns
        -------
        None
        """
        amat = self.amat_var.astype(int)
        # all counts are uniquely determined
        if amat.shape[1] == 0:
            self.sols = self.sols_var = None
            return
        
        # include sum constraint
        amat = np.vstack([np.ones(amat.shape[1], int), amat])
        y = np.array([self.n_var] + list(self.y_var))

        if self.timeout is not None:
            p = multiprocessing.Pool(processes=1)
            res = p.starmap_async(zsolve, ((amat, y, self.mins, self.maxs),))
            try:
                self.sols_var = res.get(timeout=self.timeout)[0]
            except multiprocessing.TimeoutError:
                warnings.warn('Warning: latent count solver timed out')
                self.sols_var = None
        else:
            self.sols_var = zsolve(amat, y, self.mins, self.maxs)

        if self.sols_var is None:
            self.sols = None
        else:
            self.sols = np.zeros((len(self.sols_var), self.amat.shape[1]), int)
            self.sols[:,self.idx_fix] = self.z_fix
            self.sols[:,self.idx_var] = self.sols_var


    def enum_sols_4ti2_fn(self):
        """
        Currently not used by `LatentMult`.
        Enumerates all nonnegative integer solutions to Az = y after 
        pre-processing. If there are multiple solutions, the count that are not 
        uniquely determined have their solutions stored in `self.sols_var`. 
        Otherwise, `self.sols_var` is set to None.
            
        Returns
        -------
        None
        """
        amat = self.amat_var.astype(int)
        self.sols_var = (
            zsolve_4ti2(np.vstack([np.ones(amat.shape[1], int), amat]),
                        np.array([self.n_var] + list(self.y_var)),
                        self.fname_4ti2, self.prefix_4ti2, self.timeout) 
            if amat.shape[1] 
            else None
        )

        
    def loglike_exact(self, p, logfacts):
        """
        Computes unnormalised log-likelihood log p(y|p) exactly using 
        marginalisation based on `pytensor` functions.
        
        Parameters
        ----------
        p : pytensor.tensor.var.TensorVariable for 1D-array
            Multinomial probabilities. Entries are nonnegative and sum to 1.
        logfacts : 1D numpy.ndarray
            Array where the k-th entry (k = 0, 1, ...) is log(k!). Length must
            be greater than the pool size `self.n`.
        
        Returns
        -------
        pytensor.tensor.var.TensorVariable for scalar
            Exact log-likelihood p(y|p).
        """           
        # no info on counts that cannot be uniquely determined
        if not self.amat_var.size: 
            return (pt.sum(logpow(p[self.idx_fix], self.z_fix)) 
                    + logpow(pt.sum(p[self.idx_var]), self.n_var))

        return pt.logsumexp((logpow(p, self.sols) - 
                             logfacts[self.sols]).sum(axis=-1))


    def loglike_exact_jax(self, p, logfacts):
        """
        Computes unnormalised log-likelihood log p(y|p) exactly using 
        marginalisation based on `jax` functions.
        
        Parameters
        ----------
        p : jaxlib.xla_extension.ArrayImpl for 1D-array
            Multinomial probabilities. Entries are nonnegative and sum to 1.
        logfacts : jaxlib.xla_extension.ArrayImpl for 1D-array
            Array where the k-th entry (k = 0, 1, ...) is log(k!). Length must
            be greater than the pool size `self.n`.
        
        Returns
        -------
        jaxlib.xla_extension.ArrayImpl for scalar
            Exact log-likelihood p(y|p).
        """            
        # no info on counts that cannot be uniquely determined
        if not self.amat_var.size: 
            return (jnp.sum(jsp.special.xlogy(self.z_fix, p[self.idx_fix])) + 
                    jsp.special.xlogy(self.n_var, jnp.sum(p[self.idx_var])))

        return jsp.special.logsumexp((jsp.special.xlogy(self.sols, p) -
                                      logfacts[self.sols]).sum(axis=-1))
    

    def loglike_mn(self, p, mn_stab=1e-9):    
        """
        Approximates unnormalised log-likelihood log p(y|p) with a multinormal
        approximation based on `pytensor` functions.
        
        Parameters
        ----------
        p : pytensor.tensor.var.TensorVariable for 1D-array
            Multinomial probabilities. Entries are nonnegative and sum to 1.
        mn_stab : float >= 0, default 1e-9
            Stabilising constant added to the covariance diagonal for the
            covariance to be non-singular.
        
        Returns
        -------
        pytensor.tensor.var.TensorVariable for scalar
            Approximate log-likelihood p(y|p).
        """                   
        # all counts can be uniquely determined
        if not self.idx_var.size: 
            return pt.sum(logpow(self.z_fix, p))
        
        # probability for categories whose counts are not uniquely determined
        qsum = pt.sum(p[self.idx_var])
        
        # no info on counts that cannot be uniquely determined
        if not self.amat_var.size: 
            return (pt.sum(logpow(self.z_fix, p[self.idx_fix])) + 
                    logpow(self.n_var, qsum))
        
        # normalise probabilities
        q = p[self.idx_var]/qsum

        delta = self.y_var - self.amat_var @ (self.n_var*q) # observed - mean
        aq = pt.dot(self.amat_var, q)
        chol = pt.slinalg.cholesky(self.n_var*
                                   (pt.dot(self.amat_var*q, self.amat_var.T) - 
                                    pt.outer(aq, aq) + 
                                    np.eye(len(self.y_var))*mn_stab))
        chol_inv_delta = pt.slinalg.solve_triangular(chol, delta, lower=True)
        quadform = pt.dot(chol_inv_delta, chol_inv_delta)
        logdet = pt.sum(pt.log(pt.diag(chol)))

        return (pt.sum(logpow(self.z_fix, p[self.idx_fix])) + 
                logpow(self.n_var, qsum) - 0.5*quadform - logdet)


    def loglike_mn_jax(self, p, mn_stab=1e-9): 
        """
        Approximates unnormalised log-likelihood log p(y|p) with a multinormal
        approximation based on `jax` functions.
        
        Parameters
        ----------
        p : jaxlib.xla_extension.ArrayImpl for 1D-array
            Multinomial probabilities. Entries are nonnegative and sum to 1.
        mn_stab : float >= 0, default 1e-9
            Stabilising constant added to the covariance diagonal for the
            covariance to be non-singular.
        
        Returns
        -------
        jaxlib.xla_extension.ArrayImpl for scalar
            Approximate log-likelihood p(y|p).
        """        
        # all counts can be uniquely determined   
        if not self.idx_var.size: 
            return jsp.special.xlogy(self.z_fix, p)
        
        # probability for categories whose counts are not uniquely determined
        qsum = jnp.sum(p[self.idx_var])
        
        # no info on counts that cannot be uniquely determined
        if not self.amat_var.size: 
            return (jnp.sum(jsp.special.xlogy(self.z_fix, p[self.idx_fix])) + 
                    jsp.special.xlogy(self.n_var, qsum))
        
        # normalise multinomial probabilities
        q = p[self.idx_var]/qsum

        aq = jnp.dot(self.amat_var, q)
        delta = self.y_var - self.n_var*aq # observed - mean
        chol = jsp.linalg.cholesky(self.n_var*
                                   (jnp.dot(self.amat_var*q, self.amat_var.T) - 
                                    jnp.outer(aq, aq) + 
                                    jnp.eye(len(self.y_var))*mn_stab),
                                   lower=True)
        chol_inv_delta = jsp.linalg.solve_triangular(chol, delta, lower=True)
        quadform = jnp.dot(chol_inv_delta, chol_inv_delta)
        logdet = jnp.sum(jnp.log(jnp.diag(chol)))

        return (jnp.dot(self.z_fix, jnp.log(p[self.idx_fix])) + 
                self.n_var*jnp.log(qsum) - 0.5*quadform - logdet)


    def prep_amat(self):
        """
        Performs pre-processing to the configuration matrix `self.amat`, with 
        use of mixed-integer programming (MIP). Pre-processing includes the 
        following steps:
            1) Find a nonnegative integer solution to y = Az.
            2) Determine which latent counts can be uniquely determined using 
               MIP.
            3) Remove redundant rows by reducing the configuration matrix with 
               `sympy` for that the configuration matrix is full rank.

        Returns
        -------
        None
        """

        rng = np.random.default_rng(self.seed)

        n = self.n # pool size
        # add row of ones for sum constraint
        amat = np.vstack([np.ones(len(self.amat[0]), int), 
                          self.amat]).astype(int)
        y = np.array([n] + list(self.y)).astype(int)

        assert all(0 <= yval <= n for yval in y), (
            "Observed counts must be between 0 and pool size")
        
        # number of latent counts and number of multinomial categories
        r, h = amat.shape
        
        # bounds on latent counts, initialised with trivial bounds
        mins = np.zeros(h, int)
        maxs = np.array([min(y[row] if amat[row,col] else y[0]-y[row]
                             for row in range(r)) for col in range(h)])
        # indices of categories that could have a positive count
        nzs = np.where(maxs > 0)[0]
        # number of categories that could have a positive count
        hnz = len(nzs)
        # pmask is False for entries forced to 0
        pmask = maxs > 0 

        # if initial points are to be found, walk length must be positive
        if self.num_pts > 0:
            assert self.walk_len > self.num_pts, (
                "Walk length must be positive for initial points to be found")
        
        # find initial values for latent counts that do not have a unique value
        inits = []
        if self.walk_len > 0 and hnz: 
            basis = int_basis_4ti2(amat[:,pmask], 
                                   self.fname_4ti2, self.prefix_4ti2, False)
            bsize = len(basis)

            # only one possible latent count if basis is empty
            if bsize == 0:
                self.walk_len = 0
            
            # find starting point for random walk
            prob = pulp.LpProblem("test", pulp.LpMinimize)
            z = pulp.LpVariable.dicts("z", nzs, lowBound=0, cat='Integer')
            for j in range(r):
                prob += (pulp.lpSum([amat[j,k]*z[k] for k in nzs]) == y[j])
            prob.solve(self.pulp_solver)  
            assert prob.status == 1
            sol = np.zeros(h, int)
            sol[pmask] = subsol = np.array([round(z[k].varValue) for k in nzs])

            if self.walk_len and self.num_pts > 0:
                save = (self.walk_len - 1 - 
                        (self.num_pts-1) * (self.walk_len // self.num_pts))
            else:
                save = self.walk_len
                
            # if bsize > 1:
            #     geo_denom = math.log(1 - max(0.2, 1/math.sqrt(bsize)))
            for i in range(self.walk_len):
                step = rng.choice(basis)
                # step = np.zeros(hnz, int)            
                # sgn_dict = {}
                # repeats = 1 if bsize == 1 else int(math.log(random.random())/geo_denom) + 1
                # for _ in range(repeats):
                #     bidx = int(random.random()*bsize)
                #     sgn = sgn_dict.get(bidx)
                #     if sgn is None:
                #         sgn = sgn_dict[bidx] = random.getrandbits(1)
                #     if sgn:
                #         step += basis[bidx]
                #     else:
                #         step -= basis[bidx]
                
                # take random step size
                pos_mask = step > 0
                neg_mask = step < 0
                lb = np.min(subsol[pos_mask]//step[pos_mask], initial=y[0])
                ub = np.min(subsol[neg_mask]//(-step[neg_mask]), initial=y[0])
                subsol = subsol + step*rng.integers(-lb, ub+1)
                sol[pmask] = subsol
                assert np.all(subsol >= 0), ("There is an issue with the "
                                             "configuration matrix")
                
                if i == save:
                    inits.append(sol.copy())
                    save += self.walk_len // self.num_pts

                # update bounds
                mins = np.minimum(mins, sol)
                maxs = np.maximum(maxs, sol)
        
        # convert `inits` to numpy.darray
        inits = np.array(inits)
        
        # find exact bounds
        if self.find_bounds and hnz:
            # check which latent counts may not be uniquely determined
            pmask = mins != maxs
            nzs = set(nzs)

            # mask is true if category j may not be uniquely determined
            for j, mask in enumerate(pmask):
                if mask or j not in nzs:
                    continue
                mins[j] = optim(amat, y, j, pulp.LpMinimize, self.pulp_solver)
                maxs[j] = optim(amat, y, j, pulp.LpMaximize, self.pulp_solver)
            pmask = mins != maxs

        # indices for categories whose count are uniquely determined
        fix_js = [j for j, mask in enumerate(pmask) if not mask]
        # uniquely determined haplotype counts corresponding to `fix_js`
        fix_vals = [mins[j] for j in fix_js]

        # indices for categories whose counts may not be uniquely determined
        var_js = [j for j, mask in enumerate(pmask) if mask]

        # remove contribution from uniquely determined latent counts
        y = y - amat[:,~pmask]@mins[~pmask] 
        amat = amat[:,pmask]
        # inits = inits[:,pmask]

        # see if any rows of the configuration matrix are redundant
        _, inds = sympy.Matrix(amat).T.rref()
        ind_set = set(inds)

        # first row must be kept, or all counts must be uniquely determined
        assert 0 in ind_set or not any(pmask), (
            "Row reduction failed, please remove redundant observed counts")

        # filter for rows to be kept
        row_mask = np.array(sorted(ind_set))
        y = y[row_mask]
        amat = amat[row_mask]

        # if the configuration matrix is square and of full rank, we can
        # uniquely determine all counts
        if amat.size and amat.shape[0] == amat.shape[1]:
            z = np.linalg.solve(amat, y).astype(int)
            fix_js += var_js
            fix_vals += list(z)
            var_js = []
            y = np.array([])
            amat = np.zeros((0,0), int)
        
        # if all counts can be determined, set `inits` all to same value
        if not var_js:
            assert len(fix_js) == h
            init = np.array(fix_vals)[np.argsort(fix_js)]
            inits = np.array([init for _ in range(self.num_pts)])

        assert (np.sum(1-amat[:1]) == 0 and
                amat.shape[1] == len(var_js) and 
                len(amat) == len(y)), (
            "There is an issue with the configuration matrix")
        
        # configuration matrix for counts that are not uniquely determined
        # convert to float for multinormal approximation
        # note that sum constraint is excluded
        self.amat_var = amat[1:].astype(float)

        # total number of samples corresponding to undetermined counts
        self.n_var = y[0] if len(y)>0 else 0
        # observed counts after removing categories with determined counts
        # note that pool size is excluded
        self.y_var = y[1:]

        # indices for categories whose counts may not be uniquely determined        
        self.idx_var = np.array(var_js, int)
        # indices for categories whose counts are uniquely determined       
        self.idx_fix = np.array(fix_js, int)
        # uniquely determined haplotype counts corresponding to `self.idx_fix`
        self.z_fix = np.array(fix_vals, int)
        
        # initial points for latent count sampling (all categories included)
        self.inits = np.array(inits, int)
        # lower bounds for categories corresponding to `self.idx_var`
        self.mins = mins[pmask]
        # upper bounds for categories corresponding to `self.idx_var`
        self.maxs = maxs[pmask]

        # find all solutions and save all counts to `self.sols`, whereas
        # `self.sols_var` stores the counts corresponding to `self.idx_var`
        if self.enum_sols:
            self.enum_sols_fn()
        else:
            self.sols = self.sols_var = None


mb_dict = {} # cache for calls to markov
zb_dict = {} # cache for calls to zbasis

def int_basis_4ti2(amat, fname, prefix_4ti2, markov=False):
    """
    Calls `4ti2-markov` or `4ti2-zbasis` to compute a integer basis.
    
    Parameters
    ----------
    amat : numpy.darray of 2D-array[0,1]
        Matrix to find integer basis for. Sum constraint should be included.
    fname : str
        Prefix for temporary files used by 4ti2, must not be used by any 
        concurrent instances of 4ti2.
    prefix_4ti2 : str
        Prefix to 4ti2's commands depending on installation method, e.g. "" or 
        "4ti2-".
    markov : bool, default False
        Whether to compute a Markov basis.

    Returns
    -------
    numpy.darray for 2D-array[int]
        An integer basis of `amat`.
    """    
    code = encode_amat(amat)
    # retrieve basis if previously computed
    if markov and code in mb_dict:
        return mb_dict[code]
    if not markov and code in zb_dict:
        return zb_dict[code]
    
    r, h = amat.shape
    
    # write input
    with open(f'{fname}.mat', 'w') as fp:        
        fp.write(f'{r} {h}\n')
        for row in amat:
            fp.write(f"{' '.join([str(int(x)) for x in row])}\n")
    
    # run 4ti2 command
    sp.run([prefix_4ti2+('markov' if markov else 'zbasis'), '-q', fname])
    
    # parse output
    out_fname = fname + ('.mar' if markov else '.lat')    
    with open(out_fname) as fp:
        nrow, ncol = map(int, fp.readline().split())
        if nrow:
            basis = np.array([list(map(int, line.split())) for line in fp])
        else:
            basis = np.zeros((nrow, ncol), int)
    
    # check that basis is valid
    assert len(basis.shape) == 2 and (basis.size > 0 or nrow == 0), (
        "Basis returned is invalid")

    # store basis to cache
    if markov:
        mb_dict[code] = basis
    else:
        zb_dict[code] = basis

    # remove output file as it may clash with other 4ti2 functionality
    os.remove(out_fname)

    return basis


def zsolve_4ti2(amat, y, fname, prefix_4ti2, timeout=None):
    """
    Currently not used by `LatentMult`.
    Solves the linear system amat*z = y for nonnegative integer z.
    
    Parameters
    ----------
    amat : numpy.darray of 2D-array[0,1]
        Configuration matrix. Sum constraint should be included.
    y : numpy.darray of 1D-array[int >= 0]
        Vector of observed counts.
    fname : str
        Prefix for temporary files used by 4ti2, must not be used by any 
        concurrent instances of 4ti2.
    prefix_4ti2 : str
        Prefix to 4ti2's commands depending on installation method, e.g. "" or 
        "4ti2-".
    timeout : int > 0, optional
        Maximum time (seconds) allowed for `zsolve` to run. Defaults to `None` 
        which imposes no runtime restriction.
        
    Returns
    -------
    numpy.darray for 2D-array[int >= 0], or None
        Returns all nonnegative integer solutions to amat*z = y, or `None` 
        if `zsolve` does not finish in time.
    """    
    r, h = amat.shape

    # write inputs
    with open(f'{fname}.mat', 'w') as fp:        
        fp.write(f'{r} {h}\n')
        for row in amat:
            fp.write(f"{' '.join([str(int(x)) for x in row])}\n")
    
    with open(f'{fname}.rhs', 'w') as fp:
        fp.write(f'1 {r}\n')
        fp.write(f"{' '.join([str(int(x)) for x in y])}\n")
        
    with open(f'{fname}.sign', 'w') as fp:
        fp.write(f'1 {h}\n')
        fp.write(f"{' '.join(['1']*h)}\n")   
    
    # remove output file if present
    if os.path.isfile(f'{fname}.zinhom'):
        sp.run(['rm', f'{fname}.zinhom'])
    # run 4ti2 command
    if timeout is None:
        sp.run([prefix_4ti2+'zsolve', '-q', fname])
    else:
        sp.run(['timeout', f'{timeout}s', prefix_4ti2+'zsolve', '-q', fname])
    
    # timeout
    if not os.path.isfile(f'{fname}.zinhom'):
        warnings.warn('Warning: latent count solver timed out')
        return None
    
    # parse output
    with open(f'{fname}.zinhom') as fp:
        s, _ = map(int, fp.readline().split())
        sols = np.array([list(map(int, line.split())) for line in fp])
    
    return sols


def optim(amat, y, idx, sense, pulp_solver):
    """
    Minimise or maximise z[idx] over all nonnegative integer solutions z to 
    amat*z = y by calling a solver from `pulp`.
    
    Parameters
    ----------
    amat : numpy.darray of 2D-array[0,1]
        Configuration matrix. Sum constraint should be included.
    y : numpy.darray of 1D-array[int >= 0]
        Vector of observed counts.
    idx : int >= 0
        Index of latent count to optimise.
    sense : pulp.LpMinimize or pulp.LpMaxmize
        Whether to minimise or maximise z[idx].
    pulp_solver : pulp.core.LpSolver_CMD, optional
        Mixed-integer programming solver that is passed to pulp.LpProblem.solve.
        See https://coin-or.github.io/pulp/technical/solvers.html for a list of 
        possible solvers.
    
    Returns
    -------
    int >= 0
        Returns optimum value of z[idx].
    """
    r, h = amat.shape    
    prob = pulp.LpProblem("test", sense)
    z = pulp.LpVariable.dicts("z", range(h), lowBound=0, cat='Integer')
    prob += z[idx]
    for j in range(r):
        prob += (pulp.lpSum([amat[j,k]*z[k] for k in range(h)]) == y[j])
    prob.solve(pulp_solver)
    return pulp.value(prob.objective)


def zsolve(amat, y, mins=None, maxs=None, cap=int(1e7)):
    """
    Finds all nonnegative solutions to amat*z = y using a branch-and-bound
    algorithm. Haplotypes whose counts are uniquely determined should be omitted
    for better efficiency.

    Parameters
    ----------
    amat : numpy.darray of 2D-array[0,1]
        Configuration matrix. Sum constraint should be included.
    y : numpy.darray of 1D-array[int >= 0]
        Vector of observed counts.
    mins : list[int], optional
        Lower bounds for latent counts. Trivial bounds are determined if not 
        provided.
    maxs : list[int], optional
        Upper boudns for latent counts. Trivial bounds are determined if not 
        provided.
    cap : int > 0, default 1e7
        Algorithm is aborted if the number of solutions is at least `cap`.

    Returns
    -------
    numpy.darray for 2d-array[int >= 0], or None
        Returns nonnegative solutions if there are less than `cap` of them,
        otherwise returns None.
    """
    assert set(list(amat.flatten())).issubset({0,1}), (
        "This enumeration algorithm requires the configuration matrix to "
        "contain 0s and 1s only")
    assert set(amat[0]) == {1}, (
        "The first row of the configuration matrix must be a sum constraint")
    
    # number of equations and number of categories
    R, H = amat.shape
    assert H >= R, ("Either configuration matrix is not of full rank, "
                    "or latent counts should be uniquely determined")

    # find R linearly independent columns as pivot columns, `pivots` are indices
    _, pivots = sympy.Matrix(amat).rref()
    # indices of nonpivot columns, ordered by most 1s first
    nonpivots = sorted([h for h in range(H) if h not in pivots], 
                       key=lambda h: -amat[:,h].sum())
    
    # consider an augmented configuration matrix (ACM) which has R-1 rows 
    # appended, which are the last R-1 rows of `amat` with entries flipped

    # list of lists for each nonpivot column, where the h-th list consists of
    # the indices where the h-th nonpivot column of the ACM is 1
    row_lookup = [[r for r in range(R) if amat[r,h]] + 
                  [R+r for r in range(R-1) if not amat[r+1,h]] 
                  for h in nonpivots]
    # list of lists for each row of the ACM, where the r-th list consists of the 
    # indices where the r-th row of the ACM is 1
    col_lookup = ([[h for h in range(H) if amat[r,h]] for r in range(R)] + 
                  [[h for h in range(H) if not amat[r,h]] for r in range(1,R)])
    # list of lists for each row of `amat`, where the r-th list consists of the
    # nonpivot column indices where the r-th row of the ACM is 1
    col_lookup_npvt = [[h for h in nonpivots if amat[r,h]] for r in range(R)]
    
    # find trivial bounds if no bounds provided
    if mins is None:
        mins = [0]*H
    if maxs is None:
        maxs = [min(y[r] if amat[r,h] else y[0]-y[r] for r in range(R)) 
                for h in range(H)]
    
    # store all solutions
    sols = []
    # current candidate solution
    cand = [0]*H
    # number of rows of ACM
    Raug = 2*R-1
    # bounds on possible values of ACM*cand
    acm_mins = [sum(mins[h] for h in col_lookup[r]) for r in range(Raug)]
    acm_maxs = [sum(maxs[h] for h in col_lookup[r]) for r in range(Raug)]
    
    # matrix inverse of pivot columns, which we call the pivot inverse
    inv = scipy.linalg.inv(amat[:,pivots])
    # number of nonpivot columns
    imax = H - R
    
    # y = ACM*z
    y = list(y) + [y[0] - val for val in y[1:]]

    # matrix product of pivot inverse and the last nonpivot column
    inv_diff = inv[:,[r for r in range(R) 
                      if amat[r,nonpivots[imax-1]]]].sum(axis=1)
    
    # number of solutions
    nsols = 0

    # define recursive branching function
    # `idx` is the current index of `nonpivots`
    def branch(idx):
        nonlocal nsols
        # too many solutions
        if cap is not None and nsols >= cap:
            return

        h = nonpivots[idx]
        a, b = mins[h], maxs[h]
        
        # updated bounds on cand[h]
        hi = min(b, a+min(y[r]-acm_mins[r] for r in row_lookup[idx]))
        lo = max(a, b+max(y[r]-acm_maxs[r] for r in row_lookup[idx]))
        
        if lo > hi:
            return

        # only last nonpivot column's latent count to vary
        if idx+1 == imax:            
            cand[h] = lo
            # determine latent counts of the pivot columns, store to `vec`
            vec = np.dot(inv, [y[r] - sum(cand[h] for h in col_lookup_npvt[r]) 
                               for r in range(R)])
            for val in range(lo, hi+1):
                # adjust latent counts of the pivot columns given that the
                # last nonpivot column's latent count is increased by 1
                if val != lo:
                    vec -= inv_diff

                # round `vec` to integer
                rvec = np.rint(vec)

                # if `vec` is close to integer, we have a valid solution
                if all(rv>=0 and abs(rv-v)<1e-9 for rv,v in zip(rvec,vec)): 
                    cand[h] = val
                    for v, pidx in zip(rvec, pivots):
                        cand[pidx] = v
                    sols.append(cand.copy())
                    nsols += 1
            return        

        # check how much the updated bounds improved from `mins` and `maxs`
        c1 = lo-a
        c2 = b-lo

        # set `cand[h]` to its updated lower bound
        cand[h] = lo

        # update ACM bounds given some fixed value of `cand[h]`
        for r in row_lookup[idx]:                           
            acm_mins[r] += c1
            acm_maxs[r] -= c2

        # go to next pivot column
        branch(idx+1)

        # iterate `cand[h]` from `lo+1` to `hi` (inclusive)
        for val in range(lo+1, hi+1):
            cand[h] = val

            # update ACM bounds given the change of `cand[h]`
            for r in row_lookup[idx]:
                acm_mins[r] += 1
                acm_maxs[r] += 1
            
            # go to next pivot column
            branch(idx+1)    

        # before backtracking, undo ACM bounds updates due to `cand[h]` 
        c1 = hi-a
        c2 = b-hi
        for r in row_lookup[idx]:
            acm_mins[r] -= c1
            acm_maxs[r] += c2

    # start branch-and-bound algorithm
    branch(0)    

    if nsols >= cap:
        warnings.warn('Warning: latent count solver aborted as there are too '
                      'many solutions')
        return None

    return np.array(sols).astype(int)