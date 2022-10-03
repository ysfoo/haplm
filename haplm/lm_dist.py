import numpy as np
import math
import random
import pulp
import sympy

from scipy.special import loggamma

from collections import defaultdict as dd

import os
import subprocess as sp
import aesara
import aesara.tensor as at


class LatentMult():
    """Stores data relevant to a latent multinomial observation."""    
    def __init__(self, amat, y, n, basis_fname, pulp_solver, prefix_4ti2,
                 merge=False, walk_len=500, num_pts=0, logfacts=None):
        """
        Initialises the `LatentMult` object. For pre-processing details see `prep_amat`.
        
        Parameters
        ----------
        amat : 2D-array
            Configuration matrix.
        y : 1D-array
            Vector of observed counts.
        n : int
            Sample size.
        basis_fname : string
            Prefix for files used by 4ti2, must not clash with files used by `zsolve`.
        pulp_solver : pulp.core.LpSolver_CMD
            A solver object that can solve mixed-integer programming problems. See https://coin-or.github.io/pulp/technical/solvers.html for a list of possible solvers.
        prefix_4ti2 : string
            Prefix to 4ti2's commands, either "" or "4ti2-" depending on installation.
        merge : bool
            Whether identical columns of `amat` should be merged, defaults to False.
        walk_len : int
            Number of iterations for random walk, defaults to 1000.
        num_pts : int
            Number of MCMC initialisation points for latent count vectors, defaults to 0.
        logfacts : 1D-array
            Array of log factorials of numbers from 0 to n. Will be computed during initialisation if None (default).
        """
        self.amat = np.array(amat)
        self.y = np.array(y)
        self.n = n
        self.basis_fname = basis_fname
        self.pulp_solver = pulp_solver
        self.prefix_4ti2 = prefix_4ti2
        self.merge = merge
        
        self.logfacts = loggamma(np.arange(1,n+2)) if logfacts is None else logfacts
        
        # pre-process
        res = prep_amat(self.amat, self.y, n, basis_fname, pulp_solver, prefix_4ti2, merge, walk_len, num_pts)
        self.amat_var, self.n_var, self.y_var, self.idx_var, self.z_fix, self.idx_fix, self.inits = res  
        
        # constants for logp
        self.logbinom_fix = self.logfacts[self.n] - self.logfacts[list(self.z_fix)].sum()
        self.logbinom_fix_nvar = self.logbinom_fix - self.logfacts[self.n_var]
        
    def compute_basis(self, markov=False):
        """Computes integer basis of pre-processed configuration matrix.
        
        Parameters
        ----------
        markov : bool
            Whether to compute a Markov basis.
        """
        self.markov = markov
        amat = self.amat_var.astype(int)
        if amat.shape[1] == 0:
            return
        self.basis = int_basis_4ti2(np.vstack([np.ones(amat.shape[1], int), amat]), self.basis_fname,
                                    self.prefix_4ti2, markov) if amat.shape[1] else None
    
    def enum_sols(self, fn, timeout=None):
        """
        Enumerates all nonnegative integer solutions to Az = y after pre-processing.
        If there are multiple solutions, the count that are not uniquely determined 
        have their solutions stored in `self.sols_var`. Otherwise, it is set to None.
        
        Parameters
        ----------
        fn : string
            Prefix for files used by 4ti2, must not clash with files used by `markov` or `zbasis`.
        timeout : int
            Maximum time (seconds) allowed for `zsolve` to run. Defaults to `None` which imposes no runtime restriction.
            
        Returns
        -------
        None
        """
        amat = self.amat_var.astype(int)
        self.sols_var = zsolve_4ti2(np.vstack([np.ones(amat.shape[1], int), amat]),
                                    np.array([self.n_var] + list(self.y_var)),
                                    fn, self.prefix_4ti2, timeout) if amat.shape[1] else None
        
    def loglike_exact(self, p):
        """
        TODO: Aesara instead
        Computes log p(y|p) exactly using marginalisation.
        
        Parameters
        ----------
        p : 1D-array
            Multinomial probabilities.
        
        Returns
        -------
        Aesara ...
            Log-likelihood p(y|p).
        """
        if self.merge == True:
            raise NotImplementedError
            
        if not self.idx_var: # all counts can be uniquely determined
            return at.dot(self.z_fix, at.log(p))
        
        if not self.amat_var.size: # no info on counts that cannot be uniquely determined
            return at.dot(self.z_fix, at.log(p[self.idx_fix])) + self.n_var*at.log(at.sum(p[self.idx_var]))
            
        return (at.dot(self.z_fix, at.log(p[self.idx_fix]))
                + at.logsumexp((self.sols_var*at.log(p[self.idx_var])-self.logfacts[self.sols_var]).sum(axis=-1)))
    
    def loglike_mn(self, p):
        if self.merge == True:
            raise NotImplementedError
            
        if not self.idx_var: # all counts can be uniquely determined
            return at.dot(self.z_fix, at.log(p))
        
        qsum = at.sum(p[self.idx_var])
        
        if not self.amat_var.size: # no info on counts that cannot be uniquely determined
            return at.dot(self.z_fix, at.log(p[self.idx_fix])) + self.n_var*at.log(qsum)
        
        q = p[self.idx_var]/qsum

        delta = self.y_var - self.amat_var @ (self.n_var*q) # observed - mean
        aq = at.dot(self.amat_var, q)
        chol = at.slinalg.cholesky(self.n_var*(at.dot(self.amat_var*q, self.amat_var.T)
                                               - at.outer(aq, aq)))
        chol_inv_delta = at.slinalg.solve_lower_triangular(chol, delta)
        quadform = at.dot(chol_inv_delta, chol_inv_delta)
        logdet = at.sum(at.log(at.diag(chol)))

        return at.dot(self.z_fix, at.log(p[self.idx_fix])) + self.n_var*at.log(qsum) - 0.5*(logdet+quadform)  


def prep_amat(amat, y, n, basis_fname, pulp_solver,
              prefix_4ti2, merge=False,
              walk_len=500, num_pts=0):
    """
    Performs pre-processing to the configuration matrix `amat`, with heavy use of mixed-integer programming (MIP). Pre-processing includes the following steps:
    1) Merging identical columns of `amat`. This step can reduce the size of the feasible set, though it is optional.
    2) Perform a random walk on the feasible set, noting any latent counts that can take more than one value. This reduces the number of MIP problems to solve in step 4.
    3) Record some latent counts in step 2 to use as MCMC initial points.
    4) Determine which latent counts can be uniquely determined using MIP.
    5) Remove redundant rows by reducing the configuration matrix with `sympy`.
    
    Parameters
    ----------
    amat : 2D-array
        Configuration matrix.
    y : 1D-array
        Vector of observed counts.
    n : int
        Sample size.
    basis_fname : string
        Prefix for files used by 4ti2, must not clash with files used by `zsolve`.
    pulp_solver : pulp.core.LpSolver_CMD
        A solver object that can solve mixed-integer programming problems. See https://coin-or.github.io/pulp/technical/solvers.html for a list of possible solvers.
    prefix_4ti2 : string
        Prefix to 4ti2's commands, either "" or "4ti2-" depending on installation.
    merge : bool
        Whether identical columns of `amat` should be merged, defaults to False.
    walk_len : int
        Number of iterations for random walk, defaults to 1000.
    num_pts : int
        Number of MCMC initialisation points for latent count vectors, defaults to 0.
        
    Returns
    -------
    tuple
        1) Configuration matrix after removing columns corresponding to uniquely determined latent counts and redundant rows.
        2) Number of samples excluding samples with latent haplotypes whose counts can be uniquely determined.
        3) Array of observed latent counts after removing columns corresponding to uniquely determined latent counts and redundant entries.
        4) List containing latent haplotypes corresponding to the columns of the reduced configuration matrix. The list contains list of integers if `merge` is True, otherwise contains integers.
        5) Array of latent counts that can be uniquely determined.
        6) List containing latent haplotypes whose counts can be uniquely determined. The list contains list of integers if `merge` is True, otherwise contains integers.
        7) Array of MCMC initialisation points for latent counts that cannot be uniquely determined.
    """    
    # add row of ones
    amat = np.vstack([np.ones(len(amat[0]), int), amat])
    y = np.array([n] + list(y))
    
    # column indices corresponding to variable latent counts
    var_js = [[j] for j in range(amat.shape[1])] if merge else list(range(amat.shape[1]))
    
    if merge: # merge identical columns
        r, h = amat.shape
        c = dd(list)
        for j, col in enumerate(amat.T):
            c[tuple(col)] += var_js[j]
        if len(c) < h:
            atmat = []
            var_js = []
            for col, jlist in c.items():
                var_js.append(jlist)
                atmat.append(list(col))
            amat = np.array(atmat).T 
            
    r, h = amat.shape
    
    if walk_len > 0: # find some latent counts that do not have a unique value
        basis = int_basis_4ti2(amat, basis_fname, prefix_4ti2, False)
        bsize = len(basis)
        assert bsize > 0
        
        # find starting point for random walk
        prob = pulp.LpProblem("test", pulp.LpMinimize)
        z = pulp.LpVariable.dicts("z", range(h), lowBound=0, cat='Integer')
        for j in range(r):
            prob += (pulp.lpSum([amat[j,k]*z[k] for k in range(h)]) == y[j])
        prob.solve(pulp_solver)  
        assert prob.status == 1
        sol = np.array([round(z[k].varValue) for k in range(h)])
        
        mins = sol.copy()
        maxs = sol.copy()
        
        inits = []
        if num_pts > 0:
            save = walk_len - 1 - (num_pts-1) * (walk_len // num_pts)
        else:
            save = walk_len
            
        if bsize > 1:
            geo_denom = math.log(1 - max(0.2, 1/math.sqrt(bsize)))
        for i in range(walk_len):
            step = np.zeros(h, int)            
            sgn_dict = {}
            repeats = 1 if bsize == 1 else int(math.log(random.random())/geo_denom) + 1
            for _ in range(repeats):
                bidx = int(random.random()*bsize)
                sgn = sgn_dict.get(bidx)
                if sgn is None:
                    sgn = sgn_dict[bidx] = random.getrandbits(1)
                if sgn:
                    step += basis[bidx]
                else:
                    step -= basis[bidx]
            
            pos_mask = step > 0
            neg_mask = step < 0
            lb = np.min(sol[pos_mask]//step[pos_mask], initial=y[0])
            ub = np.min(sol[neg_mask]//(-step[neg_mask]), initial=y[0])
            sol = sol + step*np.random.randint(-lb, ub+1)
            assert np.all(sol >= 0)
            
            if i == save:
                inits.append(sol)
                save += walk_len // num_pts
            mins = np.minimum(mins, sol)
            maxs = np.maximum(maxs, sol)
            
        assert len(inits) == num_pts
        if inits:
            inits = np.array(inits)
        else:
            inits = np.zeros((0, h))
        
        # check which latent counts cannot be uniquely determined
        pmask = mins != maxs

        for j, mask in enumerate(pmask):
            if mask:
                continue
            mins[j] = optim(amat, y, j, pulp.LpMinimize, pulp_solver)
            maxs[j] = optim(amat, y, j, pulp.LpMaximize, pulp_solver)
        pmask = mins != maxs
        
        fix_js = [var_js[j] for j, mask in enumerate(pmask) if not mask]
        fix_vals = [mins[j] for j, mask in enumerate(pmask) if not mask]
        var_js = [var_js[j] for j, mask in enumerate(pmask) if mask]
        y = y - amat[:,~pmask]@mins[~pmask] # remove contribution from fix latent counts
        amat = amat[:,pmask]
        inits = inits[:,pmask]
        
        # see if any rows are redundant
        _, inds = sympy.Matrix(amat).T.rref()
        ind_set = set(inds)
        assert 0 in ind_set or sum(pmask) == 0
        row_mask = np.array([j in ind_set for j in range(r)])
        y = y[row_mask]
        amat = amat[row_mask]
        
        assert np.sum(1-amat[:1]) == 0 # first row are ones
        assert amat.shape[1] == len(var_js)
        assert len(amat) == len(y)
    
    return amat[1:].astype(float), y[0] if len(y)>0 else 0, y[1:], var_js, np.array(fix_vals), fix_js, inits


mb_dict = {} # cache for calls to markov
zb_dict = {} # cache for calls to zbasis

def int_basis_4ti2(amat, fn, prefix_4ti2, markov=False):
    """
    Calls `4ti2-markov` or `4ti2-zbasis` to compute a integer basis.
    
    Parameters
    ----------
    amat : 2D-array
        Matrix to find Markov basis for.
    fn : string
        Prefix for files used by 4ti2, must not clash with files used by `zsolve`.
    markov : bool
        Whether to compute a Markov basis, defaults to False.
    prefix_4ti2 : string
        Prefix to 4ti2's commands, either "" or "4ti2-" depending on installation.

    Returns
    -------
    2D-array
        An integer basis of `amat`.
    """
    
    code = encode_amat(amat)
    if markov and code in mb_dict:
        return mb_dict[code]
    if not markov and code in zb_dict:
        return zb_dict[code]
    
    r, h = amat.shape
    
    with open(f'{fn}.mat', 'w') as fp:        
        fp.write(f'{r} {h}\n')
        for row in amat:
            fp.write(f"{' '.join([str(int(x)) for x in row])}\n")
    
    sp.run([prefix_4ti2+('markov' if markov else 'zbasis'), '-q', fn])
    
    out_fn = fn + ('.mar' if markov else '.lat')    
    with open(out_fn) as fp:
        _, _ = map(int, fp.readline().split())
        basis = np.array([list(map(int, line.split())) for line in fp])
    os.remove(out_fn)
        
    assert len(basis.shape) == 2 and basis.size > 0 # check that basis is valid
    if markov:
        mb_dict[code] = basis
    else:
        zb_dict[code] = basis
    
    return basis


def zsolve_4ti2(amat, y, fn, prefix_4ti2, timeout=None):
    """
    Solves the linear system amat*z = y for nonnegative integer z.
    
    Parameters
    ----------
    amat : 2D-array
        Configuration matrix.
    y : 1D-array
        Vector of observed counts.
    fn : string
        Prefix for files used by 4ti2, must not clash with files used by `markov` or `zbasis`.
    prefix_4ti2 : string
        Prefix to 4ti2's commands, either "" or "4ti2-" depending on installation.
    timeout : int
        Maximum time (seconds) allowed for `zsolve` to run. Defaults to `None` which imposes no runtime restriction.
        
    Returns
    -------
    2D-array, or None
        Returns all nonnegative integer solutions to amat*z = y, or `None` if `zsolve` does not finish in time.
    """
    
    r, h = amat.shape
    with open(f'{fn}.mat', 'w') as fp:        
        fp.write(f'{r} {h}\n')
        for row in amat:
            fp.write(f"{' '.join([str(int(x)) for x in row])}\n")
    
    with open(f'{fn}.rhs', 'w') as fp:
        fp.write(f'1 {r}\n')
        fp.write(f"{' '.join([str(int(x)) for x in y])}\n")
        
    with open(f'{fn}.sign', 'w') as fp:
        fp.write(f'1 {h}\n')
        fp.write(f"{' '.join(['1']*h)}\n")   
     
    if os.path.isfile(f'{fn}.zinhom'):
        sp.run(['rm', f'{fn}.zinhom'])
    if timeout is None:
        sp.run([prefix_4ti2+'zsolve', '-q', fn])
    else:
        sp.run(['timeout', f'{timeout}s', prefix_4ti2+'zsolve', '-q', fn])
    
    if not os.path.isfile(f'{fn}.zinhom'):
        return None
    
    with open(f'{fn}.zinhom') as fp:
        s, _ = map(int, fp.readline().split())
        sols = np.array([list(map(int, line.split())) for line in fp])
    
    return sols


def optim(amat, y, idx, sense, pulp_solver):
    """
    Minimise or maximise z[idx] over all nonnegative integer solutions z to amat*z = y.
    
    Parameters
    ----------
    amat : 2D-array
        Configuration matrix.
    y : 1D-array
        Vector of observed counts.
    idx : int
        Index of latent count to optimise.
    sense : pulp.LpMinimize or pulp.Maxmize
        Whether to minimise or maximise z[idx].
    pulp_solver : pulp.core.LpSolver_CMD
        A solver object that can solve mixed-integer programming problems. See https://coin-or.github.io/pulp/technical/solvers.html for a list of possible solvers.
    
    Returns
    -------
    int
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


def encode_amat(amat):
    """
    Encodes a binary matrix into a tuple of integers.
    
    Parameters
    ----------
    amat : 2D-array
        Matrix to encode.
        
    Returns
    -------
    tuple
        A tuple consisting of the number of rows of `amat`, and each column of `amat` interpreted as a binary number.
    """
    r, _ = amat.shape
    return tuple([r] + list(np.dot(1<<np.arange(r)[::-1], amat)))


def decode_amat(code):
    """
    Decodes a tuple as the result of a binary matrix encoded by `encode_amat`.
    
    Parameters
    ----------
    code : tuple
        Output of `encode_amat`.
        
    Returns
    -------
    2D-array
        Decoded binary matrix.
    """
    return np.transpose([np.fromstring(' '.join(np.binary_repr(num).zfill(code[0])), sep=' ', dtype=int)
                         for num in code[1:]]).astype(aesara.config.floatX)

def mat_by_marker(G):
    """
    Constructs the configuration matrix for the case where observed counts are allele counts for each marker.

    Parameters
    ----------
    G : int
        Number of markers.

    Returns
    -------
    2D-array
        Configuration matrix for latent multinomial vector.
    """
    return np.array([[(h >> i) & 1 for h in range(2**G)] for i in range(G)], int)

import shutil
def find_4ti2_prefix():
    if shutil.which('markov') is not None:
        return ''
    if shutil.which('4ti2-markov') is not None:
        return '4ti2-'
    else:
        raise RuntimeError("Cannot find 4ti2 prefix, please replace this function call with the prefix to 4ti2's `markov`")