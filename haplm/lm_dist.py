import numpy as np
import math
import random
import pulp
import sympy

from scipy.special import loggamma
import scipy.linalg

from collections import defaultdict as dd

import os
import shutil
import subprocess as sp
import multiprocessing
import aesara
import aesara.tensor as at
import signal

from haplm.hap_utils import encode_amat


def find_4ti2_prefix():
    if shutil.which('markov') is not None:
        return ''
    if shutil.which('4ti2-markov') is not None:
        return '4ti2-'
    else:
        raise RuntimeError("Cannot find 4ti2 prefix, please replace this function call with the prefix to 4ti2's `markov`")


class LatentMult():
    """Stores data relevant to a latent multinomial observation."""    
    def __init__(self, amat, y, n, basis_fname=None, pulp_solver=None, prefix_4ti2=None,
                 enum_sols=False, find_bounds=True, walk_len=500, num_pts=0, logfacts=None, timeout=None):
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
        enum_sols : bool
            Whether to enumerate all latent count solutions.
        find_bounds : bool
            Whether to find bounds to all latent counts,  defaults to True. Setting to False runs much faster, but the only latent counts that can be uniquely determined are those that are zero, and are associated with an observed count of zero.
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
        
        self.logfacts = loggamma(np.arange(1,n+2)) if logfacts is None else logfacts
        
        # pre-process
        self.prep_amat(enum_sols, find_bounds, walk_len, num_pts, timeout)
        
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

    def enum_sols(self, timeout=None):
        """
        Enumerates all nonnegative integer solutions to Az = y after pre-processing.
        If there are multiple solutions, the count that are not uniquely determined 
        have their solutions stored in `self.sols_var`. Otherwise, it is set to None.
            
        Returns
        -------
        None
        """
        amat = self.amat_var.astype(int)
        if amat.shape[1] == 0:
            self.sols_var = None
            return
            
        amat = np.vstack([np.ones(amat.shape[1], int), amat])
        y = np.array([self.n_var] + list(self.y_var))

        if timeout is not None:
            p = multiprocessing.Pool(processes=1)
            res = p.starmap_async(zsolve, ((amat, y, self.mins, self.maxs),))
            try:
                self.sols_var = res.get(timeout=timeout)[0]
            except multiprocessing.TimeoutError:
                self.sols_var = None
        else:
            self.sols_var = zsolve(amat, y, self.mins, self.maxs)

        

    def enum_sols_4ti2(self, fn, timeout=None):
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
        if not self.idx_var: # all counts can be uniquely determined
            return at.dot(self.z_fix, at.log(p))
        
        if not self.amat_var.size: # no info on counts that cannot be uniquely determined
            return at.dot(self.z_fix, at.log(p[self.idx_fix])) + self.n_var*at.log(at.sum(p[self.idx_var]))
            
        return (at.dot(self.z_fix, at.log(p[self.idx_fix]))
                + at.logsumexp((self.sols_var*at.log(p[self.idx_var])-self.logfacts[self.sols_var]).sum(axis=-1)))
    
    def loglike_mn(self, p, mn_stab=1e-9):            
        if not self.idx_var: # all counts can be uniquely determined
            return at.dot(self.z_fix, at.log(p))
        
        qsum = at.sum(p[self.idx_var])
        
        if not self.amat_var.size: # no info on counts that cannot be uniquely determined
            return at.dot(self.z_fix, at.log(p[self.idx_fix])) + self.n_var*at.log(qsum)
        
        q = p[self.idx_var]/qsum

        delta = self.y_var - self.amat_var @ (self.n_var*q) # observed - mean
        aq = at.dot(self.amat_var, q)
        chol = at.slinalg.cholesky(self.n_var*(at.dot(self.amat_var*q, self.amat_var.T)
                                               - at.outer(aq, aq) + np.eye(len(self.y_var))*mn_stab))
        chol_inv_delta = at.slinalg.solve_triangular(chol, delta, lower=True)
        quadform = at.dot(chol_inv_delta, chol_inv_delta)
        logdet = at.sum(at.log(at.diag(chol)))

        return at.dot(self.z_fix, at.log(p[self.idx_fix])) + self.n_var*at.log(qsum) - 0.5*quadform - logdet


    def prep_amat(self, enum_sols, find_bounds=True,
                  walk_len=500, num_pts=0, timeout=None):
        """
        Performs pre-processing to the configuration matrix `amat`, with use of mixed-integer programming (MIP). Pre-processing includes the following steps:
        1) Find a solution to y = Az.
        2) Determine which latent counts can be uniquely determined using MIP.
        3) Remove redundant rows by reducing the configuration matrix with `sympy`.
        
        Parameters
        ----------
        enum_sols : bool
                Whether to enumerate all latent count solutions.
        find_bounds : bool
                Whether to find bounds to all latent counts,  defaults to True. Setting to False runs much faster, but the only latent counts that can be uniquely determined are those that are zero, and are associated with an observed count of zero.
        walk_len : int
            Number of iterations for random walk, defaults to 500.
        num_pts : int
            Number of MCMC initialisation points for latent count vectors, defaults to 0.
            
        Returns
        -------
        None
        """    
        n = self.n
        # add row of ones
        amat = np.vstack([np.ones(len(self.amat[0]), int), self.amat]).astype(int)
        y = np.array([n] + list(self.y))
        
        # column indices corresponding to variable latent counts
        r, h = amat.shape
        inits = np.zeros((num_pts, h), int)
          
        mins = np.zeros(h, int)
        maxs = np.array([min(y[row] if amat[row,col] else y[0]-y[row]
                             for row in range(r)) for col in range(h)])
        nzs = np.where(maxs > 0)[0]
        hnz = len(nzs)
        pmask = maxs > 0 # pmask is False for entries forced to 0

        if num_pts > 0:
            assert walk_len > num_pts
            
        if walk_len > 0 and hnz: # find some latent counts that do not have a unique value
            basis = int_basis_4ti2(amat[:,pmask], self.basis_fname, self.prefix_4ti2, False)
            bsize = len(basis)

            if bsize == 0:
                walk_len = 0
            inits = []
            
            # find starting point for random walk
            prob = pulp.LpProblem("test", pulp.LpMinimize)
            z = pulp.LpVariable.dicts("z", nzs, lowBound=0, cat='Integer')
            for j in range(r):
                prob += (pulp.lpSum([amat[j,k]*z[k] for k in nzs]) == y[j])
            prob.solve(self.pulp_solver)  
            assert prob.status == 1
            sol = np.zeros(h, int)
            sol[pmask] = subsol = np.array([round(z[k].varValue) for k in nzs])

            if walk_len and num_pts > 0:
                save = walk_len - 1 - (num_pts-1) * (walk_len // num_pts)
            else:
                save = walk_len
                
            if bsize > 1:
                geo_denom = math.log(1 - max(0.2, 1/math.sqrt(bsize)))
            for i in range(walk_len):
                step = np.zeros(hnz, int)            
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
                lb = np.min(subsol[pos_mask]//step[pos_mask], initial=y[0])
                ub = np.min(subsol[neg_mask]//(-step[neg_mask]), initial=y[0])
                subsol = subsol + step*np.random.randint(-lb, ub+1)
                sol[pmask] = subsol
                assert np.all(subsol >= 0)
                
                if i == save:
                    inits.append(sol)
                    save += walk_len // num_pts
                mins = np.minimum(mins, sol)
                maxs = np.maximum(maxs, sol)
                
            if inits:
                inits = np.array(inits)
            else:
                inits = np.zeros((num_pts, h))
            assert len(inits) == num_pts
            
        if find_bounds and hnz:
            # check which latent counts cannot be uniquely determined
            pmask = mins != maxs
            nzs = set(nzs)

            for j, mask in enumerate(pmask):
                if not enum_sols and (mask or j not in nzs):
                    continue
                mins[j] = optim(amat, y, j, pulp.LpMinimize, self.pulp_solver)
                maxs[j] = optim(amat, y, j, pulp.LpMaximize, self.pulp_solver)
            pmask = mins != maxs

        fix_js = [j for j, mask in enumerate(pmask) if not mask]
        fix_vals = [mins[j] for j, mask in enumerate(pmask) if not mask]
        var_js = [j for j, mask in enumerate(pmask) if mask]
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

        if amat.size and amat.shape[0] == amat.shape[1]:
            z = np.linalg.solve(amat, y).astype(int)
            fix_js += var_js
            fix_vals += list(z)
            var_js = []
            y = np.array([])
            amat = np.zeros((0,0), int)
            inits = [[] for _ in range(num_pts)]

        assert np.sum(1-amat[:1]) == 0 # first row are ones
        assert amat.shape[1] == len(var_js)
        assert len(amat) == len(y)
        
        self.amat_var = amat[1:].astype(float)
        self.n_var = y[0] if len(y)>0 else 0
        self.y_var = y[1:]
        self.idx_var = var_js
        self.z_fix = np.array(fix_vals, int)
        self.idx_fix = fix_js
        self.inits = inits
        self.mins = mins[pmask]
        self.maxs = maxs[pmask]

        if enum_sols:
            self.enum_sols(timeout)


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
    if prefix_4ti2 is None:
        prefix_4ti2 = find_4ti2_prefix()
    
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
        nrow, ncol = map(int, fp.readline().split())
        if nrow:
            basis = np.array([list(map(int, line.split())) for line in fp])
        else:
            basis = np.zeros((nrow, ncol), int)
        
    assert len(basis.shape) == 2 and (basis.size > 0 or nrow == 0) # check that basis is valid
    if markov:
        mb_dict[code] = basis
    else:
        zb_dict[code] = basis
    os.remove(out_fn)

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
    if prefix_4ti2 is None:
        prefix_4ti2 = find_4ti2_prefix()
    
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


def zsolve(amat, y, mins=None, maxs=None, cap=int(1e6)):
    assert set(list(amat.flatten())).issubset({0,1})
    assert set(amat[0]) == {1}
    
    R, H = amat.shape
    assert H >= R
    _, pivots = sympy.Matrix(amat).rref()
    nonpivots = sorted([h for h in range(H) if h not in pivots], key=lambda h: -amat[:,h].sum())
    
    row_lookup = [[r for r in range(R) if amat[r,h]] + [R+r for r in range(R-1) if not amat[r+1,h]] for h in nonpivots]
    col_lookup = [[h for h in range(H) if amat[r,h]] for r in range(R)] + [[h for h in range(H) if not amat[r,h]] for r in range(1,R)]
    col_lookup_npvt = [[h for h in nonpivots if amat[r,h]] for r in range(R)]
    
    if mins is None:
        mins = [0]*H
    if maxs is None:
        maxs = [min(y[r] if amat[r,h] else y[0]-y[r] for r in range(R)) for h in range(H)]
    
    sols = []
    cand = [0]*H
    Raug = 2*R-1
    rowmins = [sum(mins[h] for h in col_lookup[r]) for r in range(Raug)]
    rowmaxs = [sum(maxs[h] for h in col_lookup[r]) for r in range(Raug)]
    
    inv = scipy.linalg.inv(amat[:,pivots])
    imax = H - R
    
    y = list(y) + [y[0] - val for val in y[1:]]

    vec = np.zeros(R)
    rvec = np.zeros(R)
    inv_diff = inv[:,[r for r in range(R) if amat[r,nonpivots[imax-1]]]].sum(axis=1)
    
    nsols = 0
    def branch(idx):
        nonlocal nsols
        if cap is not None and nsols == cap:
            return

        h = nonpivots[idx]
        a, b = mins[h], maxs[h]
        
        hi = min(b, a+min(y[r]-rowmins[r] for r in row_lookup[idx]))
        lo = max(a, b+max(y[r]-rowmaxs[r] for r in row_lookup[idx]))
        
        if lo > hi:
            return

        if idx+1 == imax:            
            cand[h] = lo
            np.dot(inv, [y[r] - sum(cand[h] for h in col_lookup_npvt[r]) for r in range(R)],
                   out=vec)
            for val in range(lo, hi+1):
                if val != lo:
                    np.subtract(vec, inv_diff, out=vec)
                np.rint(vec, out=rvec)
                if all(rv>=0 and abs(rv-v)<1e-9 for rv,v in zip(rvec,vec)): 
                    cand[h] = val
                    for v, pidx in zip(rvec, pivots):
                        cand[pidx] = v
                    sols.append(cand.copy())
                    nsols += 1
            return
        
        c1 = lo-a
        c2 = b-lo
        for r in row_lookup[idx]:                           
            rowmins[r] += c1
            rowmaxs[r] -= c2
        cand[h] = lo
        branch(idx+1)
        for val in range(lo+1, hi+1):
            for r in row_lookup[idx]:
                rowmins[r] += 1
                rowmaxs[r] += 1
            cand[h] = val
            branch(idx+1)    
        c1 = hi-a
        c2 = b-hi
        for r in row_lookup[idx]:
            rowmins[r] -= c1
            rowmaxs[r] += c2

    branch(0)    

    if nsols == cap:
        return None

    return np.array(sols).astype(int)