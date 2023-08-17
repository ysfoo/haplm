"""
Utility functions related to haplotypes, including partition ligation using
either AEML or MCMC-Approx as a frequency estimation subroutine.
"""

import numpy as np
import pulp
from haplm.hippo_aeml import run_AEML
from haplm.lm_dist import LatentMult
from haplm.lm_inference import latent_mult_mcmc


def mat_by_marker(G):
    """
    Constructs the configuration matrix for the case where observed counts are
    allele counts for each marker.

    Parameters
    ----------
    G : int
        Number of markers.

    Returns
    -------
    2D-array[0,1]
        Configuration matrix for latent multinomial vector.
    """
    return np.array([[(h >> i) & 1 for h in range(2**G)] for i in range(G)], 
                    int)


def str_to_num(hstr):
    """
    Converts binary string to integer representation. The i-th bit from the left
    corresponds to 2^(i-1), where i = 1, 2, ...

    Parameters
    ----------
    hstr : str
        String of binary digits.

    Returns
    -------
    int
        Integer representation of the binary string.
    """
    n_markers = len(hstr)
    return sum(1 << i for i, c in enumerate(hstr) if c == '1')


def num_to_str(hnum, strlen):
    """
    Obtain the integer representation of a binary string. The i-th bit from the 
    left corresponds to 2^(i-1), where i = 1, 2, ...

    Parameters
    ----------
    hnum : str
        Integer representation of the binary string.
    strlen : int
        Length of the binary string.

    Returns
    -------
    str
        Binary string representation of the integer.
    """
    return ''.join(['1' if (hnum >> i) & 1 else '0' for i in range(strlen)])


def PL_aeml(ns, ys, n_markers, hap_fn, aeml_dir, solver, thres, maxhaps=None, 
            trials=5, inithaps_fn=lambda x: []):
    """
    Determine input haplotypes using partition ligation based, with AEML as the 
    frequency estimate subroutine. Includes 
    check to ensure that the feasible set is nonempty.

    Parameters
    ----------
    ns : list[int]
        Number of haplotype samples for each pool.
    ys : list[list[int]]
        List consisting of a list per pool, where the inner list consists of the
        allele counts of each marker.
    n_markers : int > 0
        Number of markers.
    hap_fn : str
        Filename for input haplotype list to AEML.
    aeml_dir : str
        Directory containing the C program for AEML.
    solver : pulp.core.LpSolver_CMD, optional
        Mixed-integer programming solver that is passed to pulp.LpProblem.solve.
        See https://coin-or.github.io/pulp/technical/solvers.html for a list of 
        possible solvers.
    thres : float > 0
        Frequency threshold of haplotypes to be selected.
    maxhaps : int, optional
        Maximum number of concatenated haplotypes. If None, no maximum is
        imposed.
    trials : int > 0, default 5
        Number of trials for expectation maximization.
    inithaps_fn : Callable[[int], iterable[iterable]], default : lambda x: []
        Function that takes in the number of markers over an segment, and
        returns an iterable of haplotypes that are automatically included in the
        list of input haplotypes to be determined.

    Returns
    -------
    tuple (list[list[0,1]] or None, bool)
        2-tuple consisting of (i) a list of input haplotypes, and (ii) boolean 
        indicating whether all instances of AEML converged. The first entry may
        be None if all instances of AEML over a segment failed, or if the 
        feasible set is always empty for a segment.
    """
    if n_markers < 6:
        print('only works for n_markers >= 6')
        
    n_3m = 4 - n_markers % 4 if n_markers % 4 else 0
    n_4m = (n_markers - 3*n_3m) // 4
    blocks = ([(4*x, 4*x+4) for x in range(n_4m)] + 
              [(4*n_4m + 3*x, 4*n_4m + 3*x + 3) for x in range(n_3m)])
    assert blocks[-1][-1] == n_markers
    
    hap_lists = [mat_by_marker(block[1]-block[0]).T for block in blocks]

    convg = True
    
    while len(blocks) > 1: 
        halfn = len(blocks)//2
        new_lists, convgs = zip(*[PL_aeml_pair(ns, ys, 
                                               blocks[2*x], blocks[2*x+1],
                                               hap_lists[2*x], hap_lists[2*x+1], 
                                               hap_fn, aeml_dir, solver, thres,
                                               maxhaps, trials, inithaps_fn)
                                  for x in range(halfn)])
        new_lists = list(new_lists)
        if not all(convgs):
            convg = False
        if any(haps is None for haps in new_lists):
            return None, False
        hap_lists = new_lists + ([hap_lists[-1]] if len(blocks) % 2 else [])
        blocks = [(blocks[2*x][0], blocks[2*x+1][1])
                  for x in range(halfn)] + ([blocks[-1]] 
                                            if len(blocks) % 2 
                                            else [])        
        assert len(blocks) == len(hap_lists)        
        
    assert blocks[0][0] == 0 and blocks[0][-1] == n_markers
    return hap_lists[0], convg
    
    
def PL_aeml_pair(ns, ys, block1, block2, hap_list1, hap_list2, 
                 hap_fn, aeml_dir, solver, thres, maxhaps, trials, 
                 inithaps_fn=lambda x: []):
    """
    Perform estimation + ligation step for two adjacent segments using AEML as 
    the frequency estimation subroutine. Includes check to ensure that the 
    feasible set is nonempty.

    Parameters
    ----------
    ns : list[int]
        Number of haplotype samples for each pool.
    ys : list[list[int]]
        List consisting of a list per pool, where the inner list consists of the
        allele counts of each marker (over all markers, not just the markers 
        covered by two segments).
    block1 : tuple
        Tuple of two integers (start, end), where `start` is the 0-based index 
        of the leftmost marker of the first segment, and `end` is the 1 larger 
        than the 0-based index of the rightmost marker of the first segment. 
    block2 : tuple
        Tuple of two integers (start, end), where `start` is the 0-based index 
        of the leftmost marker of the second segment, and `end` is the 1 larger 
        than the 0-based index of the rightmost marker of the second segment. 
    hap_list1 : list[list[0,1]]
        List of input haplotypes over the first segment.
    hap_list2 : list[list[0,1]]
        List of input haplotypes over the second segment.
    hap_fn : str
        Filename for input haplotype list to AEML.
    aeml_dir : str
        Directory containing the C program for AEML.
    solver : pulp.core.LpSolver_CMD, optional
        Mixed-integer programming solver that is passed to pulp.LpProblem.solve.
        See https://coin-or.github.io/pulp/technical/solvers.html for a list of 
        possible solvers.
    thres : float > 0
        Frequency threshold of haplotypes to be selected.
    maxhaps : int, optional
        Maximum number of concatenated haplotypes. If None, no maximum is
        imposed.
    trials : int > 0, default 5
        Number of trials for expectation maximization.
    inithaps_fn : Callable[[int], iterable[iterable]], default : lambda x: []
        Function that takes in the number of markers over an segment, and
        returns an iterable of haplotypes that are automatically included in the
        list of input haplotypes to be determined.

    Returns
    -------
    tuple (list[list[0,1]] or None, bool)
        2-tuple consisting of (i) a list of input haplotypes, and (ii) boolean 
        indicating whether all instances of AEML converged. The first entry may 
        be None if all instances of AEML over a segment failed, or if the 
        feasible set is always empty for a segment.
    """
    assert block2[0] == block1[1]
    
    print(f'AEML for markers {block1[0]+1}-{block1[1]}: ', end='')    
    with open(hap_fn, 'w') as fp:
        fp.write(f'{len(hap_list1)}\n')
        for harr in hap_list1:
            fp.write(' '.join([str(h) for h in harr]))
            fp.write('\n')
    aeml_1 = run_AEML(ns, [y[block1[0]:block1[1]] for y in ys], aeml_dir,
                      trials=trials, stab=1e-9, hap_fn=hap_fn)
    
    print(f'AEML for markers {block2[0]+1}-{block2[1]}: ', end='')
    with open(hap_fn, 'w') as fp:
        fp.write(f'{len(hap_list2)}\n')
        for harr in hap_list2:
            fp.write(' '.join([str(h) for h in harr]))
            fp.write('\n')
    aeml_2 = run_AEML(ns, [y[block2[0]:block2[1]] for y in ys], aeml_dir,
                      trials=trials, stab=1e-9, hap_fn=hap_fn)
    
    if aeml_1 is None or aeml_2 is None:
        return None, False
    
    remaining = sorted({p for p in list(aeml_1['pest']) + list(aeml_2['pest']) 
                       if p < thres})
    # check that the feasible set is not empty
    while remaining:
        haps = concat_haps(aeml_1['pest'], aeml_2['pest'], hap_list1, hap_list2,
                           thres=thres, maxhaps=maxhaps, inithaps_fn=inithaps_fn)
        for n, y in zip(ns, ys):
            amat = np.vstack([np.ones(len(haps), int), 
                             np.array(haps).T]).astype(int)
            y = np.array([n] + list(y[block1[0]:block2[1]]))
            nzs = np.arange(len(haps))
            # check if y = Az has a solution
            prob = pulp.LpProblem("test", pulp.LpMinimize)
            z = pulp.LpVariable.dicts("z", nzs, lowBound=0, cat='Integer')
            for j in range(amat.shape[0]):
                prob += (pulp.lpSum([amat[j,k]*z[k] for k in nzs]) == y[j])
            prob.solve(solver)  
            if prob.status != 1:
                # no solution
                assert prob.status == -1
                break
        else:
            # done, break out of while loop
            break
        thres = remaining.pop()
        print(f'decrease threshold to {thres}')
    else:
        # failed
        print(f'failed to combine {len(hap_list1)} haplotypes '
              f'from {block1[0]+1}-{block1[1]} and {len(hap_list2)} haplotypes '
              f'from {block2[0]+1}-{block2[1]}')
        print(y)
        for row in amat:
            print(' '.join([str(x) for x in row]))
        return None, False

    return haps, aeml_1['convg'] and aeml_2['convg']


def PL_mn_approx(ns, ys, n_markers, n_sample, n_burnin, chains,
                 solver, prefix_4ti2, thres, maxhaps=None, inithaps_fn=lambda x: []):
    """
    Determine input haplotypes using partition ligation based, with MCMC-Exact
    as the frequency estimate subroutine. Includes check to ensure that the 
    feasible set is nonempty.

    Parameters
    ----------
    ns : list[int]
        Number of haplotype samples for each pool.
    ys : list[list[int]]
        List consisting of a list per pool, where the inner list consists of the
        allele counts of each marker.
    n_markers : int > 0
        Number of markers.
    n_sample : int > 0
        Number of inference iterations.
    n_burnin : int > 0
        Number of burn-in iterations.
    chains : int > 0
        The number of MCMC chains to run.
    solver : pulp.core.LpSolver_CMD, optional
        Mixed-integer programming solver that is passed to pulp.LpProblem.solve.
        See https://coin-or.github.io/pulp/technical/solvers.html for a list of 
        possible solvers.
    prefix_4ti2 : str, default to calling `find_4ti2_prefix()`
        Prefix to 4ti2's commands depending on installation method, e.g. "" or 
        "4ti2-".
    thres : float > 0
        Frequency threshold of haplotypes to be selected.
    maxhaps : int, optional
        Maximum number of concatenated haplotypes. If None, no maximum is
        imposed.
    inithaps_fn : Callable[[int], iterable[iterable]], default : lambda x: []
        Function that takes in the number of markers over an segment, and
        returns an iterable of haplotypes that are automatically included in the
        list of input haplotypes to be determined.

    Returns
    -------
    tuple (list[list[0,1]] or None, bool)
        2-tuple consisting of (i) a list of input haplotypes, and (ii) boolean 
        indicating whether all instances of AEML converged. The first entry may
        be None if all instances of AEML over a segment failed, or if the 
        feasible set is always empty for a segment.
    """
    if n_markers < 6:
        print('only works for n_markers >= 6')
        
    n_3m = 4 - n_markers % 4 if n_markers % 4 else 0
    n_4m = (n_markers - 3*n_3m) // 4
    blocks = ([(4*x, 4*x+4) for x in range(n_4m)] + 
              [(4*n_4m + 3*x, 4*n_4m + 3*x + 3) for x in range(n_3m)])
    assert blocks[-1][-1] == n_markers
    
    hap_lists = [mat_by_marker(block[1]-block[0]).T for block in blocks]
    
    while len(blocks) > 1: 
        halfn = len(blocks)//2
        new_lists = [PL_mn_approx_pair(ns, ys, 
                                       blocks[2*x], blocks[2*x+1],
                                       hap_lists[2*x], hap_lists[2*x+1], 
                                       n_sample, n_burnin, chains,
                                       solver, prefix_4ti2, 
                                       thres, maxhaps, inithaps_fn)
                     for x in range(halfn)]
        if any(haps is None for haps in new_lists):
            return None
        hap_lists = new_lists + ([hap_lists[-1]] if len(blocks) % 2 else [])
        blocks = [(blocks[2*x][0], blocks[2*x+1][1])
                  for x in range(halfn)] + ([blocks[-1]] 
                                            if len(blocks) % 2 
                                            else [])
        assert len(blocks) == len(hap_lists)        
        
    assert blocks[0][0] == 0 and blocks[0][-1] == n_markers
    return hap_lists[0]
    
    
def PL_mn_approx_pair(ns, ys, block1, block2, hap_list1, hap_list2, 
                      n_sample, n_burnin, chains, solver, prefix_4ti2, 
                      thres, maxhaps, inithaps_fn=lambda x: []):
    """
    Perform estimation + ligation step for two adjacent segments using LC-Sampling 
    as the frequency estimation subroutine. Includes check to ensure that the 
    feasible set is nonempty.

    Parameters
    ----------
    ns : list[int]
        Number of haplotype samples for each pool.
    ys : list[list[int]]
        List consisting of a list per pool, where the inner list consists of the
        allele counts of each marker (over all markers, not just the markers 
        covered by two segments).
    block1 : tuple
        Tuple of two integers (start, end), where `start` is the 0-based index 
        of the leftmost marker of the first segment, and `end` is the 1 larger 
        than the 0-based index of the rightmost marker of the first segment. 
    block2 : tuple
        Tuple of two integers (start, end), where `start` is the 0-based index 
        of the leftmost marker of the second segment, and `end` is the 1 larger 
        than the 0-based index of the rightmost marker of the second segment. 
    hap_list1 : list[list[0,1]]
        List of input haplotypes over the first segment.
    hap_list2 : list[list[0,1]]
        List of input haplotypes over the second segment.
    n_markers : int > 0
        Number of markers.
    n_sample : int > 0
        Number of inference iterations.
    n_burnin : int > 0
        Number of burn-in iterations.
    chains : int > 0
        The number of MCMC chains to run.
    solver : pulp.core.LpSolver_CMD, optional
        Mixed-integer programming solver that is passed to pulp.LpProblem.solve.
        See https://coin-or.github.io/pulp/technical/solvers.html for a list of 
        possible solvers.
    prefix_4ti2 : str, default to calling `find_4ti2_prefix()`
        Prefix to 4ti2's commands depending on installation method, e.g. "" or 
        "4ti2-".
    thres : float > 0
        Frequency threshold of haplotypes to be selected.
    maxhaps : int, optional
        Maximum number of concatenated haplotypes. If None, no maximum is
        imposed.
    inithaps_fn : Callable[[int], iterable[iterable]], default : lambda x: []
        Function that takes in the number of markers over an segment, and
        returns an iterable of haplotypes that are automatically included in the
        list of input haplotypes to be determined.

    Returns
    -------
    tuple (list[list[0,1]] or None, bool)
        2-tuple consisting of (i) a list of input haplotypes, and (ii) boolean 
        indicating whether all instances of AEML converged. The first entry may 
        be None if all instances of AEML over a segment failed, or if the 
        feasible set is always empty for a segment.
    """
    assert block2[0] == block1[1]
    
    print(f'LC-Sampling for markers {block1[0]+1}-{block1[1]}: ', end='')    
    lm_list = []
    for n, y in zip(ns, ys):        
        lm = LatentMult(hap_list1.T, y[block1[0]:block1[1]], n, '../../4ti2-files/pl-mn',
                        solver, prefix_4ti2)
        lm_list.append(lm)
    idata_1 = latent_mult_mcmc(lm_list, len(hap_list1), n_sample, n_burnin, ['mn_approx']*len(ys),
                               jaxify=True, chains=chains)
    
    print(f'LC-Sampling for markers {block2[0]+1}-{block2[1]}: ', end='')
    lm_list = []
    amat = mat_by_marker(block2[1]-block2[0])
    for n, y in zip(ns, ys):            
        lm = LatentMult(hap_list2.T, y[block2[0]:block2[1]], n, '../../4ti2-files/pl-mn',
                        solver, prefix_4ti2)
        lm_list.append(lm)
    idata_2 = latent_mult_mcmc(lm_list, len(hap_list2), n_sample, n_burnin, ['mn_approx']*len(ys),
                               jaxify=True, chains=chains)
    
    pmean_1 = idata_1.posterior.p.values.mean(axis=(0,1))
    pmean_2 = idata_2.posterior.p.values.mean(axis=(0,1))
    remaining = sorted({p for p in list(pmean_1) + list(pmean_2)
                       if p < thres})
    # check that the feasible set is not empty
    while remaining:
        haps = concat_haps(pmean_1, pmean_2, hap_list1, hap_list2,
                           thres=thres, maxhaps=maxhaps, inithaps_fn=inithaps_fn)
        for n, y in zip(ns, ys):
            amat = np.vstack([np.ones(len(haps), int), 
                             np.array(haps).T]).astype(int)
            y = np.array([n] + list(y[block1[0]:block2[1]]))
            nzs = np.arange(len(haps))
            # check if y = Az has a solution
            prob = pulp.LpProblem("test", pulp.LpMinimize)
            z = pulp.LpVariable.dicts("z", nzs, lowBound=0, cat='Integer')
            for j in range(amat.shape[0]):
                prob += (pulp.lpSum([amat[j,k]*z[k] for k in nzs]) == y[j])
            prob.solve(solver)  
            if prob.status != 1:
                # no solution
                assert prob.status == -1
                break
        else:
            # done, break out of while loop
            break
        thres = remaining.pop()
        print(f'decrease threshold to {thres}')
    else:
        # failed
        print(f'failed to combine {len(hap_list1)} haplotypes '
              f'from {block1[0]+1}-{block1[1]} and {len(hap_list2)} haplotypes '
              f'from {block2[0]+1}-{block2[1]}')
        print(y)
        for row in amat:
            print(' '.join([str(x) for x in row]))
        return None
    return haps


def select_haps(pest, thres):    
    """
    Determine haplotypes whose frequency estimates that are at least some
    threshold.

    Parameters
    ----------
    pest : list[float]
        List of haplotype frequency estimates.
    thres : float > 0
        Frequency threshold of haplotypes to be selected.

    Returns
    -------
    Set of indices of `pest` whose values are at least `thres`.
    """
    select = set(np.where(np.array(pest) >= thres)[0])
    # n_markers = len(haps[0])
    # encountered = set()
    # for i in np.argsort(-pest):
    #     select.add(i)
    #     for pos, digit in enumerate(haps[i]):
    #         encountered.add((-1 if digit == 0 else 1)*(pos + 1))
    #     if len(encountered) == 2*n_markers:
    #         break
    return select


def concat_haps(pest1, pest2, haps1, haps2, thres, maxhaps=None, 
                inithaps_fn=lambda x: []):
    """
    Concatenate the most likely pairs of partial haplotypes from two adjacent 
    segments given their frequency estimates, where the partial haplotypes must 
    have their frequency estimates be at least a certain threshold.

    Parameters
    ----------
    pest1 : list[float]
        List of haplotype frequency estimates for the first segment.
    pest2 : list[float]
        List of haplotype frequency estimates for the second segment.
    haps1 : list[list[0,1]]
        List of input haplotypes over the first segment.
    haps2 : list[list[0,1]]
        List of input haplotypes over the second segment.
    thres : float > 0
        Frequency threshold of haplotypes to be selected.
    maxhaps : int, optional
        Maximum number of concatenated haplotypes. If None, no maximum is
        imposed.
    inithaps_fn : Callable[[int], iterable[iterable]], default : lambda x: []
        Function that takes in the number of markers over an segment, and
        returns an iterable of haplotypes that are automatically included in the
        list of input haplotypes to be determined.

    Returns
    -------
    """
    totlen = len(haps1[0]) + len(haps2[0])
    inithaps = inithaps_fn(totlen)
    select1 = select_haps(pest1, thres)
    select2 = select_haps(pest2, thres)

    pair_dict = {(idx1, idx2): (pest1[idx1], pest2[idx2])
                 for idx1 in select1 for idx2 in select2}
    hap_pairs = sorted(pair_dict, key=lambda x: (-min(pair_dict[x]),
                                                 -max(pair_dict[x])))

    inithaps_set = {tuple(hap) for hap in inithaps}
    hap_list = [list(hap) for hap in inithaps_set]
    to_add = (maxhaps - len(inithaps_set) 
              if maxhaps is not None 
              else len(hap_pairs))

    for idx1, idx2 in hap_pairs:
        if to_add <= 0:
            break
        
        cand = list(haps1[idx1]) + list(haps2[idx2])
        if tuple(cand) not in inithaps_set:
            hap_list.append(cand)
            to_add -= 1

    return hap_list

