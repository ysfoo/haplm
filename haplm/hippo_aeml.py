"""
Wrapper code to call the HIPPO and AEML C programs.
"""

import numpy as np
from scipy.stats import entropy

from time import time
import subprocess as sp
import arviz as az
import xarray

def run_AEML(ns, ys, aeml_dir, trials=10,
             print_trial=True, seed=None, n_iterations=10**4, stab=None,
             use_ent=False, hap_fn=None):
    """
    Run the AEML executable. Note that the observed counts must be allele counts
    of each marker.

    Parameters
    ----------
    ns : list[int]
        Number of haplotype samples for each pool.
    ys : list[list[int]]
        List consisting of a list per pool, where the inner list consists of the
        allele counts of each marker.
    aeml_dir : str
        Directory containing the AEML executable.
    trials : int > 0, default 10
        Number of trials for expectation maximization.
    print_trial : bool, default True
        Boolean for whether each trial number is printed.
    seed : int, optional
        Random seed.
    n_iterations : int > 0, default 10**4
        Maximum number of iterations per trial.
    stab : float > 0, optional
        Stabilising constant in case of a near-singular covariance matrix. No
        stabilisation if not specified.
    use_ent : bool, default False
        If True, selects the best trial as the frequency estimate with the 
        highest entropy. Otherwise, selects the best trial according to the 
        highest log-likelihood. 
    hap_fn : str
        Filename for input haplotype list to AEML.

    Returns
    -------
    dict
        Dictionary with the following key-value pairs:
            pest : array
                Haplotype frequency estimates.
            time : float
                Time taken by AEML in seconds.
            convg : bool
                Whether AEML converged for `pest`.
    """
    n_pools = len(ys)
    n_markers = len(ys[0])

    # write observed data to 'hippo_aeml.data'
    write_data(ns, ys)

    maxval = -np.inf # max objective value so far
    pest = None # frequency estimates
    found_convg = False # whether AEML converged for pest

    if print_trial:
        print('Trial', end='')

    t0 = time()
    for trial in range(trials):
        if print_trial:
            print(f' {trial+1}', end='', flush=True)
        trial += 1

        # write input
        if seed is not None:
            with open('AEML_seed', 'w') as fp:
                fp.write(str(seed ^ trial + 1))

        par_fn = 'aeml.par'
        with open(par_fn, 'w') as fp:
            fp.write(f'data_file hippo_aeml.data\n')
            fp.write(f'n_loci {n_markers}\n')
            fp.write(f'n_pools {n_pools}\n')

            # first attempt is from all equal init, as per original publication
            fp.write(f'random_init {min(1, trial)}\n') 

            fp.write(f'n_iterations {n_iterations}\n')
            if stab is not None:
                fp.write(f'stab {stab}\n')
            if hap_fn is not None:
                fp.write(f'hap_file {hap_fn}\n')

        if hap_fn is not None:
            with open(hap_fn) as fp:
                H = int(next(fp))
                hap_dict = {''.join(line.split()): i 
                            for i, line in enumerate(fp)}
        else:
            H = 2**n_markers

        # execute AEML
        with open('AEML.log', 'w') as fp:
            sp.run([f'{aeml_dir}AEML', par_fn], stdout=fp, stderr=fp)

        # parse output
        with open('AEML.log') as fp:
            for line in fp:
                pass
            if 'Exits' not in line:
                assert "Covariance matrix is zero matrix!" in line
                continue # did not terminate properly   

        # no convergence if max iterations is reached
        with open('AEML_monitor.out') as fp:
            lines = 0
            for line in fp:
                lines += 1
            loglike = float(line.strip())
        convg = lines < n_iterations

        # get estimate
        pest = np.zeros(H)
        with open('AEML.out') as fp:
            for line in fp:
                hstr, pstr = line.split()
                if hap_fn is None:
                    h = sum(1 << i for i, b in enumerate(hstr) if b=='1')
                else:
                    h = hap_dict[hstr]
                pest[h] = float(pstr)

        if use_ent:
            # choose entropy as objective value
            currval = entropy(pest)
        else:
            # choose approx log-likelihood as objective value
            currval = loglike

        # continue loop if objective value is not better
        if currval <= maxval:
            continue

        if convg:
            found_convg = True
        maxval = currval
        selected = pest.copy() # store best estimate so far

    if print_trial:
        print()

    # all trials failed
    if not maxval > -np.inf:
        if stab is None:
            raise RuntimeError("Singular covariance, "
                               "need to include stabilising constant")

        print('Stabilising constant too small, multiplying by 10')
        stab *= 10

        return run_AEML(ns, ys, aeml_dir, trials,
                        print_trial, seed, n_iterations, stab,
                        use_ent, hap_fn)

    return {'pest': selected, 'time': time() - t0, 'convg': found_convg}


def run_hippo(ns, ys, n_sample, n_burnin, hippo_dir, 
              thin=1, chains=5, alpha=None, gamma=None, stab=None,
              hap_fn=None, print_chain_num=True, seed=0):
    """
    Run the HIPPO executable. Note that the observed counts must be allele 
    counts of each marker.

    Parameters
    ----------
    ns : list[int]
        Number of haplotype samples for each pool.
    ys : list[list[int]]
        List consisting of a list per pool, where the inner list consists of the
        allele counts of each marker.
    n_sample : int > 0
        Number of iterations in the inference phase.
    n_burnin : int > 0
        Number of iterations in the burn-in phase.
    hippo_dir : str
        Directory containing the HIPPO executable.
    thin : int > 0, default 1
        Positive integer that controls the fraction of post-warmup samples that 
        are retained.
    chains : int > 0, default 5
        Number of MCMC chains to run.
    alpha : float > 0, optional
        Dirichlet concentration for haplotype frequencies. Use HIPPO's default
        value of 0.00001 if not specified.
    gamma : float > 0, optional
        Penalty parameter for number of haplotypes. Use HIPPO's default value of 
        8.0 if not specified.
    stab : float > 0, optional
        Stabilising constant in case of a near-singular covariance matrix. No
        stabilisation if not specified.
    hap_fn : str
        Filename for input haplotype list to AEML.
    print_chain_num : bool, default True
        Boolean for whether each chain number is printed.
    seed : int, optional
        Random seed.

    Returns
    -------
    InferenceData
        ArviZ InferenceData object that contains the posterior samples, 
        together sample stats for each chain. The sample stats consists of:
            time_incl_tune : 
                Time taken by each HIPPO chain in seconds, including the burn-in 
                phase.
            time_excl_tune :
                Time taken by each HIPPO chain in seconds, excluding the burn-in 
                phase.
            pmode :
                Posterior mode of each chain.
            avg_logpost :
                Average log-posterior density of each chain.
    """
    n_pools = len(ys)
    n_markers = len(ys[0])
    H = 2**n_markers

    # write observed data to 'hippo_aeml.data'
    write_data(ns, ys)
    
    par_fn = 'hippo.par'
    with open(par_fn, 'w') as fp:
        fp.write(f'data_file hippo_aeml.data\n')
        fp.write(f'n_loci {n_markers}\n')
        fp.write(f'n_pools {n_pools}\n')            
        fp.write(f'n_iterations {n_sample+n_burnin}\n')
        fp.write(f'n_burnin {n_burnin}\n')
        fp.write('tol 0\n')
        fp.write('variable_list 2\n')
        fp.write('write_trace 1\n')
        fp.write(f'thin {thin}\n')
        if alpha is not None:
            fp.write(f'alpha {alpha}\n')
        if gamma is not None:
            fp.write(f'gamma {gamma}\n')
        if stab is not None:
            fp.write(f'stab {stab}\n')
        if hap_fn is not None:
            fp.write(f'hap_file {hap_fn}\n')
    
    if print_chain_num:
        print('Chain', end='')

    trace = []
    times_excl_tune = []
    times_incl_tune = []
    modes = []
    avg_logposts = []
    max_loglike = -np.inf

    # run chains sequentially because of program limitation
    for chain in range(chains):
        if print_chain_num:
            print(f' {chain+1}', end='', flush=True)
        
        t = time()
        with open('hippo_seed', 'w') as fp:
            fp.write(str(seed^chain+1))

        # run HIPPO executable
        with open('hippo.log', 'w') as fp:
            sp.run([f'{hippo_dir}hippo', par_fn], stdout=fp, stderr=fp)
        times_incl_tune.append(time() - t)

        # get trace
        with open('trace.out') as fp:
            trace.append([parse_trace_line(line, H) for line in fp])

        # store log-posterior density reported by `monitor.out`
        logpost = []
        with open('monitor.out') as fp:
            for i, line in enumerate(fp):
                if i*100 < n_burnin:
                    continue
                logpost.append(float(line.split()[0]))
        avg_logposts.append(np.mean(logpost))

        # get time after burn-in
        with open('hippo.log') as fp:
            for line in fp:
                pass
            assert 'Time after burn-in is' in line
            times_excl_tune.append(float(line.split()[-1]))

        # get posterior mode
        with open('MAP.out') as fp:
            loglike = float(next(fp).strip())
            if loglike > max_loglike:
                pmode = [0]*H
                for line in fp:
                    hstr, pstr = line.split()
                    h = sum(1 << i for i, hchar in enumerate(hstr) 
                            if hchar == '1')
                    pmode[h] = float(pstr)

    if print_chain_num:
        print()

    posterior = {'p': (['chain', 'draw', 'p_dim'], np.array(trace))}
    sample_stats = {'time_incl_tune': (['chain'], np.array(times_incl_tune)),
                    'time_excl_tune': (['chain'], np.array(times_excl_tune)),
                    'pmode': (['p_dim'], np.array(pmode)),
                    'avg_logpost': np.array(avg_logposts),
                    }

    return az.InferenceData(posterior=xarray.Dataset(posterior),
                            sample_stats=xarray.Dataset(sample_stats))


def write_data(ns, ys):
    '''
    Write observed data for AEML / HIPPO.

    Parameters
    ----------
    ns : list[int]
        Number of haplotype samples for each pool.
    ys : list[list[int]]
        List consisting of a list per pool, where the inner list consists of the
        allele counts of each marker.

    Returns
    -------
    None
    '''
    with open('hippo_aeml.data', 'w') as fp_out:
        for n, y in zip(ns, ys):
            tokens = [str(n)] + [str(yval) for yval in y]
            fp_out.write(' '.join(tokens))
            fp_out.write('\n')


def parse_trace_line(line, H):
    '''
    Parse a line from `trace.out`, which consists of pairs of tokens. Each pair
    consists of the haplotype index and the corresponding estimate.

    Parameters
    ----------
    line : str
        A line from `trace.out`.
    H : int > 0
        Number of haplotypes.

    Returns
    -------
    list[float]
        Haplotype frequency estimates.
    '''
    p = [0]*H
    tokens = line.split()
    for i in range(0, len(tokens), 2):
        p[int(tokens[i])] = float(tokens[i+1])
    return p
