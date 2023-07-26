"""
This file consists code adapted and modified from pymc.sampling.jax (v5.1.2) and 
numpyro.infer.hmc_gibbs (v0.11.0). There are three key differences from the original 
version of the code:

1. Workaround for a bug due to jax._src.maps.xmap (as of v0.4.6) which occurs when the 
argument `postprocessing_chunks` in pymc.sampling.jax.sample_numpyro_nuts is used, for 
details see . The use of `xmap` is replaced with the basic `jax.vmap` instead. 

2. Run NumPyro's NUTS-within-Gibbs sampler based on a PyMC model. HMCGibbs, as implemented 
by NumPyro, must be defined via a NumPyro model. However, PyMC interfaces with NumPyro via 
a potential function instead. As implemented here, the class HMCGibbsPotFn extends HMCGibbs 
with some methods overridden such that the class is compatible with a potential function 
instead of a NumPyro model.

3. Change the Gibbs kernel after MCMC warmup. For latent count sampling methods, we augment 
each Markov basis with additional proposal directions right before the inference phase, 
based on which latent counts have the loweset ESS during the warmup phase. This change is 
specified for `sampling_numpyro_nuts_gibbs` via the `update_gibbs_fn` argument.
"""

from datetime import datetime
from time import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import sys
import copy

import numpy as np
import xarray
import arviz as az
import pytensor.tensor as pt
from pytensor.tensor import TensorVariable

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, HMC, NUTS, HMCGibbs
from numpyro.infer.hmc import hmc
from numpyro.infer.hmc_gibbs import HMCGibbsState

from arviz.data.base import make_attrs
from pymc.backends.arviz import find_constants, find_observations

import pymc as pm
from pymc import Model, modelcontext
from pymc.initial_point import StartDict
from pymc.util import (
    RandomState,
    _get_seeds_per_chain,
    get_default_varnames,
)
from pymc.sampling.jax import (
    _get_batched_jittered_initial_points,
    get_jaxified_logp,
    get_jaxified_graph,
    _sample_stats_to_xarray,
    _update_numpyro_nuts_kwargs,
    _update_coords_and_dims,
)
import jax
import jax.numpy as jnp


### Adapted from code based on pymc.sampling.jax

# rewrite pymc.sampling.jax._postprocess_samples to avoid xmap
def _postprocess_samples(
    jax_fn: List[TensorVariable],
    raw_mcmc_samples: List[TensorVariable],
    postprocessing_backend: str,
    num_chunks: Optional[int] = None,
) -> List[TensorVariable]:
    if num_chunks is not None:
        # dims are vars, chains, draws, ...
        raw_mcmc_samples = jax.device_put(
                               raw_mcmc_samples, 
                               jax.devices(postprocessing_backend)[0]
                           )
        f = jax.vmap(jax.vmap(jax_fn))
        draws = len(raw_mcmc_samples[0][0])
        segs = list(range(0, draws, draws // num_chunks)) + [draws]
        # dims are chunks, vars, chains, draws, ...
        outputs = [f(*[var_samples[:,i:j] for var_samples in raw_mcmc_samples])
                   for i, j in zip(segs[:-1], segs[1:])]
        # dims of var_chunks are chunks, chains, draws, ...
        return [jnp.concatenate(var_chunks, axis=1) 
                for var_chunks in zip(*outputs)]
    else:
        return jax.vmap(jax.vmap(jax_fn))(
            *jax.device_put(raw_mcmc_samples, 
                            jax.devices(postprocessing_backend)[0])
        )


# rewrite pymc.sampling.jax._get_log_likelihood to use the rewritten 
# _postprocess_samples
def _get_log_likelihood(
    model: Model, samples, backend=None, num_chunks: Optional[int] = None
) -> Dict:
    """Compute log-likelihood for all observations"""
    elemwise_logp = model.logp(model.observed_RVs, sum=False)
    jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=elemwise_logp)
    result = _postprocess_samples(jax_fn, samples, backend, 
                                  num_chunks=num_chunks)
    return {v.name: r for v, r in zip(model.observed_RVs, result)}


# rewrite pymc.sampling.jax.sample_numpyro_nuts to use the rewritten 
# _postprocess_samples
def sample_numpyro_nuts(
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.8,
    random_seed: Optional[RandomState] = None,
    initvals: Optional[Union[StartDict, Sequence[Optional[StartDict]]]] = None,
    model: Optional[Model] = None,
    var_names: Optional[Sequence[str]] = None,
    progressbar: bool = True,
    keep_untransformed: bool = False,
    chain_method: str = "parallel",
    postprocessing_backend: Optional[str] = None,
    postprocessing_chunks: Optional[int] = None,
    idata_kwargs: Optional[Dict] = None,
    nuts_kwargs: Optional[Dict] = None,
) -> az.InferenceData:
    """
    Draw samples from the posterior using the NUTS method from the ``numpyro`` library.

    Parameters
    ----------
    draws : int, default 1000
        The number of samples to draw. The number of tuned samples are discarded by
        default.
    tune : int, default 1000
        Number of iterations to tune. Samplers adjust the step sizes, scalings or
        similar during tuning. Tuning samples will be drawn in addition to the number
        specified in the ``draws`` argument.
    chains : int, default 4
        The number of chains to sample.
    target_accept : float in [0, 1], default 0.8
        The step size is tuned such that we approximate this acceptance rate. Higher
        values like 0.9 or 0.95 often work better for problematic posteriors.
    random_seed : int, RandomState or Generator, optional
        Random seed used by the sampling steps.
    initvals: StartDict or Sequence[Optional[StartDict]], optional
        Initial values for random variables provided as a dictionary (or sequence of
        dictionaries) mapping the random variable (by name or reference) to desired
        starting values.
    model : Model, optional
        Model to sample from. The model needs to have free random variables. When inside
        a ``with`` model context, it defaults to that model, otherwise the model must be
        passed explicitly.
    var_names : sequence of str, optional
        Names of variables for which to compute the posterior samples. Defaults to all
        variables in the posterior.
    progressbar : bool, default True
        Whether or not to display a progress bar in the command line. The bar shows the
        percentage of completion, the sampling speed in samples per second (SPS), and
        the estimated remaining time until completion ("expected time of arrival"; ETA).
    keep_untransformed : bool, default False
        Include untransformed variables in the posterior samples. Defaults to False.
    chain_method : str, default "parallel"
        Specify how samples should be drawn. The choices include "sequential",
        "parallel", and "vectorized".
    postprocessing_backend : Optional[str]
        Specify how postprocessing should be computed. gpu or cpu
    postprocessing_chunks: Optional[int], default None
        Specify the number of chunks the postprocessing should be computed in. More
        chunks reduces memory usage at the cost of losing some vectorization, None
        uses jax.vmap
    idata_kwargs : dict, optional
        Keyword arguments for :func:`arviz.from_dict`. It also accepts a boolean as
        value for the ``log_likelihood`` key to indicate that the pointwise log
        likelihood should not be included in the returned object. Values for
        ``observed_data``, ``constant_data``, ``coords``, and ``dims`` are inferred from
        the ``model`` argument if not provided in ``idata_kwargs``. If ``coords`` and
        ``dims`` are provided, they are used to update the inferred dictionaries.
    nuts_kwargs: dict, optional
        Keyword arguments for :func:`numpyro.infer.NUTS`.

    Returns
    -------
    InferenceData
        ArviZ ``InferenceData`` object that contains the posterior samples, together
        with their respective sample stats and pointwise log likeihood values (unless
        skipped with ``idata_kwargs``).
    """

    import numpyro

    from numpyro.infer import MCMC, NUTS

    model = modelcontext(model)

    if var_names is None:
        var_names = model.unobserved_value_vars

    vars_to_sample = list(get_default_varnames(var_names, include_transformed=keep_untransformed))

    coords = {
        cname: np.array(cvals) if isinstance(cvals, tuple) else cvals
        for cname, cvals in model.coords.items()
        if cvals is not None
    }

    dims = {
        var_name: [dim for dim in dims if dim is not None]
        for var_name, dims in model.named_vars_to_dims.items()
    }

    (random_seed,) = _get_seeds_per_chain(random_seed, 1)

    tic1 = datetime.now()
    print("Compiling...", file=sys.stdout)

    init_params = _get_batched_jittered_initial_points(
        model=model,
        chains=chains,
        initvals=initvals,
        random_seed=random_seed,
    )
    
    logp_fn = get_jaxified_logp(model, negative_logp=False)

    nuts_kwargs = _update_numpyro_nuts_kwargs(nuts_kwargs)
    nuts_kernel = NUTS(
        potential_fn=logp_fn,
        target_accept_prob=target_accept,
        **nuts_kwargs,
    )

    pmap_numpyro = MCMC(
        nuts_kernel,
        num_warmup=tune,
        num_samples=draws,
        num_chains=chains,
        postprocess_fn=None,
        chain_method=chain_method,
        progress_bar=progressbar,
    )

    tic2 = datetime.now()
    print("Compilation time = ", tic2 - tic1, file=sys.stdout)

    print("Sampling...", file=sys.stdout)

    map_seed = jax.random.PRNGKey(random_seed)
    if chains > 1:
        map_seed = jax.random.split(map_seed, chains)

    pmap_numpyro.run(
        map_seed,
        init_params=init_params,
        extra_fields=(
            "num_steps",
            "potential_energy",
            "energy",
            "adapt_state.step_size",
            "accept_prob",
            "diverging",
        ),
    )

    raw_mcmc_samples = pmap_numpyro.get_samples(group_by_chain=True)

    tic3 = datetime.now()
    print("Sampling time = ", tic3 - tic2, file=sys.stdout)

    print("Transforming variables...", file=sys.stdout)
    jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=vars_to_sample)
    result = _postprocess_samples(
        jax_fn, raw_mcmc_samples, postprocessing_backend, num_chunks=postprocessing_chunks
    )
    mcmc_samples = {v.name: r for v, r in zip(vars_to_sample, result)}

    tic4 = datetime.now()
    print("Transformation time = ", tic4 - tic3, file=sys.stdout)

    if idata_kwargs is None:
        idata_kwargs = {}
    else:
        idata_kwargs = idata_kwargs.copy()

    if idata_kwargs.pop("log_likelihood", False):
        tic5 = datetime.now()
        print("Computing Log Likelihood...", file=sys.stdout)
        log_likelihood = _get_log_likelihood(
            model,
            raw_mcmc_samples,
            backend=postprocessing_backend,
            num_chunks=postprocessing_chunks,
        )
        tic6 = datetime.now()
        print("Log Likelihood time = ", tic6 - tic5, file=sys.stdout)
    else:
        log_likelihood = None

    attrs = {
        "sampling_time": (tic3 - tic2).total_seconds(),
    }

    posterior = mcmc_samples
    # Update 'coords' and 'dims' extracted from the model with user 'idata_kwargs'
    # and drop keys 'coords' and 'dims' from 'idata_kwargs' if present.
    _update_coords_and_dims(coords=coords, dims=dims, idata_kwargs=idata_kwargs)
    # Use 'partial' to set default arguments before passing 'idata_kwargs'
    to_trace = partial(
        az.from_dict,
        log_likelihood=log_likelihood,
        observed_data=find_observations(model),
        constant_data=find_constants(model),
        sample_stats=_sample_stats_to_xarray(pmap_numpyro),
        coords=coords,
        dims=dims,
        attrs=make_attrs(attrs, library=numpyro),
    )
    az_trace = to_trace(posterior=posterior, **idata_kwargs)
    return az_trace


# analogue to pymc.sampling.jax._sample_stats_to_xarray for HMCGibbs
def _sample_stats_to_xarray_hmc_gibbs(posterior):
    """Extract sample_stats from NumPyro posterior for HMCGibbs."""
    rename_key = {
        "hmc_state.energy": "energy",
        "hmc_state.diverging": "diverging",
        "hmc_state.potential_energy": "lp",
        "hmc_state.adapt_state.step_size": "step_size",
        "hmc_state.num_steps": "n_steps",
        "hmc_state.accept_prob": "acceptance_rate",
    }
    data = {}
    for stat, value in posterior.get_extra_fields(group_by_chain=True).items():
        if isinstance(value, (dict, tuple)):
            continue
        name = rename_key.get(stat, stat)
        value = value.copy()
        data[name] = value
        if stat == "num_steps":
            data["tree_depth"] = np.log2(value).astype(int) + 1
    return data


# analogue to sample_numpyro_nuts for HMCGibbs
def sample_numpyro_nuts_gibbs(
    gibbs_fn: Callable, 
    gibbs_sites: Sequence[str],
    gibbs_cond: Sequence[TensorVariable] = [],
    draws: int = 1000,
    tune: int = 1000,
    thinning: Optional[int] = 1,
    chains: int = 4,
    target_accept: float = 0.8,
    random_seed: Optional[RandomState] = None,
    initvals: Optional[Union[StartDict, Sequence[Optional[StartDict]]]] = None,
    model: Optional[Model] = None,
    var_names: Optional[Sequence[str]] = None,
    progressbar: bool = True,
    keep_untransformed: bool = False,
    chain_method: str = "parallel",
    postprocessing_backend: Optional[str] = None,
    postprocessing_chunks: Optional[int] = None,
    idata_kwargs: Optional[Dict] = None,
    nuts_kwargs: Optional[Dict] = None,
    update_gibbs_fn: Optional[Callable] = None,
    **kwargs,
) -> az.InferenceData:
    """
    Draw samples from the posterior using the NUTS method from the ``numpyro``.
    In contrast to `pymc.sampling.jax.sample_numpyro_nuts`, the trace is returned
    along with the Numpyro MCMC sampler object.

    Parameters
    ----------
    gibbs_fn : callable[[jax.random.PRNGKey, dict, dict], dict]
        Function that takes in a JAX random seed, current values of the non-HMC
        sites as a dictionary, current values of the HMC sites as a dictionary,
        and returns a dictionary of the updated non-HMC sites.
    gibbs_sites : list[str]
        List of variable names of non-NUTS sites.
    gibbs_cond : list[str]
        List of variable names of site values needed for sampling via `gibbs_fn`.
    draws : int, default 1000
        The number of samples to draw. The number of tuned samples are discarded by
        default.
    tune : int, default 1000
        Number of iterations to tune. Samplers adjust the step sizes, scalings or
        similar during tuning. Tuning samples will be drawn in addition to the number
        specified in the ``draws`` argument.
    thinning : int > 0, default 1
        Positive integer that controls the fraction of post-warmup samples that are
        retained.
    chains : int, default 4
        The number of chains to sample.
    target_accept : float in [0, 1], default 0.8
        The step size is tuned such that we approximate this acceptance rate. Higher
        values like 0.9 or 0.95 often work better for problematic posteriors.
    random_seed : int, RandomState or Generator, optional
        Random seed used by the sampling steps.
    initvals: StartDict or Sequence[Optional[StartDict]]
        Initial values for random variables provided as a dictionary (or sequence of
        dictionaries) mapping the random variable (by name or reference) to desired
        starting values.
    model : Model, optional
        Model to sample from. The model needs to have free random variables. When inside
        a ``with`` model context, it defaults to that model, otherwise the model must be
        passed explicitly.
    var_names : sequence of str, optional
        Names of variables for which to compute the posterior samples. Defaults to all
        variables in the posterior.
    progressbar : bool, default True
        Whether or not to display a progress bar in the command line. The bar shows the
        percentage of completion, the sampling speed in samples per second (SPS), and
        the estimated remaining time until completion ("expected time of arrival"; ETA).
    keep_untransformed : bool, default False
        Include untransformed variables in the posterior samples. Defaults to False.
    chain_method : str, default "parallel"
        Specify how samples should be drawn. The choices include "sequential",
        "parallel", and "vectorized".
    postprocessing_backend : Optional[str]
        Specify how postprocessing should be computed. gpu or cpu
    postprocessing_chunks: Optional[int], default None
        Specify the number of chunks the postprocessing should be computed in. More
        chunks reduces memory usage at the cost of losing some vectorization, None
        uses jax.vmap
    idata_kwargs : dict, optional
        Keyword arguments for :func:`arviz.from_dict`. It also accepts a boolean as
        value for the ``log_likelihood`` key to indicate that the pointwise log
        likelihood should not be included in the returned object. Values for
        ``observed_data``, ``constant_data``, ``coords``, and ``dims`` are inferred from
        the ``model`` argument if not provided in ``idata_kwargs``. If ``coords`` and
        ``dims`` are provided, they are used to update the inferred dictionaries.
    nuts_kwargs : dict, optional
        Keyword arguments for :func:`numpyro.infer.NUTS`.
    update_gibbs_fn : callable, optional
        Function that returns the updated `gibbs_fn` after the burn-in phase. The function
        signature is `update_gibbs_fn(samples, tune, gibbs_sites, gibbs_idxs)`, where the
        arguments are:
            samples : list[array]
                List of MCMC samples per variable from the warmup phase as returned by
                `numpyro.infer.mcmc.MCMC.get_samples(group_by_chain=True)`. 
            tune : int
                Number of warmup iterations.
            gibbs_sites : list[str]
                List of variable names of non-NUTS sites which `gibbs_fn` updates.
            gibbs_idxs : list[str]
                Indices of `samples` that correspond to `gibbs_sites`.

    Returns
    -------
    tuple (InferenceData, numpyro.infer.mcmc.MCMC)
        2-tuple consisting of (i) ArviZ ``InferenceData`` object that contains 
        the posterior samples, together with their respective sample stats and 
        pointwise log likeihood values (unless skipped with ``idata_kwargs``),
        and (ii) sampler as a Numpyro MCMC object.
    """
    model = modelcontext(model)

    if var_names is None:
        var_names = model.unobserved_value_vars

    vars_to_sample = list(get_default_varnames(var_names, include_transformed=keep_untransformed))

    coords = {
        cname: np.array(cvals) if isinstance(cvals, tuple) else cvals
        for cname, cvals in model.coords.items()
        if cvals is not None
    }

    dims = {
        var_name: [dim for dim in dims if dim is not None]
        for var_name, dims in model.named_vars_to_dims.items()
    }

    (random_seed,) = _get_seeds_per_chain(random_seed, 1)
    
    # modification: get NUTS and Gibbs indices
    nuts_idxs, gibbs_idxs = [], []
    for idx, var in enumerate(model.value_vars):
        if var.name in gibbs_sites:
            gibbs_idxs.append(idx)
        else:
            nuts_idxs.append(idx)

    tic1 = datetime.now()
    print("Compiling...", file=sys.stdout)

    init_params = _get_batched_jittered_initial_points(
        model=model,
        chains=chains,
        initvals=initvals,
        random_seed=random_seed,
    )

    # modification: replace jittered initial values for non-NUTS sites if 
    # already specified in initvals
    for name, idx in zip(gibbs_sites, gibbs_idxs):
        if chains == 1:
            if name not in initvals:
                continue
            if not isinstance(initvals, dict):
                initvals = initvals[0]
            init_params[idx] = initvals[name]
        else:
            if name not in initvals[0]:
                continue
            for i in range(chains):
                init_params[idx][i] = initvals[i][name]

    logp_fn = get_jaxified_logp(model, negative_logp=False)
    
    # modification: after each NUTS iteration, transform variables needed for gibbs_fn
    if gibbs_cond:
        jax_fn = get_jaxified_graph(
            inputs=[model.value_vars[idx] for idx in nuts_idxs],
            outputs=model.replace_rvs_by_values([model[name] for name in gibbs_cond])
        )
        postprocess_fn = lambda z: {name: val for name, val in zip(gibbs_cond, jax_fn(*z))}
    else:
        postprocess_fn = None

    nuts_kwargs = _update_numpyro_nuts_kwargs(nuts_kwargs)
    nuts_kernel = NUTS(
        potential_fn=logp_fn,
        target_accept_prob=target_accept,
        **nuts_kwargs,
    )
    nuts_sites = [var.name for var in model.value_vars if var.name not in gibbs_sites]
    nuts_gibbs_kernel = HMCGibbsPotFn(nuts_kernel, gibbs_fn, nuts_sites, gibbs_sites, 
                                      nuts_idxs, gibbs_idxs, postprocess_fn)

    pmap_numpyro = MCMC(
        nuts_gibbs_kernel,
        num_warmup=tune,
        num_samples=draws,
        thinning=thinning,
        num_chains=chains,
        postprocess_fn=None,
        chain_method=chain_method,
        progress_bar=progressbar,
    )

    tic2 = datetime.now()
    print("Compilation time = ", tic2 - tic1, file=sys.stdout)

    print("Sampling...", file=sys.stdout)

    warmup_seed, sample_seed = jax.random.split(jax.random.PRNGKey(random_seed), 2)
    if chains > 1:
        warmup_seed = jax.random.split(warmup_seed, chains)
        sample_seed = jax.random.split(sample_seed, chains)
    
    # modification: run burn-in and inference phases separately if Gibbs kernel changes
    if update_gibbs_fn is not None:
        pmap_numpyro.thinning = 1
        print('Burn-in phase...')
        pmap_numpyro.warmup(
            warmup_seed,
            init_params=init_params,
            collect_warmup=True,
            extra_fields=(
                "hmc_state.num_steps",
                "hmc_state.potential_energy",
                "hmc_state.energy",
                "hmc_state.adapt_state.step_size",
                "hmc_state.accept_prob",
                "hmc_state.diverging",
            ),
        )
        print('Adapting Gibbs kernel...')
        t = time()
        nuts_gibbs_kernel._gibbs_fn = update_gibbs_fn(
            pmap_numpyro.get_samples(group_by_chain=True), 
            tune, gibbs_sites, gibbs_idxs
        )
        extend_t = time() - t
        pmap_numpyro.thinning = thinning
        init_params = pmap_numpyro._warmup_state.z
        print('Inference phase...')
    
    # usual run (warmup does not rerun if already run)
    pmap_numpyro.run(
        sample_seed,
        init_params=init_params,
        extra_fields=(
            "hmc_state.num_steps",
            "hmc_state.potential_energy",
            "hmc_state.energy",
            "hmc_state.adapt_state.step_size",
            "hmc_state.accept_prob",
            "hmc_state.diverging",
        ),
    )

    raw_mcmc_samples = pmap_numpyro.get_samples(group_by_chain=True)

    tic3 = datetime.now()
    print("Sampling time = ", tic3 - tic2, file=sys.stdout)

    print("Transforming variables...", file=sys.stdout)
    jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=vars_to_sample)
    result = _postprocess_samples(
        jax_fn, raw_mcmc_samples, postprocessing_backend, num_chunks=postprocessing_chunks
    )
    mcmc_samples = {v.name: r for v, r in zip(vars_to_sample, result)}

    tic4 = datetime.now()
    print("Transformation time = ", tic4 - tic3, file=sys.stdout)

    if idata_kwargs is None:
        idata_kwargs = {}
    else:
        idata_kwargs = idata_kwargs.copy()

    if idata_kwargs.pop("log_likelihood", False):
        tic5 = datetime.now()
        print("Computing Log Likelihood...", file=sys.stdout)
        log_likelihood = _get_log_likelihood(
            model,
            raw_mcmc_samples,
            backend=postprocessing_backend,
            num_chunks=postprocessing_chunks,
        )
        tic6 = datetime.now()
        print("Log Likelihood time = ", tic6 - tic5, file=sys.stdout)
    else:
        log_likelihood = None

    attrs = {
        "sampling_time": (tic3 - tic2).total_seconds(),
    }

    posterior = mcmc_samples
    # Update 'coords' and 'dims' extracted from the model with user 'idata_kwargs'
    # and drop keys 'coords' and 'dims' from 'idata_kwargs' if present.
    _update_coords_and_dims(coords=coords, dims=dims, idata_kwargs=idata_kwargs)
    # Use 'partial' to set default arguments before passing 'idata_kwargs'
    to_trace = partial(
        az.from_dict,
        log_likelihood=log_likelihood,
        observed_data=find_observations(model),
        constant_data=find_constants(model),
        sample_stats=_sample_stats_to_xarray_hmc_gibbs(pmap_numpyro),
        coords=coords,
        dims=dims,
        attrs=make_attrs(attrs, library=numpyro),
    )
    az_trace = to_trace(posterior=posterior, **idata_kwargs)
    az_trace.attrs['extend_time'] = extend_t

    # modification: return trace and sampler
    return az_trace, pmap_numpyro



### Adapted from code based on numpyro.infer.hmc_gibbs

class HMCGibbsPotFn(HMCGibbs):
    """
    HMC-within-Gibbs sampler where the potential function is provided by PyMC.
    For other details, refer to numpyro.infer.HMCGibbs.

    Parameters
    ----------
    inner_kernel : numpyro.infer.HMC
        HMC or NUTS sampler.
    gibbs_fn : callable[[jax.random.PRNGKey, dict, dict], dict]
        Function that takes in a JAX random seed, current values of the non-HMC
        sites as a dictionary, current values of the HMC sites as a dictionary,
        and returns a dictionary of the updated non-HMC sites.
    hmc_sites : list[str]
        List of variable names of HMC sites.
    gibbs_sites : list[str]
        List of variable names of non-HMC sites.
    hmc_idxs : list[int]
        List of indices of the argument of `inner_kernel` that corresponds to 
        the HMC sites.
    gibbs_idxs : list[int]
        List of indices of the argument of `inner_kernel` that corresponds to 
        the non-HMC sites.
    postprocess_fn : callable[[HMCState.z], dict]
        Function that creates the dictionary of HMC states given HMCState.z, 
        which can be used in `gibbs_fn`.
    """
    def __init__(self, inner_kernel, gibbs_fn, hmc_sites, gibbs_sites, hmc_idxs, 
                 gibbs_idxs, postprocess_fn):
        if not isinstance(inner_kernel, HMC):
            raise ValueError("inner_kernel must be a HMC or NUTS sampler.")
        if not callable(gibbs_fn):
            raise ValueError("gibbs_fn must be a callable")
        assert (
            inner_kernel._potential_fn is not None
        ), (
            "HMCGibbsPotFn does not support models not specified "
            "via a potential function."
        )

        self.inner_kernel = copy.copy(inner_kernel)
        self._hmc_sites = hmc_sites
        self._gibbs_sites = gibbs_sites
        self._gibbs_fn = gibbs_fn
        self._hmc_idxs = hmc_idxs
        self._gibbs_idxs = gibbs_idxs
        self._n_inputs = len(hmc_idxs) + len(gibbs_idxs)

        if postprocess_fn is None:
            # default _postprocess_fn turns list into dictionary
            self._postprocess_fn = lambda z: {name: val for name, val in zip(
                                                  self._hmc_sites, z
                                              )}
        else:
            self._postprocess_fn = postprocess_fn
        
    
    # Create potential_fn from the potential function passed from PyMC
    def potential_fn_gen(self, _gibbs_sites):
        def potential_fn(z_hmc):
            assert(len(_gibbs_sites) == len(self._gibbs_idxs))
            assert(len(z_hmc) == len(self._hmc_idxs))
            inputs = [None]*self._n_inputs
            for idx, z in zip(self._hmc_idxs, z_hmc):
                inputs[idx] = z
            for idx, z in zip(self._gibbs_idxs, _gibbs_sites):
                inputs[idx] = z
            return self.inner_kernel._potential_fn(inputs)
        return potential_fn
        
        
    # HMCGibbs.init because of a different potential_fn
    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs.copy()

        rng_key, key_z = jax.random.split(rng_key)
        gibbs_sites = [init_params[gibbs_idx] for gibbs_idx in self._gibbs_idxs]
        hmc_sites = [init_params[hmc_idx] for hmc_idx in self._hmc_idxs]
        model_kwargs["_gibbs_sites"] = gibbs_sites
        
        # overwrite the HMC's _init_state method, perform the `hmc` side effect separately
        self.inner_kernel._init_state = lambda rng_key, model_args, model_kwargs, init_params: init_params        
        self.inner_kernel._init_fn, self.inner_kernel._sample_fn = hmc(
                potential_fn_gen=self.potential_fn_gen,
                kinetic_fn=self.inner_kernel._kinetic_fn,
                algo=self.inner_kernel._algo,
            )
        hmc_state = self.inner_kernel.init(
            key_z, num_warmup, hmc_sites, model_args, model_kwargs
        )

        return jax.device_put(HMCGibbsState(init_params, hmc_state, rng_key))
    
    
    # HMCGibbs.sample because of a different potential_fn
    def sample(self, state, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        rng_key, rng_gibbs = jax.random.split(state.rng_key)

        def potential_fn(z_gibbs, z_hmc):
            return self.potential_fn_gen(z_gibbs)(z_hmc)

        z_gibbs_dict = {name: state.z[idx]
        for name, idx in zip(self._gibbs_sites, self._gibbs_idxs)}
        z_hmc_dict = self._postprocess_fn(state.hmc_state.z)
        z_gibbs_dict = self._gibbs_fn(
            state.hmc_state.i, rng_key=rng_gibbs, 
            gibbs_sites=z_gibbs_dict, hmc_sites=z_hmc_dict
        )
        z_gibbs = [z_gibbs_dict[name] for name in self._gibbs_sites]

        if self.inner_kernel._forward_mode_differentiation:
            pe = potential_fn(z_gibbs, state.hmc_state.z)
            z_grad = jax.jacfwd(partial(potential_fn, z_gibbs))(state.hmc_state.z)
        else:
            pe, z_grad = jax.value_and_grad(partial(potential_fn, z_gibbs))(
                state.hmc_state.z
            )
        hmc_state = state.hmc_state._replace(z_grad=z_grad, potential_energy=pe)

        model_kwargs_ = model_kwargs.copy()
        model_kwargs_["_gibbs_sites"] = z_gibbs
        hmc_state = self.inner_kernel.sample(hmc_state, model_args, model_kwargs_)

        inputs = [None]*self._n_inputs
        for idx, z in zip(self._hmc_idxs, hmc_state.z):
            inputs[idx] = z
        for idx, z in zip(self._gibbs_idxs, z_gibbs):
            inputs[idx] = z

        return HMCGibbsState(inputs, hmc_state, rng_key)

    
    # override HMCGibbs.postprocess_fn as we do not implement it
    def postprocess_fn(self, args, kwargs):
        if self.inner_kernel._postprocess_fn is None:
            return numpyro.util.identity
        else:
            raise NotImplementedError