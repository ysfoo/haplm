# HapLM: Bayesian inference for pooled <ins>hap</ins>lotype data using <ins>l</ins>atent <ins>m</ins>ultinomials

This module implements MCMC methods for latent multinomial distributions, see [[1]](#1) for details. The key application is for pooled genetic data, where counts of incomplete haplotypes are reported for separate pools. Each incomplete haplotype corresponds to a subset of full haplotypes, and the number of full haplotypes per pool are multinomially distributed. The observation of partial sums of these multinomial counts leads to a latent multinomial model. There are 3 proposed methods, all of which can be used either under a setting where the multinomial frequencies are shared across all pools, or under a hierarchical setting where the multinomial frequencies are given a hierarchical prior. According to the method names given in [[1]](#1), the proposed methods are:

- **MCMC-Exact**. An exact method that performs MCMC inference by enumerating all possible latent counts that are compatible with observed data. This is computationally intensive, but is considered the gold standard in terms of accuracy. The MCMC sampler is the No-U-Turn-Sampler (NUTS).
- **MCMC-Approx**. An approximate method that performs MCMC inference via a multinormal approximation, also using NUTS. In the filenames, this method is named as `mn_approx`. This is similar to the HIPPO [[2]](#2) method, with the key difference being that this method does not sample the list of input haplotypes, and uses NUTS as the MCMC sampler.
- **LC-Sampling**. An exact method that performs MCMC inference by sampling the latent counts as model parameters under a Metropolis-within-Gibbs scheme. In the filenames, this method is named as `cgibbs` (Collapsed Gibbs) when a conjugate prior is used for the non-hierarchical case. In the hierarchical case, there is no conjugacy and the method is named `gibbs`, and NUTS is used for sampling the continuous model parameters.

The proposed methods are implemented under `haplm/`. Two existing methods, HIPPO [[2]](#2) and AEML [[3]](#3), are presented with modifictions under `hippo_aeml/`; see `hippo_aeml/README.md` for details. The existing methods are written only for the setting where multinomial frequencies are shared, and the incomplete haplotypes are restricted to be information on each marker separately.

Comparisons between the proposed and existing methods, and further examples for the hierarchical case are provided under `test/`:

- `sim_study` Compare all methods with synthetic datasets over 3 markers (8 possible haplotypes). The pool size is varied across datasets to demonstrate how each method scales with pool size.
- `encode` Compare all methods except MCMC-Exact with 100 datasets simulated based on [human data from the 1000 Genomes Project](https://www.internationalgenome.org/data-portal/population/CEU). Each datasets covers 8 markers (256 possible haplotypes). Partition ligation is used to determine input haplotypes where appropriate. MCMC-Exact is excluded due to computational reasons.
- `time-series` Demonstrate a hierarchical use-case for the proposed methods. The synthetic dataset used consists of 30 time points of allele counts over 3 markers (8 haplotypes), where the haplotype frequencies vary over time. The haplotype frequencies are given a Gaussian process (GP) prior.
- `dhps` Application of MCMC-Exact to a real dataset of genetic data relevant to antimalarial drug resistance, collated by the [Worldwide Antimalarial Resistance Network](https://www.wwarn.org/tracking-resistance/sp-molecular-surveyor). 241 data points are used over 3 markers of the `Pfdhps` gene, where the incomplete haplotypes reported are not the same across data points.

There is also a comparison of an exact and an approximate (multinormal) likelihood for a toy example of the latent multinomial distribution under `mn_acc`, which demonstrates cases where the multinormal approximation breaks down.

This project is licensed under the terms of the GNU General Public License v3.0.

## Setup

The preferred method of installation is [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge), a more reliable clone of [Conda](https://docs.conda.io/en/latest/). See [here](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html#mamba) for a user guide to Mamba. 

Create a mamba environment and with the required dependencies:
```
mamba create -n [YOUR_ENV_NAME] 4ti2 atpbar numpyro pulp "pymc>=5" "sympy>=1.12"
mamba activate [YOUR_ENV_NAME]
```

To install the core funtionality as a module, clone the repository and run the following command from the root directory of this repository:
```
pip install .
```

To run the HIPPO and AEML C programs, you will need to install [GCC](https://gcc.gnu.org/) and [GSL](https://www.gnu.org/software/gsl/), and run the following commands from the `hippo_aeml/` directory:
```
gcc hippo.c -o hippo -lgsl -lgslcblas -lm
gcc AEML.c -o AEML -lgsl -lgslcblas -lm
```

Further dependencies are required to run the spatiotemporal example under `test/dhps/`:
```
mamba install cartopy xlrd
```

## Tips

Depending on your computational resources, you may want to adjust to number of MCMC chains and CPU cores used. In particular, there is an overhead for multithreaded operations used by PyMC, and [some people recommend setting the environment variables for the number of threads used by libraries such as MKL and OpenMP to 1](https://discourse.pymc.io/t/regarding-the-use-of-multiple-cores/4249). This can be done by editing the terminal configuration file, e.g. ~/.bashrc, or by adding the lines
```python
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
```
at the very top of the Python script being run.

Some methods run commands from [`4ti2`](https://4ti2.github.io/), which requires a directory for `4ti2` files to be stored. It is recommended to a create a directory `4ti2-files/` from the root directory of this repository.

More efficient integer programming solvers can speed up the pre-processing phase of the proposed methods. This can be configured by specifying a [`pulp`](https://coin-or.github.io/pulp/) solver when creating `LatentMult` objects.

Sampling with NUTS may sometimes take a long time (>5 minutes) to compile, which tends to occur for the multinormal approximation. If a long compilation time is observed, consider setting the `jaxify` argument (e.g. in `haplm.lm_inference.latent_mult_mcmc`) to `True`. This should speed up compilation at the cost of slower sampling.

## References

<a id="1">[1]</a> 
Foo, Y. S., Flegg, J. A. (2023). Haplotype inference from pooled genetic data with a latent multinomial model. To appear.

<a id="2">[2]</a> 
Pirinen, M. (2009). Estimating population haplotype frequencies from pooled SNP data using incomplete database information. Bioinformatics, 25(24), 3296–3302. https://doi.org/10.1093/bioinformatics/btp584

<a id="3">[3]</a> 
Kuk, A. Y. C., Zhang, H., & Yang, Y. (2009). Computationally feasible estimation of haplotype frequencies from pooled DNA with and without Hardy-Weinberg equilibrium. Bioinformatics, 25(3), 379–386. https://doi.org/10.1093/bioinformatics/btn623