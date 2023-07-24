# HapLM: Bayesian inference for pooled <ins>hap</ins>lotype data using <ins>l</ins>atent <ins>m</ins>ultinomials

This module implements MCMC methods for latent multinomial distributions, see [[1]](#1) for details. The key application is for pooled genetic data, where counts of incomplete haplotypes are reported for separate pools. Each incomplete haplotype corresponds to a subset of full haplotypes, and the number of full haplotypes per pool are multinomially distributed. The observation of partial sums of these multinomial counts leads to a latent multinomial model. There are 3 proposed methods, all of which can be used either under a setting where the multinomial frequencies are shared across all pools, or under a hierarchical setting where the multinomial frequencies are given a hierarchical prior. According to the method names given in [[1]](#1), the proposed methods are:

- *MCMC-Exact*. All possible latent counts that are compatible with observed data ...
- *MCMC-Approx*. ...
- *LC-Sampling*. ...

The proposed methods are implemented under `haplm/`. Two existing methods, HIPPO [[2]](#2) and AEML [[3]](#3), are presented with modifictions under `hippo_aeml/`; see `hippo_aeml/README.md` for details. The existing methods are written only for the setting where multinomial frequencies are shared, and the incomplete haplotypes are restricted to be information on each marker separately.

Comparisons between the proposed and existing methods, and further examples for the hierarchical case are provided under `test/`:

- `sim_study` ...
- `encode` ...
- `time-series` ...
- `dhps` ...

There is also a comparison of an exact and an approximate (multinormal) likelihood for a toy example of the latent multinomial distribution under `mn_acc`, which demonstrates cases where the multinormal approximation breaks down.

This project is licensed under the terms of the GNU General Public License v3.0.

## Installation

Create a [conda](https://docs.conda.io/en/latest/) environment and with the required dependencies:
```
conda create -c conda-forge -n [YOUR_ENV_NAME] atpbar numpyro pulp "pymc>=5" "sympy>=1.12"
conda activate [YOUR_ENV_NAME]
```

To install the core funtionality as a module, run the following command from the root directory of this repository:
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
conda install -c conda-forge cartopy xlrd
```

## References

<a id="1">[1]</a> 
Foo, Y. S., Flegg, J. A. (2023). Haplotype inference from pooled genetic data with a latent multinomial model. To appear.

<a id="2">[2]</a> 
Pirinen, M. (2009). Estimating population haplotype frequencies from pooled SNP data using incomplete database information. Bioinformatics, 25(24), 3296–3302. https://doi.org/10.1093/bioinformatics/btp584

<a id="3">[3]</a> 
Kuk, A. Y. C., Zhang, H., & Yang, Y. (2009). Computationally feasible estimation of haplotype frequencies from pooled DNA with and without Hardy-Weinberg equilibrium. Bioinformatics, 25(3), 379–386. https://doi.org/10.1093/bioinformatics/btn623