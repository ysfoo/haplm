# Modifications to HIPPO and AEML

*Yong See Foo, July 2023*

HIPPO [[1]](#1) and AEML [[2]](#2) are haplotype frequency estimation methods based on pooled genetic data, with the original programs available [here](https://www.mv.helsinki.fi/home/mjxpirin/download.html). HIPPO is a Bayesian method that samples the list of input haplotypes as part of its MCMC procedure; AEML is a fast frequentist method that requires the list of input haplotypes to be specified. This README documents modifications implemented for better comparison with proposed alternatives for inferring haplotype frequencies from pooled genetic data.

## Remove assumption of diploid individuals
HIPPO and AEML are both written for analysing human haplotype data, where each individual carries two haplotypes. This assumption is removed, and so haplotypes of haploid organisms can also be handled. In this modified version, the first number in every line of the input `data_file` is no longer the number of individuals in a pool, but the number of haplotype samples in a pool.

## Add stabilising constant to covariance matrices
HIPPO and AEML both rely on a multinormal approximation, whose covariance matrix may be non-singular for certain values of haplotype frequencies. In this modified version, there is an option for a small stabilising constant to be added to the diagonal of the covariance matrices.

## AEML: Include estimates larger than 1e-3 in final output
The original version of AEML only outputs haplotype frequencies that are larger than 1e-3. In this modified version, frequency estimates of all input haplotypes are reported.

## HIPPO: Record trace and implement thinning
The original version of HIPPO only reports the posterior mean and variance as the chain statistics. In this modified version, there is an option to report the full trace to obtain other statistics (e.g. effective sample sizes) downstream. The chain produced by HIPPO may have very high autocorrelation, so there is an option for thinning to reduce the number of samples recorded by the trace.

## HIPPO: Record post-warmup sampling time
This modified version of HIPPO outputs the sampling time excluding the warmup phase to `stdout`.

## HIPPO: Output maximimum likelihood attained
This modified version of HIPPO outputs the maximum likelihood density as the first line of `MAP.out` before outputting the maximum likelihood estimate.

## References

<a id="1">[1]</a> 
Pirinen, M. (2009). Estimating population haplotype frequencies from pooled SNP data using incomplete database information. Bioinformatics, 25(24), 3296–3302. https://doi.org/10.1093/bioinformatics/btp584

<a id="2">[2]</a> 
Kuk, A. Y. C., Zhang, H., & Yang, Y. (2009). Computationally feasible estimation of haplotype frequencies from pooled DNA with and without Hardy-Weinberg equilibrium. Bioinformatics, 25(3), 379–386. https://doi.org/10.1093/bioinformatics/btn623

