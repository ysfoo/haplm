## Spatiotemporal mapping of *Pfdhps* haplotype frequencies in sub-Saharan Africa 

The code in this directory performs Bayesian inference on the haplotype frequencies of *Pfdhps* haplotypes at positions 437/540/581, see [[1]](#1). Haplotype frequencies are modelled as a softmax transformation of independent Gaussian processes. The data is available at the Worldwide Antimalarial Resistance Network SP Molecular Surveyor [website](https://www.wwarn.org/dhfr-dhps-surveyor). We also use incidence rates of *Plasmodium falciparum* as covariates, obtained from the Malaria Atlas Project at a resolution of $0.2^\circ\times 0.2^\circ$. Local copies of this data may be available upon request.

## Steps to reproduce results from the paper
1. Create the directory `data/dhps/` from the root directory of this repository.
2. Run the commands `pip install pycountry_convert` and `mamba install tqdm cartopy pandas xlrd` to install helper functions for data preprocessing.
3. Run the script `inference.py` for data preprocessing and MCMC inference.
4. Run the script `gen_pred.py` with the year as a command line argument to compute the posterior distribution of haplotype frequencies for that year. A large file is created to store the posterior distribution produced, so the script is to be run multiple times (by year) to reduce peak RAM usage.
5. The results are plotted in `results.ipynb`.
6. To perform model checking, run 10-fold validation with `python validation.py 10 <fold>`, where `<fold>` takes on the values 1, 2, ..., 10. The results are plotted in `cv_results.ipynb`.

## List of publications that reported inconsistent counts

We find that 15 pools had observed counts that could not have been produced by any latent count vector, indicating that the data point is errorneous. The corresponding publications are:
- http://www.ncbi.nlm.nih.gov/pubmed/?term=26437774
- http://www.ncbi.nlm.nih.gov/pubmed/?term=28381273
- https://pubmed.ncbi.nlm.nih.gov/31932374/
- http://www.ncbi.nlm.nih.gov/pubmed/?term=24055717
- http://www.ncbi.nlm.nih.gov/pubmed/?term=27209063
- http://www.ncbi.nlm.nih.gov/pubmed/?term=27647575
- https://www.ncbi.nlm.nih.gov/pubmed/?term=30897090
- https://pubmed.ncbi.nlm.nih.gov/31438951/
- https://www.ncbi.nlm.nih.gov/pubmed/?term=22540158

## References

<a id="1">[1]</a> 
Foo, Y. S., Flegg, J. A. (2023). A spatiotemporal model of multi-marker antimalarial resistance. medRxiv preprint medRxiv:2023.10.03.23296508. [https://doi.org/10.1101/2023.10.03.23296508](https://doi.org/10.1101/2023.10.03.23296508)
