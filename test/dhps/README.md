## Spatiotemporal mapping of *Pfdhps* haplotype frequencies in sub-Saharan Africa 

The code in this directory performs Bayesian inference on the haplotype frequencies of *Pfdhps* haplotypes at positions 437/540/581. Haplotype frequencies are modelled as a softmax transformation of independent Gaussian processes. The data is available at the Worldwide Antimalarial Resistance Network SP Molecular Surveyor [website](https://www.wwarn.org/dhfr-dhps-surveyor).

## Steps to reproduce results from the paper
1. Create the directory `data/dhps/` from the root directory of this repository.
2. Run the commands `pip install pycountry_convert` and `mamba install tqdm cartopy pandas xlrd` to install helper functions for data preprocessing.
3. Run the script `inference.py` for data preprocessing and MCMC inference.
4. Run the script `gen_pred.py` with the year as a command line argument to compute the posterior distribution of haplotype frequencies for that year. A large file is created to store the posterior distribution produced, so the script is to be run multiple times (by year) to reduce peak RAM usage.
5. The results are plotted in `results.ipynb`.