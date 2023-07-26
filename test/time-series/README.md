This example demonstrates how the proposed methods can be applied to a hierarchical model. Here, we model time-varying haplotype frequencies as a softmax transformation of independent Gaussian processes.

## Steps to reproduce results from the paper
1. Create the directory `data/time-series/` from the root directory of this repository.
2. Run the notebook `sim_data.ipynb` to simulate the data.
3. Perform inference with MCMC-Exact, MCMC-Approx, and LC-Sampling with the scripts `inference_exact.py`, `inference_mn_approx.py`, and `inference_cgibbs.py` respectively.
4. The results are plotted in `results.ipynb`.