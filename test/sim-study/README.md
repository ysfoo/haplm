This example compares the proposed methods with HIPPO and AEML for data simulated based on synthetic data. ...

## Data files
how to read data ...

## Steps to reproduce results from the paper
1. Create the directory `data/sim-study/` from the root directory of this repository.
2. From this directory, run the following lines in Python to simulate data:
```python
from sim_data import gen_sim_data
for pool_size in range(20, 101, 20):
    for ds_idx in range(1, 6):
        fn_prefix = f'../../data/sim-study/psize{pool_size}_m{n_markers}_id{ds_idx}'
        gen_sim_data(n_pools=20, n_markers=3, pool_size=pool_size, 
                     alphas=0.4*np.ones(8), seed=ds_idx, fname=fn_prefix)
```
3. Perform inference with HIPPO, AEML, MCMC-Exact, MCMC-Approx, and LC-Sampling with the scripts `inference_hippo.py`, `inference_aeml.py`, `inference_exact.py`, `inference_mn_approx.py`, and `inference_cgibbs.py` respectively.
4. The results are plotted in `results.ipynb`.

Note that large pool sizes incur a much longer runtime for `inference_exact.py`.