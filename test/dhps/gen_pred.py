import numpy as np
import pymc as pm
import arviz as az
import scipy
import pickle as pkl
import collections
import sys

import pytensor.tensor as pt
from haplm.hap_util import mat_by_marker
from haplm.gp_util import GP, SphereGneiting

from PIL import Image

yr = int(sys.argv[1])
im = Image.open(f'../../data/dhps/Africa_pfpr_{yr}.tif') 
pfpr = np.asarray(im)
pfpr_masked = np.ma.masked_less(pfpr, 0)
    
# number of pixels to predict on per year
mask = pfpr_masked.mask
N_pred = mask.size - mask.sum()

RAD_TO_DEG = 180/np.pi
topleft = np.array([im.tag[33922][3], im.tag[33922][4]]) * RAD_TO_DEG
step = np.array([im.tag[33550][0], -im.tag[33550][1]]) * RAD_TO_DEG
# coordinates for points of prediction
yidxs, xidxs = np.where(np.logical_not(mask))
points = topleft + np.array([xidxs, yidxs]).T*step

Xnew = np.c_[points, yr*np.ones(N_pred), pfpr_masked.compressed()]

with open('../../test/dhps/data.pkl', 'rb') as fp:
    data = pkl.load(fp)
    ys = data['ys']
    ns = data['ns']
    amats = data['amats']
    infos = data['infos']
    X = data['X']
    lm_list = data['lm_list']
N = len(ys)
H = 8

idata = az.from_netcdf('../../data/dhps/exact.netcdf')

with pm.Model() as model:
    sigma = pm.InverseGamma('sigma', alpha=3, beta=1)
    alpha = pm.InverseGamma('alpha', alpha=3, beta=3, shape=H)
    ls_t = pm.InverseGamma('ls_t', alpha=3, beta=5, shape=H)
    ls_s = pm.InverseGamma('ls_s', alpha=3, beta=7, shape=H)    

    mean = pm.ZeroSumNormal('mu', sigma=2, shape=H)
    beta = pm.ZeroSumNormal('beta', sigma=2, shape=H)
    gps = [GP(cov_func=SphereGneiting(ls_s[i], ls_t[i]))
           for i in range(H)]
    gp_priors = [gp.prior("f"+str(i), X=X[:,:3]) for i, gp in enumerate(gps)]

with model:
    fs_pred = [gps[h].marg_cond(f'f_pred{h}', Xnew=Xnew[:,:3]) for h in range(H)]
    ps_pred = pm.Deterministic(f'p_pred', pm.math.softmax(pt.stack(fs_pred, axis=1)
                                               *alpha + mean + pt.outer(pt.as_tensor_variable(Xnew[:,3]), beta)
                                               + sigma*pm.Normal('noise_pred', shape=(N_pred,H)),
                                               axis=-1))
    pred_idata = pm.sample_posterior_predictive(idata, var_names=['p_pred'])

pred_samples = np.vstack(pred_idata.posterior_predictive.p_pred)
np.save(f'../../data/dhps/exact_{yr}_pred_samples.npy', pred_samples)

pred_samples = np.load(f'../../data/dhps/exact_{yr}_pred_samples.npy')

sumstats = {}
sumstats['mean'] = pred_samples.mean(axis=0)
sumstats['sd'] = pred_samples.std(axis=0)
sumstats['median'] = np.median(pred_samples, axis=0)
sumstats['mad'] = np.median(np.abs(pred_samples - sumstats['median'][None,:,:]), axis=0)
sumstats['quantiles'] = np.quantile(pred_samples, np.arange(0, 1.01, 0.05), axis=0)

amat = mat_by_marker(3)
geno_pred_samples = np.dot(pred_samples, amat.T)

sumstats = {}
sumstats['geno_mean'] = geno_pred_samples.mean(axis=0)
sumstats['geno_sd'] = geno_pred_samples.std(axis=0)
sumstats['geno_median'] = np.median(geno_pred_samples, axis=0)
sumstats['geno_mad'] = np.median(np.abs(geno_pred_samples - sumstats['geno_median'][None,:,:]), axis=0)
sumstats['geno_quantiles'] = np.quantile(geno_pred_samples, np.arange(0, 1.01, 0.05), axis=0)

with open(f'../../data/dhps/exact_{yr}_sumstats.pkl', 'wb') as fp:
    pkl.dump(sumstats, fp)