'''
Runs MCMC inference for latent multinomial model applied to the full WWARN molecular marker dataset.
'''


import pandas as pd
import numpy as np
import scipy
import pickle as pkl
import collections
import pycountry_convert
from PIL import Image
from time import time
from tqdm import tqdm

import pymc as pm
import pymc.sampling_jax
import jax.numpy as jnp
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify
from haplm.gp_util import GP, SphereGneiting
import numpyro

# used for integer programming, checking y = Az has solution for z
import pulp
from haplm.lm_dist import LatentMult, find_4ti2_prefix
# from haplm.numpyro_util import sample_numpyro_nuts


MARKER_DATA_FNAME = "dhfr_dhps_surveyor_data.xls"
get_pfpr_fname = lambda yr: f'../../data/dhps/Africa_pfpr_{yr}.tif'

# uncomment this to download latest version of WWARN data
# import urllib.request
# urllib.request.urlretrieve("http://www.wwarn.org/dhfr-dhps-surveyor/dhfrdhpsSurveyor/database/dhfr_dhps_surveyor_data.xls",
#                            MARKER_DATA_FNAME)

draws = 500
tune = 500
chains = 5
numpyro.set_host_device_count(chains)

prefix_4ti2 = find_4ti2_prefix()
solver = pulp.apis.COIN_CMD(msg=False) # change to more efficient solver if available

# read marker data
dhfr_dhps = pd.read_excel(MARKER_DATA_FNAME)
dhfr_dhps.drop(columns=['Mixed present', 'authors', 'publication year', 'publication Url', 'title',
                        'notes', 'pubMedId', 'percentage', 'included or excluded', 'marker group'],
               inplace=True)

# get continents
country_to_cont = {}
print('Need to add the following countries to country_to_cont manually:')
for c in set(dhfr_dhps['country']):
    try:
        code = pycountry_convert.country_name_to_country_alpha2(c)
        country_to_cont[c] = pycountry_convert.country_alpha2_to_continent_code(code)
    except KeyError:
        print(c)
country_to_cont['Sénégal'] = 'AF'

# keep African countries
dhfr_dhps = dhfr_dhps[dhfr_dhps['country'].map(country_to_cont) == 'AF']
# remove Libya
dhfr_dhps = dhfr_dhps[dhfr_dhps['country'] != 'Libya']
# drop entries where site is unknown
dhfr_dhps = dhfr_dhps[dhfr_dhps['site'].notna()]
# remove earlier than 2000
dhfr_dhps = dhfr_dhps[dhfr_dhps['study end year'] >= 2000]
# drop entries where end year - start year > 3
dhfr_dhps = dhfr_dhps[dhfr_dhps['study end year'] - dhfr_dhps['study start year'] <= 3]
# dhps haplotypes only
dhps = dhfr_dhps[~dhfr_dhps['mutation'].str.contains('dhfr')]

# rows with the same study id, substudy number, site are the same pool
pools = dhps.groupby(['study id', 'substudy number', 'site'])
print('Number of rows after filtering:', len(dhps))
print('Number of pools after filtering:', len(pools))

pfpr_dict = {}
for yr in range(2000, 2021):
    im = Image.open(get_pfpr_fname(yr))
    pfpr = np.asarray(im)
    pfpr_masked = np.ma.masked_less(pfpr, 0)
    pfpr_dict[yr] =  pfpr_masked

# number of pixels to predict on per year
mask = pfpr_masked.mask
N_pred = mask.size - mask.sum()

RAD_TO_DEG = 180/np.pi
topleft = np.array([im.tag[33922][3], im.tag[33922][4]]) * RAD_TO_DEG
step = np.array([im.tag[33550][0], -im.tag[33550][1]]) * RAD_TO_DEG
# coordinates for points of prediction
yidxs, xidxs = np.where(np.logical_not(mask))
points = topleft + np.array([xidxs, yidxs]).T*step

t = time()

# tree structure for finding closest pixel
tree = scipy.spatial.KDTree(points, leafsize=100)

closest_dict = {}
diff_dict = {}
for lon, lat in zip(dhps['lon'], dhps['lat']):
    if (lon, lat) in closest_dict:
        continue
    closest_dict[(lon, lat)] = tree.query([lon, lat], k=1)[1]
    lon_close, lat_close = points[closest_dict[(lon, lat)]]
    diff_dict[(lon, lat)] = [lon-lon_close, lat-lat_close]

# convert mutation strings to ternary format
# 1 for mutation present, 0 for mutation absent, -1 if N/A
# notice that there are mixed mutations e.g. 540K/E
# but we consider mixed as mutated
mut_dict = {}
mut_pos = ['437', '540', '581']
n_mut = len(mut_pos)
print('Mutations recorded:')
for m in set(dhps['mutation']):   
    code = [-1]*n_mut
    exclude = False
    for token in m[5:].split('-'):
        pos = ''.join([d for d in token if d.isdigit()])
        if pos not in mut_pos:
            exclude = True
            break
        code[mut_pos.index(pos)] = 1 if token[0].isdigit() else 0
    if exclude:
        continue
    print(m)
    mut_dict[m] = tuple(code)


# convert ternary array to binary array (rows of A)
def t2b(tarr):
    G = len(tarr)
    H = 1 << G
    pos = sum(1 << j for j in range(G) if tarr[j] == 1)
    neg = sum(1 << j for j in range(G) if tarr[j] == 0)
    barr = np.zeros(H,int)
    for i in range(H):
        if (pos & i == pos) and not (neg & i):
            barr[i] = 1
    return barr




fail = 0
ys = []
ns = []
codes = []
amats = []
infos = []
X = []

for iden, pool in pools:
    pool = pool[pool['mutation'].isin(mut_dict)]
    for cname in pool.columns:
        if cname in ['tested', 'present', 'mutation']:
            continue
        assert len(pool[cname].unique()) == 1 # ensure all non-mutation entries are the same
    assert len(pool['mutation'].unique()) == len(pool) # ensure no double entries
    
    n = pool['tested'].max() # number of haplotypes in pool
    # boolean tuple for whether this pool has information about each marker
    has_info = tuple(any(mut_dict[mutation][i] != -1 for mutation in pool['mutation']) for i in range(n_mut))    
    
    # configuration matrices are constructed based on collapsed haplotypes
    code_to_prop = {}
    for tested, present, mutation in zip(pool['tested'],pool['present'],pool['mutation']):
        code = tuple(m for m, b in zip(mut_dict[mutation], has_info) if b)
        code_to_prop[code] = code_to_prop.get(code, 0) + present/tested # sum to include mixed infections
    y = []
    amat = []
    codevec = []
    for code, prop in code_to_prop.items():
        y.append(int(round(prop*n)))
        amat.append(t2b(code))
        codevec.append(code)
    y = np.array(y)
    amat = np.array(amat)
    
    # checks if at least one solution to y = Az exists
    H = 1 << sum(has_info)
    prob = pulp.LpProblem("test", pulp.LpMinimize)
    z = pulp.LpVariable.dicts("z", range(H), lowBound=0, cat='Integer')
    prob += pulp.lpSum(z) == n
    for j in range(len(y)):
        prob += (pulp.lpSum([amat[j,k]*z[k] for k in range(H)]) == y[j])
    prob.solve(solver)  
    if prob.status == 1:
        ys.append(y)
        ns.append(n)
        codes.append(codevec)
        amats.append(amat)
        infos.append(has_info)
        row = pool.iloc[0,:]
        lon, lat, start, end = row[['lon','lat','study start year','study end year']]
        pos = closest_dict[(lon, lat)]
        pfpr_val = np.mean([pfpr_dict[yr][yidxs[pos],xidxs[pos]]
                            for yr in range(int(start), int(end)+1)])
        X.append([lon, lat, (start+end)/2, pfpr_val])
    else:
        fail += 1
N = len(ys)
print(fail, 'pools have inconsistent counts')
print(f'Use {N} pools for inference')


X = np.array(X)
lm_list = []
N = len(ys)
for i in tqdm(range(N)):
    lm = LatentMult(amats[i], ys[i], ns[i], '../../4ti2-files/dhps-exact',
                    solver, prefix_4ti2, enum_sols=True)
    lm_list.append(lm)
    nsol = ('failed' if lm.sols_var is None else len(lm.sols)) if lm.idx_var.size else 1

pre_time = time() - t

with open('processed.pkl', 'wb') as fp:
    pkl.dump({
              'ys': ys,
              'ns': ns,
              'amats': amats,
              'codes': codes,
              'infos': infos,
              'X': X,
              'lm_list': lm_list,
             }, fp)


# dict that maps boolean tuple of whether pool has info about each marker
# to list of pool indices
# if pool has info on all markers, the index is separately stored in idxs_all_haps
idxs_by_info = collections.defaultdict(list)
idxs_all_haps = []
for idx, info in enumerate(infos):
    if info == (True, True, True):
        idxs_all_haps.append(idx)
    else:
        idxs_by_info[info].append(idx)
N = len(ys)
H = 8
G = 3


# return list of list of full haplotypes for each collapsed haplotype,
# given boolean tuple of whether there is info about each marker
# collapsed haplotypes omit markers for which there is no info
def cols_given_info(has_info):
    G = len(has_info)
    H = 1 << G
    G_sub = sum(has_info)
    H_sub = 1 << G_sub
    cols = [[] for _ in range(H_sub)]
    
    # marker indices that have info
    test_gs = [g for g in range(G) if has_info[g]]
    # for each full haplotype, find the corresponding collapsed haplotype
    for h in range(H):
        cols[(sum(((h >> g) & 1) << g_sub for g_sub, g in enumerate(test_gs)))].append(h)
    return cols

idxs_allhaps = [i for i, info in enumerate(infos) if info == (True, True, True)]
infos_excl_allhaps, idxs_excl_allhaps = zip(*[(info, i) for i, info in enumerate(infos) if info != (True, True, True)])

pmat_dict = {}
for info in set(infos):
    cols_list = cols_given_info(info)
    pmat = np.zeros((len(cols_list), H), int)
    for i, cols in enumerate(cols_list):
        pmat[i, cols] = 1
    pmat_dict[info] = jnp.array(pmat)

logfacts = scipy.special.gammaln(np.arange(1, max(lm.n_var for lm in lm_list)+2))


# same approach as haplm.lm_inference.hier_latent_mult_mcmc with jaxify=True and methods=['exact']*N
def loglike_fn(p):    
    return (
            # deal with pools where there is info about all markers
            sum(lm_list[i].loglike_exact_jax(p[i], logfacts) for i in idxs_allhaps) +
            # deal with other pools, as multinomial probabilities should be summed for each collapsed haplotype
            sum(lm_list[i].loglike_exact_jax(jnp.dot(pmat_dict[info], p[i]), logfacts)
                for i, info in zip(idxs_excl_allhaps, infos_excl_allhaps)))

class LoglikeOp(Op):
    default_output = 0

    def make_node(self, p,):
        inputs = [p]
        outputs = [pt.dscalar()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        outputs[0][0] = np.asarray(loglike_fn(inputs[0]), dtype=node.outputs[0].dtype)

loglike_op = LoglikeOp()

@jax_funcify.register(LoglikeOp)
def loglike_dispatch(op, **kwargs):
    return loglike_fn


with pm.Model() as model:
    sigma = pm.InverseGamma('sigma', alpha=3, beta=1)
    alpha = pm.InverseGamma('alpha', alpha=3, beta=3, shape=H)
    ls_t = pm.InverseGamma('ls_t', alpha=3, beta=5, shape=H)
    ls_s = pm.InverseGamma('ls_s', alpha=3, beta=7, shape=H)    

    mean = pm.ZeroSumNormal('mu', sigma=2, shape=H)
    beta = pm.ZeroSumNormal('beta', sigma=2, shape=H)
    gps = [GP(cov_func=SphereGneiting(ls_s[h], ls_t[h]))
           for h in range(H)]
    ps = pm.Deterministic('p', pm.math.softmax(pt.stack([gp.prior(f'f{h}', X=X[:,:3])
                                                         for h, gp in enumerate(gps)], axis=1)
                                               *alpha + mean + pt.outer(pt.as_tensor_variable(X[:,3]), beta)
                                               + sigma*pm.Normal('noise', shape=(N,H)),
                                               axis=-1))
    loglike = pm.Potential('loglike', loglike_op(ps))

# run MCMC
t = time()
with model:
    idata = pm.sampling_jax.sample_numpyro_nuts(draws=draws, tune=tune, chains=chains, target_accept=0.9,
                                                random_seed=2023, postprocessing_chunks=25)
mcmc_time = time() - t

idata.sample_stats.attrs['preprocess_time'] = pre_time
idata.sample_stats.attrs['mcmc_walltime'] = mcmc_time

# save trace
idata.posterior = idata.posterior.drop_vars('noise')
idata.to_netcdf(f'../../data/dhps/exact.netcdf')
