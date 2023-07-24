import numpy as np

import pickle as pkl
import traceback

from haplm.hap_util import mat_by_marker, str_to_num, num_to_str
from haplm.hippo_aeml import run_AEML
from sim_data import parse_sim_data


# location of AEML program
aeml_dir = '../../hippo_aeml/'

# data parameters
n_datasets = 100
n_pools = 20
n_markers = 8
pool_size = 50

trials = 10 # number of random initialisations for AEML

hap_lists = []
with open('../../data/encode/PL.txt') as fp:
    for line in fp:
        hap_lists.append([int(x) for x in line.split()])

failed = [] # 1-based indices of datasets for which AEML failed
for ds_idx in range(1, n_datasets+1):
    try:
        hap_fn = 'encode_haplist.txt'
        with open(hap_fn, 'w') as fp:
            hap_list = [num_to_str(hnum, n_markers) for hnum in hap_lists[ds_idx-1]]
            fp.write(f'{len(hap_list)}\n')
            for hstr in hap_list:
                fp.write(' '.join(list(hstr)))
                fp.write('\n')

        fn_prefix = f'../../data/encode/psize{pool_size}_m{n_markers}_id{ds_idx}'
        ns, ys = parse_sim_data(fn_prefix+'.data')

        print(f'AEML for set {ds_idx}: ', end='')
        aeml_out = run_AEML(ns, ys, aeml_dir, trials=trials, stab=1e-9, hap_fn=hap_fn)
        if aeml_out is None:
            failed.append(ds_idx)
            with open('AEML.log') as fp:
                for line in fp:
                    continue            

        with open(f'{fn_prefix}_aeml.pkl', 'wb') as fp:
            pkl.dump(aeml_out, fp)

    except Exception as e:
        print(traceback.format_exc())
        failed.append(ds_idx)

print(f'Datasets where AEML failed: {failed}')


