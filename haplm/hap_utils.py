import numpy as np
import aesara
import pulp
from haplm.hippo_aeml import run_AEML

solver = pulp.apis.SCIP_CMD(msg=False)

def encode_amat(amat):
    """
    Encodes a binary matrix into a tuple of integers.
    
    Parameters
    ----------
    amat : 2D-array
        Matrix to encode.
        
    Returns
    -------
    tuple
        A tuple consisting of the number of rows of `amat`, and each column of `amat` interpreted as a binary number.
    """
    r, _ = amat.shape
    return tuple([r] + list(np.dot(1<<np.arange(r)[::-1], amat)))


def decode_amat(code):
    """
    Decodes a tuple as the result of a binary matrix encoded by `encode_amat`.
    
    Parameters
    ----------
    code : tuple
        Output of `encode_amat`.
        
    Returns
    -------
    2D-array
        Decoded binary matrix.
    """
    return np.transpose([np.fromstring(' '.join(np.binary_repr(num).zfill(code[0])), sep=' ', dtype=int)
                         for num in code[1:]]).astype(aesara.config.floatX)

def mat_by_marker(G):
    """
    Constructs the configuration matrix for the case where observed counts are allele counts for each marker.

    Parameters
    ----------
    G : int
        Number of markers.

    Returns
    -------
    2D-array
        Configuration matrix for latent multinomial vector.
    """
    return np.array([[(h >> i) & 1 for h in range(2**G)] for i in range(G)], int)

def str_to_num(hstr):
    n_markers = len(hstr)
    return sum(1 << i for i, c in enumerate(hstr) if c == '1')

def num_to_str(hnum, n_markers):
    return ''.join(['1' if (hnum >> i) & 1 else '0' for i in range(n_markers)])

def PL(ns, ys, n_markers, aeml_dir, trials=5):
    if n_markers < 6:
        print('only works for n_markers >= 6')
        
    n_3m = 4 - n_markers % 4 if n_markers % 4 else 0
    n_4m = (n_markers - 3*n_3m) // 4
    blocks = [(4*x, 4*x+4) for x in range(n_4m)] + [(4*n_4m + 3*x, 4*n_4m + 3*x + 3) for x in range(n_3m)]
    assert blocks[-1][-1] == n_markers
    
    hap_lists = [mat_by_marker(block[1]-block[0]).T for block in blocks]
    
    while len(blocks) > 1: 
        halfn = len(blocks)//2
        new_lists = [PL_pair(ns, ys, blocks[2*x], blocks[2*x+1],
                             hap_lists[2*x], hap_lists[2*x+1], aeml_dir, trials)
                     for x in range(halfn)]
        if any(haps is None for haps in new_lists):
            return None
        hap_lists = new_lists + ([hap_lists[-1]] if len(blocks) % 2 else [])
        blocks = [(blocks[2*x][0], blocks[2*x+1][1])
                  for x in range(halfn)] + ([blocks[-1]] if len(blocks) % 2 else [])        
        assert len(blocks) == len(hap_lists)        
        
    assert blocks[0][0] == 0 and blocks[0][-1] == n_markers
    return hap_lists[0]
    
    
def PL_pair(ns, ys, block1, block2, hap_list1, hap_list2, aeml_dir, trials=5):
    assert block2[0] == block1[1]
    hap_fn = 'encode_haplist.txt'
    
    print(f'AEML for markers {block1[0]+1}-{block1[1]}: ', end='')    
    with open(hap_fn, 'w') as fp:
        fp.write(f'{len(hap_list1)}\n')
        for harr in hap_list1:
            fp.write(' '.join([str(h) for h in harr]))
            fp.write('\n')
    aeml_1 = run_AEML(ns, [y[block1[0]:block1[1]] for y in ys], aeml_dir,
                      trials=trials, stab=1e-9, hap_fn=hap_fn)
    # for h, p in zip(hap_list1, aeml_1['pest']):
    #     print(h, p)
    
    print(f'AEML for markers {block2[0]+1}-{block2[1]}: ', end='')
    with open(hap_fn, 'w') as fp:
        fp.write(f'{len(hap_list2)}\n')
        for harr in hap_list2:
            fp.write(' '.join([str(h) for h in harr]))
            fp.write('\n')
    aeml_2 = run_AEML(ns, [y[block2[0]:block2[1]] for y in ys], aeml_dir,
                      trials=trials, stab=1e-9, hap_fn=hap_fn)
    
    if aeml_1 is None or aeml_2 is None:
        return None
    
    # note 1s encode for minor alleles
    n_markers = block2[1] - block1[0]
    atmost_1mut = np.vstack([np.zeros((1, n_markers), dtype=int), np.eye(n_markers, dtype=int)])
    
    thres = 0.01
    remaining = sorted({p for p in list(aeml_1['pest']) + list(aeml_2['pest']) if p < thres})
    while remaining:
        haps = concat_haps(aeml_1['pest'], aeml_2['pest'], hap_list1, hap_list2,
                           thres=thres, inithaps=atmost_1mut)
        for n, y in zip(ns, ys):
            amat = np.vstack([np.ones(len(haps), int), np.array(haps).T]).astype(int)
            y = np.array([n] + list(y[block1[0]:block2[1]]))
            nzs = np.arange(len(haps))
            # check if y = Az has a solution
            prob = pulp.LpProblem("test", pulp.LpMinimize)
            z = pulp.LpVariable.dicts("z", nzs, lowBound=0, cat='Integer')
            for j in range(amat.shape[0]):
                prob += (pulp.lpSum([amat[j,k]*z[k] for k in nzs]) == y[j])
            prob.solve(solver)  
            if prob.status != 1:
                # no solution
                assert prob.status == -1
                break
        else:
            # done, break out of while loop
            break
        thres = remaining.pop()
        print(f'decrease threshold to {thres}')
    else:
        # failed
        print(f'failed to combine {len(hap_list1)} haplotypes from {block1[0]+1}-{block1[1]} and {len(hap_list2)} haplotypes from {block2[0]+1}-{block2[1]}')
        print(y)
        for row in amat:
            print(' '.join([str(x) for x in row]))
        return None

    return haps

def select_haps(pest, haps, thres):    
    select = set(np.where(np.array(pest) >= thres)[0])
    # n_markers = len(haps[0])
    # encountered = set()
    # for i in np.argsort(-pest):
    #     select.add(i)
    #     for pos, digit in enumerate(haps[i]):
    #         encountered.add((-1 if digit == 0 else 1)*(pos + 1))
    #     if len(encountered) == 2*n_markers:
    #         break
    # print(encountered)
    # for i in select:
    #     print(haps[i])
    return select

def concat_haps(pest1, pest2, haps1, haps2, thres, maxhaps=None, inithaps=[]):
    totlen = len(haps1[0]) + len(haps2[0])
    select1 = select_haps(pest1, haps1, thres)
    select2 = select_haps(pest2, haps2, thres)

    pair_dict = {(idx1, idx2): min(pest1[idx1], pest2[idx2])
                 for idx1 in select1 for idx2 in select2}
    hap_pairs = sorted(pair_dict, key=lambda x: -pair_dict[x])

    inithaps_set = {tuple(hap) for hap in inithaps}
    hap_list = [list(hap) for hap in inithaps]
    to_add = maxhaps - len(inithaps) if maxhaps is not None else len(hap_pairs)

    for idx1, idx2 in hap_pairs:
        if to_add <= 0:
            break
        
        cand = list(haps1[idx1]) + list(haps2[idx2])
        if tuple(cand) not in inithaps_set:
            hap_list.append(cand)
            to_add -= 1

    return hap_list

