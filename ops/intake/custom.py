import re
import numpy as np
import pandas as pd
import ops.filenames
from ops.constants import *

def parse_czi_export(f):
    pat = '.*_s(\d+)c(\d+)m(\d+)_ORG.tif'
    scene, channel, m = re.findall(pat, f)[0]
    return {WELL: int(scene), CHANNEL: int(channel), SITE: int(m) - 1}

def make_czi_file_table(files, wells):
    """Provide list of wells to map from czi scenes to well names.
    """
    wells = np.array(wells)
    df1 = pd.DataFrame([parse_czi_export(f) for f in files])
    df2 = (pd.DataFrame([ops.filenames.parse_filename(f) for f in files])
            .drop(WELL, axis=1))

    return (pd.concat([df1, df2], axis=1)
     .sort_values([CYCLE, WELL, SITE, CHANNEL])
     .assign(subdir=lambda x: 
             x.apply(lambda row: 'input/{mag}_{cycle}'.format(**row), axis=1))           
     .assign(well=lambda x: wells[x[WELL] - 1])
     .assign(new_file=lambda x: 
            x.apply(lambda row: 
                ops.filenames.name_file({}, tag='sbs', **row), axis=1))
    )

def load_czi_export_info(f_xml):
    from xmljson import badgerfish as bf
    from xml.etree.ElementTree import fromstring

    columns = {'startX': GLOBAL_X, 'startY': GLOBAL_Y}

    with open(f_xml) as fh:
        txt = fh.read()
    images = bf.data(fromstring(txt))['ExportDocument']['Image']

    arr = []
    for img in images:
        rename = lambda x: x[1].lower() + x[2:]
        info = {rename(k): v for k, v in img['Bounds'].items()}
        info['file'] = img['Filename']['$']
        info['t_ms'] = img['T']['$'] * 1000
        arr.append(info)
    return pd.DataFrame(arr).rename(columns=columns)

# alignment with noise

from ops.process import Align

def mask_noise(data):
    d = data.copy()
    mask = d > 0

    mu = d[mask].mean()
    sig = d[mask].std()

    noise = np.random.randn((~mask).sum())

    d[~mask] = (noise * sig) + mu
    return d

def align(raw1, raw2):
    d1 = Align.normalize_by_percentile(raw1)
    d2 = Align.normalize_by_percentile(raw2)
    
    d1, d2 = pile([d1, d2])
    d1_ = mask_noise(d1)
    d2_ = mask_noise(d2)
    
    offset, error , _ = skimage.feature.register_translation(d1_, d2_)

    data = pile([raw1, raw2])
    result = Align.apply_offsets(data, [offset, [0, 0], ])
#     noised = np.array([d1_, d2_])
    return result, offset
    
def score_aligned_pair(result):
    mask = (result > 0).all(axis=0)
    pairs = result[:, mask]
    score = np.corrcoef(pairs)[0, 1]
    return score 


# UPDATE ALIGNMENT

def custom_split(img, n=32, pad=10):
    i_max, j_max = img.shape
    h = int(np.ceil(i_max/n))
    w = int(np.ceil(j_max/n))
#     print(h, w)
    rows, cols = int(i_max/h), int(j_max/w)
    tiles = []
    for i, j in product(range(rows), range(cols)):
        i0 = max(0, h*i - pad)
        j0 = max(0, w*j - pad)
        i1 = min(i_max, h*(i + 1) + pad)
        j1 = min(j_max, w*(j + 1) + pad)
        tiles += [img[i0:i1, j0:j1]]
    return tiles

def tiled_align(ref_0, ref_1, data, n=32):
    """ref_0, ref_1 are (I, J); data is (C, I, J)
    """
    aligned = np.zeros_like(data)
    scores = np.zeros(data[0].shape)
    # should include score from offset calculation and only overwrite lower scores
    ix_j, ix_i = np.meshgrid(range(ref_0.shape[1]), 
                               range(ref_0.shape[0]))
    
    kwargs = dict(n=n)
    tiles_j = custom_split(ix_j, **kwargs)
    tiles_i = custom_split(ix_i, **kwargs)
    
    tiles_0 = custom_split(ref_0, **kwargs)
    tiles_1 = custom_split(ref_1, **kwargs)
    
    tiles_data = list(zip(*[custom_split(x, **kwargs) for x in data]))
    
    for t0, t1, ti, tj, td in zip(tiles_0, tiles_1, tiles_i, tiles_j, tiles_data):
        # calculate offset
        _, offset = ops.process.Align.calculate_offsets(np.array([t0, t1]), 1)
        # determine valid pixels
        fwd_offset = lambda t: ops.process.Align.apply_offsets(np.array([t]), 
                                                                 [offset])[0]
        rev_offset = lambda t: ops.process.Align.apply_offsets(np.array([t]), 
                                                                 [-offset])[0]
        ti_, tj_, t1_ = [rev_offset(x) for x in (ti, tj, t1)]
        mask = t1_ > 0
        # coordinates in aligned
        i, j = ti_[mask], tj_[mask]
        for c, t in enumerate(td):
            # shift data (wasteful)
            t_ = fwd_offset(t)
            aligned[[c]*len(i), i, j] = t[mask]
        
    return aligned

def update_alignment(data):
    ref_0 = np.mean(data[0], axis=0)
    dapi_0 = data[0, 0]
    aligned = data.copy()
    for c in range(1, len(data)):
        dapi_1 = data[c, 0]
        data_  = data[c, :]
        ref_1 = np.mean(data[c], axis=0)

        aligned[c] = tiled_align(ref_0, ref_1, data_, n=10)
        
    return aligned