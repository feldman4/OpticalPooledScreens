import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import mode
from pims import ND2_Reader
from itertools import product

from ops.constants import *

ND2_EXPORT_FILE_PATTERN = ('.*'
        r'Well(?P<well_ix>\d+)_.*'
        r'Seq(?P<seq>\d+).*?'
        r'(?P<mag>\d+X).'    
        r'(?:(?P<cycle>[^_\.]*)_)?.*'
        r'(?P<m>\d{4})'                   
        r'(?:\.(?P<tag>.*))*\.(?P<ext>.*)')

ND2_EXPORT_FILE_PATTERN_96 = ('.*'
        r'Well(?P<well>.\d\d)_.*'
        r'Seq(?P<seq>\d+).*?'
        r'(?P<mag>\d+X).'    
        r'(?:(?P<cycle>[^_\.]*)_)?.*'
        r'(?P<m>\d{4})'                   
        r'(?:\.(?P<tag>.*))*\.(?P<ext>.*)')

def add_neighbors(df_info, num_neighbors=9, radius_leniency=10):
    xy = ['x_um', 'y_um']
    xy = [GLOBAL_X, GLOBAL_Y]
    site = SITE
    
    df = df_info.drop_duplicates(xy)
    kdt = cKDTree(df[xy].values)

    distance, index = kdt.query(df[xy].values, k=num_neighbors)

    # convert to m
    index = np.array(df[site])[index]

    m = mode(distance.max(axis=1).astype(int)).mode[0]
    filt = (distance < (m + radius_leniency)).all(axis=1)

    it = zip(df.loc[filt, site], index[filt])
    arr = []
    for m, ix in it:
        arr += [{site: m, 'ix': sorted(ix)}]

    return df_info.merge(pd.DataFrame(arr), how='left')


def get_metadata_at_coords(nd2, **coords):
    import pims_nd2
    h = pims_nd2.ND2SDK

    _coords = {'t': 0, 'c': 0, 'z': 0, 'o': 0, 'm': 0}
    _coords.update(coords)
    c_coords = h.LIMUINT_4(int(_coords['t']), int(_coords['m']), 
                           int(_coords['z']), int(_coords['o']))
    i = h.Lim_GetSeqIndexFromCoords(nd2._lim_experiment,
                                c_coords)

    h.Lim_FileGetImageData(nd2._handle, i, 
                           nd2._buf_p, nd2._buf_md)


    return {'x_um': nd2._buf_md.dXPos,
                    'y_um': nd2._buf_md.dYPos,
                    'z_um': nd2._buf_md.dZPos,
                    't_ms': nd2._buf_md.dTimeMSec,
                }

def get_axis_size(nd2,axis):
    try:
        size = list(range(nd2.sizes[axis]))
    except:
        size = [0]
    return size

def extract_nd2_metadata(f, interpolate=True, progress=None):
    """Interpolation fills in timestamps linearly for each well; x,y,z positions 
    are copied from the first time point. 
    """
    with ND2_Reader(f) as nd2:
        ts = get_axis_size(nd2,'t')
        ms = get_axis_size(nd2,'m')
        zs = get_axis_size(nd2,'z')   

        if progress is None:
            progress = lambda x: x

        if len(ts)==len(ms)==len(zs)==0:
            arr = [get_metadata_at_coords(nd2)]
        else:
            arr = []
            for t, m, z in progress(list(product(ts, ms, zs))):
                if len(ms)>1:
                    boundaries = [0, nd2.sizes['m'] - 1]
                    skip = m not in boundaries and t > 0
                else:
                    skip=False
                if interpolate and skip:
                    metadata = {}
                else:
                    metadata = get_metadata_at_coords(nd2, t=t, m=m, z=z)
                metadata['t'] = t
                metadata['m'] = m
                metadata['z'] = z
                metadata['file'] = f
                metadata.update()
                arr += [metadata]
        
    df_info = pd.DataFrame(arr)
    if interpolate:
        return (df_info
         .sort_values(['m', 't'])
         .assign(x_um=lambda x: x['x_um'].fillna(method='ffill'))
         .assign(y_um=lambda x: x['y_um'].fillna(method='ffill'))        
         .assign(z_um=lambda x: x['z_um'].fillna(method='ffill'))         
         .sort_values(['t', 'm'])
         .assign(t_ms=lambda x: x['t_ms'].interpolate())
                )
    else:
        return df_info



def build_file_table(f_nd2, f_template, wells):
    """
    Example:
    
    wells = 'A1', 'A2', 'A3', 'B1', 'B2', 'B3'
    f_template = 'input/10X_Hoechst-mNeon/10X_Hoechst-mNeon_A1.live.tif'
    build_file_table(f_nd2, f_template, wells)
    """
    rename = lambda x: name(parse(f_template), **x)

    get_well = lambda x: wells[int(re.findall('Well(\d)', x)[0]) - 1]
    df_files = (common.extract_nd2_metadata(f_nd2, progress=tqdn)
     .assign(well=lambda x: x['file'].apply(get_well))
     .assign(site=lambda x: x['m'])
     .assign(file_=lambda x: x.apply(rename, axis=1))
    )
    
    return df_files

def export_nd2(f_nd2, df_files):

    df = df_files.drop_duplicates('file_')

    with ND2_Reader(f_nd2) as nd2:

        nd2.iter_axes = 'm'
        nd2.bundle_axes = ['t', 'c', 'y', 'x']

        for m, data in tqdn(enumerate(nd2)):
            f_out = df.query('m == @m')['file_'].iloc[0]
            save(f_out, data)