import pandas as pd
import numpy as np
from glob import glob
from natsort import natsorted

# TODO: from ops.constants import *
from . import utils

def load_hist(filename, threshold):
    try:
        return (pd.read_csv(filename, sep='\s+', header=None)
            .rename(columns={0: 'count', 1: 'seq'})
            .query('count > @threshold')
            .assign(fraction=lambda x: x['count']/x['count'].sum())
            .assign(log10_fraction=lambda x: np.log10(x['fraction']))
            .assign(file=filename)
           )
    except pd.errors.EmptyDataError:
        return None


def load_sgRNA_hists(histogram_files, threshold=3):
    pat = '(?P<plate>T.)_(?P<well>(?P<row>.)(?P<col>..))_S'
    cols = ['dataset', 'plate', 'well', 'row', 'col', 
            'count', 'log10_fraction', 'fraction', 'sgRNA']
    arr = []
    for dataset, search in histogram_files.items():
        files = natsorted(glob(search))
        (pd.concat([load_hist(f, threshold) for f in files])
         .rename(columns={'seq': 'sgRNA'})
         .pipe(lambda x: pd.concat([x['file'].str.extract(pat), x], 
                                   axis=1))
         .pipe(utils.cast_cols, int_cols=['col'])
         .drop(['file'], axis=1)
         .assign(dataset=dataset)
         [cols]
         .pipe(arr.append)
         )

    return pd.concat(arr)

def calc_stats(df_hist, df_design, extra_cols=[]):
    sample_cols = ['dataset', 'plate', 'well', 'subpool'] + extra_cols
    sizes = df_design.groupby('subpool').size()
    fractions = (df_hist
     .groupby(sample_cols)
         ['fraction'].sum()
     .apply('{0:.1%}'.format)
    )

    cols = {'NGS_count': 'sgRNA_detected', 
            'NGS_missing': 'sgRNA_missing', 
            'NGS_designed': 'sgRNA_designed'}

    final_cols = ['NGS_fraction', 'NGS_Q10', 'NGS_Q50', 'NGS_Q90', 'NGS_Q90_10',
        'NGS_mean', 'NGS_std', 'NGS_max', 'NGS_min', 'sgRNA_designed', 
        'sgRNA_detected', 'sgRNA_missing']

    return (df_hist
     .groupby(sample_cols)['count']
     .describe(percentiles=[0.1, 0.5, 0.9])
     .rename(columns={'10%': 'Q10', 
                      '50%': 'Q50', 
                      '90%': 'Q90'})
     .join(sizes.rename('designed'), on='subpool')
     .assign(Q90_10=lambda x: x.eval('Q90 / Q10'))
     .assign(missing=lambda x: x.eval('designed - count').astype(int))
     .pipe(utils.cast_cols, int_cols=['count', 'max', 'min'])
     .join(fractions)
     .rename(columns=lambda x: 'NGS_' + x)
     .rename(columns=cols)
     [final_cols]
     .sort_values(['dataset', 'plate', 'well', 'sgRNA_detected'],
        ascending=[True, True, True, False])
    )     


def identify_pool(df_hist, df_design):
    cols = ['subpool', 'spots_per_oligo']
    return (df_hist
           .join(df_design.set_index('sgRNA')[cols], on='sgRNA')
           .pipe(add_design_rank, df_design)
           .sort_values(['dataset', 'plate', 'well', 'sgRNA', 'design_rank'])
           .groupby(['dataset', 'plate', 'well', 'sgRNA']).head(1)
           .sort_values(['dataset', 'plate', 'well', 'fraction'], 
              ascending=[True, True, True, False])
           .assign(mapped=lambda x: 1 - x['subpool'].isnull())
           .assign(mapped_fraction=lambda x: x.eval('fraction * mapped'))  
           )


def add_design_rank(df_hist, df_design):
    """For one file
    """
    a = df_design.groupby('subpool').size()
    b = df_hist.groupby('subpool').size()
    ranked = (((b / a) * np.log10(a))
              .dropna().sort_values(ascending=False))
    designs = {k: v for v, k in enumerate(list(ranked.index))}
    get_design = lambda x: designs.get(x, 1e10)
    return (df_hist.assign(design_rank=lambda x: 
                           x['subpool'].apply(get_design)))