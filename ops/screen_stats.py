import numpy as np
import pandas as pd

from .constants import *
from .utils import groupby_histogram, groupby_reduce_concat
from scipy.stats import wasserstein_distance, ks_2samp, ttest_ind

import seaborn as sns
import matplotlib.pyplot as plt


def process_rep(df, value='dapi_gfp_corr_nucleus',
               sgRNA_index=('sgRNA_name', 'gene_symbol')):
    """Calculate statistics for one replicate. "w_dist" corresponds to 
    "sgRNA translocation defect" in 2019 paper.
    
    Example:

    sample_index = ['replicate', 'stimulant', 'well']
    genes = ['MYD88', 'TRADD', 'nontargeting']
    stats = (df_cells
     .groupby(sample_index)
     .apply(process_rep).reset_index()
    )
    """
    sgRNA_index = list(sgRNA_index)
    nt = df.query('gene_symbol == "non-targeting"')[value]
    w_dist = lambda x: wasserstein_distance(x, nt)
    ks_test = lambda x: ks_2samp(x, nt)
    t_test = lambda x: ttest_ind(x, nt)

    return (df
     .groupby(sgRNA_index)[value]
     .pipe(groupby_reduce_concat, 'mean', 'count', 
           w_dist=w_dist, ks_test=ks_test, t_test=t_test)
     .assign(ks_pval=lambda x: x['ks_test'].apply(lambda y: y.pvalue))
     .assign(ks_stat=lambda x: x['ks_test'].apply(lambda y: y.statistic))
     .assign(ttest_pval=lambda x: x['t_test'].apply(lambda y: y.pvalue))
     .assign(ttest_stat=lambda x: x['t_test'].apply(lambda y: y.statistic))
    )


def score_genes_simple(df_stats):
    """Score genes by average translocation defect.
    """
    df_gene_stats = (df_stats
     .groupby(['gene_symbol', 'stimulant'])
     .apply(lambda x: x.eval('mean * count').sum() / x['count'].sum())
     .rename('mean')
     .reset_index()
     .pivot_table(index='gene_symbol', columns='stimulant', values='mean')
    )
    for col in df_gene_stats:
        df_gene_stats[f'{col}_rank'] = df_gene_stats[col].rank().astype(int)
    return df_gene_stats


def score_genes(df_stats):
    pass


def plot_distributions(df_cells, gene):
    
    df_neg = (df_cells
     .query('gene_symbol == "nt"').assign(sgRNA_name='nt'))
    df_gene = df_cells.query('gene_symbol == @gene')
    df_plot = pd.concat([df_neg, df_gene])

    replicates = sorted(set(df_plot['replicate']))
    bins = np.linspace(-1, 1, 100)
    hist_kws = dict(bins=bins, histtype='step', density=True, 
                    cumulative=True)
    row_order = 'TNFa', 'IL1b'
    fg = (df_plot
     .pipe(sns.FacetGrid, hue='sgRNA_name', col_order=replicates,
           col='replicate', row='stimulant', row_order=row_order)
     .map(plt.hist, 'dapi_gfp_corr_nuclear', **hist_kws)
    )
    
    return fg


# OLD (pre-binned)

def cells_to_distributions(df_cells, bins, column='dapi_gfp_corr_nuclear'):
    """
    
    Make sure to .dropna() first.
    """
    index = [GENE_SYMBOL, SGRNA_NAME, REPLICATE, STIMULANT]
    return (df_cells
     .pipe(groupby_histogram, index, column, bins)
     )


def plot_distributions_old(df_dist):
    """Old plotting function. 
    Plots from data that is already binned. Pre-filter for gene symbol of
    interest and LG non-targeting guides (shown individually).
    """

    # sgRNA names
    hue_order = (df_dist.reset_index()['sgRNA_name'].value_counts()
        .pipe(lambda x: natsorted(set(x.index))))
    colors = iter(sns.color_palette(n_colors=10))
    palette, legend_data = [], {}
    for name in hue_order:
        palette += ['black' if name.startswith('LG') else colors.next()]
        legend_data[name] = patches.Patch(color=palette[-1], label=name)


    
    def plot_lines(**kwargs):
        df = kwargs.pop('data')
        color = kwargs.pop('color')
        ax = plt.gca()
        (df
         .filter(regex='\d')
         .T.plot(ax=ax, color=color)
        )

    fg = (df_dist
     .pipe(normalized_cdf)
     .reset_index()
     .pipe(sns.FacetGrid, row='stimulant', hue='sgRNA_name', col='replicate', 
           palette=palette, hue_order=hue_order)
     .map_dataframe(plot_lines)
     .set_titles("{row_name} rep. {col_name}")
     .add_legend(legend_data=legend_data)
    )
    return fg
