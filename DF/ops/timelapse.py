from . import utils

import networkx as nx
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from collections import Counter

from scipy.interpolate import UnivariateSpline
from statsmodels.stats.multitest import multipletests


def format_stats_wide(df_stats):
    index = ['gene_symbol']
    columns = ['stat_name', 'stimulant']
    values = ['statistic', 'pval', 'pval_FDR_10']

    stats = (df_stats
     .pivot_table(index=index, columns=columns, values=values)
     .pipe(utils.flatten_cols))

    counts = (df_stats
     .pivot_table(index=index, columns='stimulant', values='count')
     .rename(columns=lambda x: 'cells_' + x))

    return pd.concat([stats, counts], axis=1)


def distribution_difference(df):
    col = 'dapi_gfp_corr_early'
    y_neg = (df
      .query('gene_symbol == "non-targeting"')
      [col]
    )
    return df.groupby('gene_symbol').apply(lambda x:
      scipy.stats.wasserstein_distance(x[col], y_neg))


def add_est_timestamps(df_all):
    s_per_frame = 24 * 60
    sites_per_frame = 2 * 364
    s_per_site = s_per_frame / sites_per_frame
    starting_time = 3 * 60

    cols = ['frame', 'well', 'site']
    df_ws = df_all[cols].drop_duplicates().sort_values(cols)

    est_timestamps = [(starting_time + i*s_per_site) / 3600
                      for i in range(len(df_ws))]

    df_ws['timestamp'] = est_timestamps

    return df_all.join(df_ws.set_index(cols), on=cols)


def add_dapi_diff(df_all):
    index = ['well', 'site', 'cell_ph']
    dapi_diff = (df_all
     .pivot_table(index=index, columns='frame', 
                  values='dapi_max')
     .pipe(lambda x: x/x.mean())
     .pipe(lambda x: x.max(axis=1) - x.min(axis=1))
     .rename('dapi_diff')
    )
    
    return df_all.join(dapi_diff, on=index)


def add_spline_diff(df, s=25):

    T_neg, Y_neg = (df
     .query('gene_symbol == "non-targeting"')
     .groupby('timestamp')
     ['dapi_gfp_corr'].mean()
     .reset_index().values.T
    )

    ix = np.argsort(T_neg)
    spl = UnivariateSpline(T_neg[ix], Y_neg[ix], s=s)

    return (df
     .assign(splined=lambda x: spl(df['timestamp']))
     .assign(spline_diff=lambda x: x.eval('dapi_gfp_corr - splined'))
    )


def get_stats(df, col='spline_diff'):
    df_diff = (df
     .groupby(['gene_symbol', 'cell'])
     [col].mean()
     .sort_values(ascending=False)
     .reset_index())

    negative_vals = (df_diff
     .query('gene_symbol == "non-targeting"')
     [col]
    )

    test = lambda x: scipy.stats.ttest_ind(x, negative_vals).pvalue

    stats = (df_diff.groupby('gene_symbol')
     [col]
     .pipe(utils.groupby_reduce_concat, 'mean', 'count', 
           pval=lambda x: x.apply(test))
     .assign(pval_FDR_10=lambda x: 
            multipletests(x['pval'], 0.1)[1]))
    
    return stats

# track nuclei

def initialize_graph(df):
    arr_df = [x for _, x in df.groupby('frame')]
    nodes = df[['frame', 'label']].values
    nodes = [tuple(x) for x in nodes]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    edges = []
    for df1, df2 in zip(arr_df, arr_df[1:]):
        edges = get_edges(df1, df2)
        G.add_weighted_edges_from(edges)
    
    return G


def get_edges(df1, df2):
    neighboring_points = 3
    get_label = lambda x: tuple(int(y) for y in x[[2, 3]])

    x1 = df1[['i', 'j', 'frame', 'label']].values
    x2 = df2[['i', 'j', 'frame', 'label']].values
    
    kdt = cKDTree(df1[['i', 'j']])
    points = df2[['i', 'j']]

    result = kdt.query(points, neighboring_points)
    edges = []
    for i2, (ds, ns) in enumerate(zip(*result)):
        end_node = get_label(x2[i2])
        for d, i1 in zip(ds, ns):
            start_node = get_label(x1[i1])
            w = d
            edges.append((start_node, end_node, w))

    return edges


def displacement(x):
    d = np.sqrt(np.diff(x['x'])**2 + np.diff(x['y'])**2)
    return d


def analyze_graph(G, cutoff=100):
    """Trace a path forward from each nucleus in the starting frame. Only keep 
    the paths that reach the final frame.
    """
    start_nodes = [n for n in G.nodes if n[0] == 0]
    max_frame = max([frame for frame, _ in G.nodes])
    
    cost, path = nx.multi_source_dijkstra(G, start_nodes, cutoff=cutoff)
    cost = {k:v for k,v in cost.items() if k[0] == max_frame}
    path = {k:v for k,v in path.items() if k[0] == max_frame}
    return cost, path


def filter_paths(cost, path, threshold=35):
    """Remove intersecting paths. 
    returns list of one [(frame, label)] per trajectory
    """
    # remove intersecting paths (node in more than one path)
    node_count = Counter(sum(path.values(), []))
    bad = set(k for k,v in node_count.items() if v > 1)
    print('bad', len(bad), len(node_count))

    # remove paths with cost over threshold
    too_costly = [k for k,v in cost.items() if v > threshold]
    bad = bad | set(too_costly)
    
    relabel = [v for v in path.values() if not (set(v) & bad)]
    assert(len(relabel) > 0)
    return relabel


def relabel_nuclei(nuclei, relabel):
    nuclei_ = nuclei.copy()
    max_label = nuclei.max() + 1
    for i, nodes in enumerate(zip(*relabel)):
        labels = [n[1] for n in nodes]
        table = np.zeros(max_label).astype(int)
        table[labels] = range(len(labels))
        nuclei_[i] = table[nuclei_[i]]

    return nuclei_


# plot traces

def plot_traces_gene_stim(df, df_neg, gene):
    import ops.figures.plotting
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(12, 12), 
                        sharex=True, sharey=True)  
    for stim, df_1 in df.groupby('stimulant'):
        if stim == 'TNFa':
            axs_ = axs[:2]
            color = ops.figures.plotting.ORANGE
        else:
            axs_ = axs[2:]
            color = ops.figures.plotting.BLUE

        x_neg, y_neg = (df_neg
         .query('stimulant == @stim')
         .groupby(['frame'])
         [['timestamp', 'dapi_gfp_corr']].mean()
         .values.T)

        for ax, (sg, df_2) in zip(axs_.flat[:], 
                                 df_1.groupby('sgRNA_name')):
            plot_traces(df_2, ax, sg, color)
            ax.plot(x_neg, y_neg, c='black')
            
    return fig


def plot_traces(df, ax, sgRNA_label, color):
    
    index = ['well', 'tile', 'cell', 'sgRNA_name']
    values = ['timestamp', 'dapi_gfp_corr']
    wide = (df
     .pivot_table(index=index, columns='frame', 
                   values=values)
    )
    x = wide['timestamp'].values
    y = wide['dapi_gfp_corr'].values
    
    ax.plot(x.T, y.T, c=color, alpha=0.2)
    ax.set_title(sgRNA_label)
    

