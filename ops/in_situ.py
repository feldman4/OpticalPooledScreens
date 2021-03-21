import numpy as np
import pandas as pd
from .constants import *
from . import utils


def extract_base_intensity(maxed, peaks, cells, threshold_peaks):
    """Extract values from `maxed` where `peaks` exceeds `threshold_peaks`. Also extract
    cell label at read position from `cells` and i,j positions of reads.
    """
    # reads outside of cells get label 0
    read_mask = (peaks > threshold_peaks)
    values = maxed[:, :, read_mask].transpose([2, 0, 1])
    labels = cells[read_mask]
    positions = np.array(np.where(read_mask)).T

    return values, labels, positions


def format_bases(values, labels, positions, cycles, bases):
    """Arrange ?x?x? arrays of base intensity information into a dataframe in "long" format (one
    row per observation).
    """
    index = (CYCLE, cycles), (CHANNEL, bases)
    try:
        df = utils.ndarray_to_dataframe(values, index)
    except ValueError:
        print('failed to reshape extracted pixels to sequencing bases, writing empty table')
        return pd.DataFrame()

    df_positions = pd.DataFrame(positions, columns=[POSITION_I, POSITION_J])
    df = (df.stack([CYCLE, CHANNEL])
          .reset_index()
          .rename(columns={0: INTENSITY, 'level_0': READ})
          .join(pd.Series(labels, name=CELL), on=READ)
          .join(df_positions, on=READ)
          .sort_values([CELL, READ, CYCLE])
          )

    return df


def do_median_call(df_bases, cycles=12, channels=4, correction_only_in_cells=False):
    """Call reads from raw base signal using median correction. Use the 
    `correction_within_cells` flag to specify if correction is based on reads within 
    cells, or all reads.
    """
    if correction_only_in_cells:
        # first obtain transformation matrix W
        X_ = dataframe_to_values(df_bases.query('cell > 0'))
        _, W = transform_medians(X_.reshape(-1, channels))

        # then apply to all data
        X = dataframe_to_values(df_bases)
        Y = W.dot(X.reshape(-1, channels).T).T.astype(int)
    else:
        X = dataframe_to_values(df_bases)
        Y, W = transform_medians(X.reshape(-1, channels))

    df_reads = call_barcodes(df_bases, Y, cycles=cycles, channels=channels)

    return df_reads


def clean_up_bases(df_bases):
    """Sort. Pre-processing for `dataframe_to_values`.
    """
    return df_bases.sort_values([WELL, TILE, CELL, READ, CYCLE, CHANNEL])


def call_cells(df_reads):
    """Determine count of top barcodes 
    """
    cols = [WELL, TILE, CELL]
    s = (df_reads
         .drop_duplicates([WELL, TILE, READ])
         .groupby(cols)[BARCODE]
         .value_counts()
         .rename('count')
         .sort_values(ascending=False)
         .reset_index()
         .groupby(cols)
         )

    return (df_reads
            .join(s.nth(0)[BARCODE].rename(BARCODE_0), on=cols)
            .join(s.nth(0)['count'].rename(BARCODE_COUNT_0), on=cols)
            .join(s.nth(1)[BARCODE].rename(BARCODE_1), on=cols)
            .join(s.nth(1)['count'].rename(BARCODE_COUNT_1), on=cols)
            .join(s['count'].sum() .rename(BARCODE_COUNT), on=cols)
            .assign(**{BARCODE_COUNT_0: lambda x: x[BARCODE_COUNT_0].fillna(0),
                       BARCODE_COUNT_1: lambda x: x[BARCODE_COUNT_1].fillna(0)})
            .drop_duplicates(cols)
            .drop([READ, BARCODE], axis=1)  # drop the read
            # drop the read coordinates
            .drop([POSITION_I, POSITION_J], axis=1)
            .filter(regex='^(?!Q_)')  # remove read quality scores
            .query('cell > 0')  # remove reads not in a cell
            )


def call_cells_mapping(df_reads, df_pool=None):
    """Determine count of top two barcodes for each cell. If `df_pool` is 
    provided, barcodes that map to designed sequences will be prioritized.
    """
    if df_pool is not None:
        guide_info_cols = [SGRNA, GENE_SYMBOL, GENE_ID]
        # map reads
        df_mapped = (pd.merge(df_reads, df_pool[[PREFIX]], 
                              how='left', left_on=BARCODE, right_on=PREFIX)
            .assign(mapped=lambda x: pd.notnull(x[PREFIX]))
            .drop(PREFIX, axis=1)
            )
        extra_col = ['mapped']
    else:
        df_mapped = df_reads
        extra_col = []

    # choose top 2 barcodes, priority given by (mapped,count)
    cols = [WELL, TILE, CELL]
    s = (df_mapped
        .drop_duplicates([WELL, TILE, READ])
        .groupby(cols + extra_col)[BARCODE]
        .value_counts()
        .rename('count')
        .reset_index()
        .sort_values(extra_col + ['count'], ascending=False)
        .groupby(cols)
        )

    df_cells = (df_reads
        .join(s.nth(0)[BARCODE].rename(BARCODE_0), on=cols)
        .join(s.nth(0)['count'].rename(BARCODE_COUNT_0), on=cols)
        .join(s.nth(1)[BARCODE].rename(BARCODE_1), on=cols)
        .join(s.nth(1)['count'].rename(BARCODE_COUNT_1), on=cols)
        .join(s['count'].sum() .rename(BARCODE_COUNT), on=cols)
        .assign(**{BARCODE_COUNT_0: lambda x: x[BARCODE_COUNT_0].fillna(0),
                    BARCODE_COUNT_1: lambda x: x[BARCODE_COUNT_1].fillna(0)})
        .drop_duplicates(cols)
        .drop([READ, BARCODE], axis=1)  # drop the read
        # drop the read coordinates
        .drop([POSITION_I, POSITION_J], axis=1)
        .filter(regex='^(?!Q_)')  # remove read quality scores
        .query('cell > 0')  # remove reads not in a cell
        )

    if df_pool is not None:
        # merge guide info
        df_cells = (pd.merge(df_cells, df_pool[[PREFIX] + guide_info_cols], 
                             how='left', left_on=BARCODE_0, right_on=PREFIX)
            .rename({col: col + '_0' for col in guide_info_cols}, axis=1)
            .drop(PREFIX, axis=1)
            )
        df_cells = (pd.merge(df_cells, df_pool[[PREFIX] + guide_info_cols], 
                             how='left', left_on=BARCODE_1, right_on=PREFIX)
            .rename({col: col + '_1' for col in guide_info_cols}, axis=1)
            .drop(PREFIX, axis=1)
            )

    return df_cells

def call_cells_mapping(df_reads,df_pool):
    """Determine count of top barcodes, barcodes prioritized if barcode maps to given pool design
    """
    guide_info_cols = [SGRNA,GENE_SYMBOL,GENE_ID]

    # map reads
    df_mapped = (pd.merge(df_reads,df_pool[[PREFIX]],how='left',left_on=BARCODE,right_on=PREFIX)
                 .assign(mapped = lambda x: pd.notnull(x[PREFIX]))
                 .drop(PREFIX,axis=1)
                )

    # choose top 2 barcodes, priority given by (mapped,count)
    cols = [WELL, TILE, CELL]
    s = (df_mapped
       .drop_duplicates([WELL, TILE, READ])
       .groupby(cols+['mapped'])[BARCODE]
       .value_counts()
       .rename('count')
       .reset_index()
       .sort_values(['mapped','count'],ascending=False)
       .groupby(cols)
        )

    df_cells = (df_reads
      .join(s.nth(0)[BARCODE].rename(BARCODE_0),       on=cols)
      .join(s.nth(0)['count'].rename(BARCODE_COUNT_0), on=cols)
      .join(s.nth(1)[BARCODE].rename(BARCODE_1),       on=cols)
      .join(s.nth(1)['count'].rename(BARCODE_COUNT_1), on=cols)
      .join(s['count'].sum() .rename(BARCODE_COUNT),   on=cols)
      .assign(**{BARCODE_COUNT_0: lambda x: x[BARCODE_COUNT_0].fillna(0),
                 BARCODE_COUNT_1: lambda x: x[BARCODE_COUNT_1].fillna(0)})
      .drop_duplicates(cols)
      .drop([READ, BARCODE], axis=1) # drop the read
      .drop([POSITION_I, POSITION_J], axis=1) # drop the read coordinates
      .filter(regex='^(?!Q_)') # remove read quality scores
      .query('cell > 0') # remove reads not in a cell
    )

    # merge guide info
    df_cells = (pd.merge(df_cells,df_pool[[PREFIX] + guide_info_cols],how='left',left_on=BARCODE_0,right_on=PREFIX)
                .rename({col:col+'_0' for col in guide_info_cols},axis=1)
                .drop(PREFIX,axis=1)
               )
    df_cells = (pd.merge(df_cells,df_pool[[PREFIX]+ guide_info_cols],how='left',left_on=BARCODE_1,right_on=PREFIX)
                .rename({col:col+'_1' for col in guide_info_cols},axis=1)
                .drop(PREFIX,axis=1)
               )

    return df_cells


def dataframe_to_values(df, value='intensity'):
    """Dataframe must be sorted on [cycle, channel]. 
    Returns N x cycles x channels.
    """
    cycles = df[CYCLE].value_counts()
    assert len(set(cycles)) == 1
    n_cycles = len(cycles)
    n_channels = len(df[CHANNEL].value_counts())
    x = np.array(df[value]).reshape(-1, n_cycles, n_channels)
    return x


def transform_medians(X):
    """Estimate and correct differences in channel intensity and spectral overlap among sequencing
    channels. For each channel, find points where the largest signal is from that channel. Use the 
    median of these points to define new basis vectors. Describe with linear transformation W 
    so that W * X = Y, where Y is the corrected data.
    """

    def get_medians(X):
        arr = []
        for i in range(X.shape[1]):
            arr += [np.median(X[X.argmax(axis=1) == i], axis=0)]
        M = np.array(arr)
        return M

    M = get_medians(X).T
    M = M / M.sum(axis=0)
    W = np.linalg.inv(M)
    Y = W.dot(X.T).T.astype(int)
    return Y, W


def call_barcodes(df_bases, Y, cycles=12, channels=4):
    """Use the argmax over channels in corrected data `Y` to identify bases in each cycle. Result
    dataframe has one row per read and preserves any extra information from `df_bases`.
    """
    bases = sorted(set(df_bases[CHANNEL]))
    if any(len(x) != 1 for x in bases):
        raise ValueError('supplied weird bases: {0}'.format(bases))
    df_reads = df_bases.drop_duplicates([WELL, TILE, READ]).copy()
    df_reads[BARCODE] = call_bases_fast(Y.reshape(-1, cycles, channels), bases)
    Q = quality(Y.reshape(-1, cycles, channels))
    # needed for performance later
    for i in range(len(Q[0])):
        df_reads['Q_%d' % i] = Q[:, i]

    return (df_reads
            .assign(Q_min=lambda x: x.filter(regex='Q_\d+').min(axis=1))
            .drop([CYCLE, CHANNEL, INTENSITY], axis=1)
            )


def call_bases_fast(values, bases):
    """Apply argmax to `values` and form strings by indexing `bases` (e.g., "ACGT" if channel
    dimension in `values` is pre-sorted).
    """
    assert values.ndim == 3
    assert values.shape[2] == len(bases)
    calls = values.argmax(axis=2)
    calls = np.array(list(bases))[calls]
    return [''.join(x) for x in calls]


def quality(X):
    """Define an ad-hoc quality score per sequencing call based on the highest and 
    second-highest channels. Adjusted empirically to give reasonable results with 4-color 
    sequencing.
    """
    X = np.abs(np.sort(X, axis=-1).astype(float))
    Q = 1 - np.log(2 + X[..., -2]) / np.log(2 + X[..., -1])
    Q = (Q * 2).clip(0, 1)
    return Q


def reads_to_fastq(df_reads, microscope='MN', dataset='DS', flowcell='FC'):
    """Convert reads to fastq format, including quality, tile position, and x,y position.
    """
    def wrap(x): return '{' + x + '}'
    def join_fields(xs): return ':'.join(map(wrap, xs))

    a = '@{m}:{d}:{f}'.format(m=microscope, d=dataset, f=flowcell)
    b = join_fields([WELL, CELL, 'well_tile', READ, POSITION_I, POSITION_J])
    c = '\n{b}\n+\n{{phred}}'.format(b=wrap(BARCODE))
    fmt = a + b + c

    well_tiles = sorted(set(df_reads[WELL] + '_' + df_reads[TILE]))
    fields = [WELL, TILE, CELL, READ, POSITION_I, POSITION_J, BARCODE]

    Q = df_reads.filter(like='Q_').values

    reads = []
    for i, row in enumerate(df_reads[fields].values):
        d = dict(zip(fields, row))
        d['phred'] = ''.join(phred(q) for q in Q[i])
        d['well_tile'] = well_tiles.index(d[WELL] + '_' + d[TILE])
        reads.append(fmt.format(**d))

    return reads


def dataframe_to_fastq(df, file, dataset):
    """Dump reads to fastq format. May be useful for applying NGS-specific tools like FastQC.
    """
    s = '\n'.join(reads_to_fastq(df, dataset))
    with open(file, 'w') as fh:
        fh.write(s)
        fh.write('\n')


def phred(q):
    """Convert 0...1 to 0...30
    No ":".
    No "@".
    No "+".
    """
    n = int(q * 30 + 33)
    if n == 43:
        n += 1
    if n == 58:
        n += 1
    return chr(n)


def add_clusters(df_cells, barcode_col=BARCODE_0, radius=50,
                 verbose=True, ij=(POSITION_I, POSITION_J)):
    """Define spatial clusters based on barcode identity. Rows with the same value in `barcode_col`
    are assigned to a cluster if `ij` positions are within `radius`. Assigns cluster index -1 to 
    clusters with only one cell.
    """
    from scipy.spatial import cKDTree
    import networkx as nx

    barcodes = df_cells[barcode_col].values

    I, J = ij
    kdt = cKDTree(np.array([df_cells[J], df_cells[I]]).T)
    num_cells = len(df_cells)

    if verbose:
        print(f'searching for clusters among {num_cells} {barcode_col} objects')
    pairs = kdt.query_pairs(radius)
    
    # detect clusters
    if len(pairs) == 0:
        clusters = []
    else:
        pairs = np.array(list(pairs))

        x = barcodes[pairs]
        y = x[:, 0] == x[:, 1]

        G = nx.Graph()
        G.add_edges_from(pairs[y])

        clusters = list(nx.connected_components(G))

    # fill in -1 for objects not in a cluster
    cluster_index = np.zeros(num_cells, dtype=int) - 1
    for i, c in enumerate(clusters):
        cluster_index[list(c)] = i

    df_cells = df_cells.copy()
    df_cells[CLUSTER] = cluster_index
    df_cells[CLUSTER_SIZE] = (df_cells
                              .groupby(CLUSTER)[barcode_col].transform('size'))
    
    # objects not in a cluster have cluster size 1
    df_cells.loc[df_cells[CLUSTER] == -1, CLUSTER_SIZE] = 1
    return df_cells


def index_singleton_clusters(clusters):
    """Replace cluster labels of -1 with ascending integers larger than the maximum cluster index.
    """
    clusters = clusters.copy()
    filt = clusters == -1
    n = clusters.max()
    clusters[filt] = range(n, n + len(filt))
    return clusters


def join_by_cell_location(df_cells, df_ph, max_distance=4):
    """Join together dataframes representing in situ sequencing data and phenotypic data on a 
    per-cell basis. Datasets must already be spatially aligned; cells are matched within 
    max_distance` (typically less than radius of cell nucleus) based on global position 
    (columns `global_x`, `global_y`).
    
    Can speed up over independent fields of view with `utils.groupby_apply2`.
    """
    from scipy.spatial import cKDTree
    i_tree = df_ph['global_y']
    j_tree = df_ph['global_x']
    i_query = df_cells['global_y']
    j_query = df_cells['global_x']

    kdt = cKDTree(list(zip(i_tree, j_tree)))
    distance, index = kdt.query(list(zip(i_query, j_query)))
    cell_ph = df_ph.iloc[index]['cell'].pipe(list)
    cols_left = ['well', 'tile', 'cell_ph']
    cols_right = ['well', 'tile', 'cell']
    cols_ph = [c for c in df_ph.columns if c not in df_cells.columns]
    return (df_cells
            .assign(cell_ph=cell_ph, distance=distance)
            .query('distance < @max_distance')
            .join(df_ph.set_index(cols_right)[cols_ph], on=cols_left)
            )
