import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_mapping_vs_threshold(df_reads, barcodes, threshold_var='peak',ax=None, **kwargs):
    """Plot the mapping rate and number of mapped spots against varying thresholds of 
    peak intensity, quality score, or a user-defined metric.

    Parameters
    ----------
    df_reads : pandas DataFrame
        Table of extracted reads from Snake.call_reads(). Can be concatenated results from
        multiple tiles, wells, etc.

    barcodes : list or set of strings
        Expected barcodes from the pool library design.

    threshold_var : string, default 'peak'
        Variable to apply varying thresholds to for comparing mapping rates. Standard variables are
        'peak' and 'QC_min'. Can also use a user-defined variable, but must be a column of the df_reads
        table.

    ax : None or matplotlib axis object, default None
        Optional. If not None, this is an axis object to plot on. Helpful when plotting on 
        a subplot of a larger figure.

    Other Parameters
    ----------------
    **kwargs
        Keyword arguments passed to sns.lineplot()
    
    Returns
    -------
    df_summary : pandas DataFrame
        Summary table of thresholds and associated mapping rates, number of spots mapped used for plotting.
    """
    # exclude spots not in cells
    df_passed = df_reads.copy().query('cell>0')

    # map reads
    df_passed.loc[:,'mapped'] = df_passed['barcode'].isin(barcodes)

    # define thresholds range
    if df_reads[threshold_var].max()<100:
        thresholds = np.array(range(0,int(np.quantile(df_passed[threshold_var],q=0.99)*1000)))/1000
    else:
        thresholds = list(range(0,int(np.quantile(df_passed[threshold_var],q=0.99)),10))

    # iterate over thresholds
    mapping_rate =[]
    spots_mapped = []
    for threshold in thresholds:
        df_passed = df_passed.query('{} > @threshold'.format(threshold_var))
        spots_mapped.append(df_passed[df_passed['mapped']].pipe(len))
        mapping_rate.append(df_passed[df_passed['mapped']].pipe(len)/df_passed.pipe(len))

    df_summary = pd.DataFrame(np.array([thresholds,mapping_rate,spots_mapped]).T,
        columns=['{}_threshold'.format(threshold_var),'mapping_rate','mapped_spots'])
    
    # plot
    if not ax:
        ax = sns.lineplot(data=df_summary, x='{}_threshold'.format(threshold_var), y='mapping_rate',**kwargs);
    else:
        sns.lineplot(data=df_summary, x='{}_threshold'.format(threshold_var), y='mapping_rate', ax=ax, **kwargs);
    ax.set_ylabel('mapping rate',fontsize=18);
    ax.set_xlabel('{} threshold'.format(threshold_var),fontsize=18);
    ax_right = ax.twinx()
    sns.lineplot(data=df_summary, x='{}_threshold'.format(threshold_var), y='mapped_spots', ax=ax_right, color='coral',**kwargs)
    ax_right.set_ylabel('mapped spots',fontsize=18);
    plt.legend(ax.get_lines()+ax_right.get_lines(),['mapping rate','mapped spots'],loc=7);

    return df_summary

def plot_count_heatmap(df,tile='tile',shape='square',plate='6W',return_summary=False,**kwargs):
    """Plot the count of items in df by well and tile in a convenient plate layout.
    Useful for evaluating cell and read counts across wells. The colorbar label can 
    be modified with:
        axes[0,0].get_figure().axes[-1].set_ylabel(LABEL)
    
    Parameters
    ----------
    df : pandas DataFrame

    tile : str, default 'tile'
        The column name to be used to group tiles, as sometimes 'site' is used.

    shape : str, default 'square'
        Shape of subplot for each well used in plot_plate_heatmap

    plate : {'6W','24W','96W'}
        Plate type for plot_plate_heatmap

    return_summary : boolean, default False
        If true, returns df_summary

    Other Parameters
    ----------------
    **kwargs
        Keyword arguments passed to plot_plate_heatmap()

    Returns
    -------
    df_summary : pandas DataFrame
        DataFrame used for plotting
        optional output, only returns if return_summary=True

    axes : np.array of matplotlib Axes objects
    """
    df_summary = (df
                  .groupby(['well',tile])
                  .size()
                  .rename('count')
                  .to_frame()
                  .reset_index()
                 )

    axes = plot_plate_heatmap(df_summary,shape=shape,plate=plate,**kwargs)

    if return_summary:
        return df_summary,axes
    return axes

def plot_cell_mapping_heatmap(df_cells, df_sbs_info, barcodes, mapping_to='one', shape='square', plate='6W',
    return_summary=False,**kwargs):
    """Plot the mapping rate of cells by well and tile in a convenient plate layout.

    Parameters
    ----------
    df_cells : pandas DataFrame
        DataFrame of all cells output from sbs mapping pipeline, e.g., concatenated outputs for all tiles and wells 
        of Snake.call_cells().

    df_sbs_info : pandas DataFrame
        DataFrame of all cells segmented from sbs images, e.g., concatenated outputs for all tiles and wells of 
        Snake.extract_phenotype_minimal(data_phenotype=nulcei,nuclei=nuclei) often used as sbs_cell_info rule in 
        Snakemake.

    barcodes : list or set of strings
        Expected barcodes from the pool library design.

    mapping_to : {'one', 'any'}
        Cells to include as 'mapped'. 'one' only includes cells mapping to a single barcode, 'any' includes cells
        mapping to at least 1 barcode.

    shape : str, default 'square'
        Shape of subplot for each well used in plot_plate_heatmap

    plate : {'6W','24W','96W'}
        Plate type for plot_plate_heatmap

    return_summary : boolean, default False
        If true, returns df_summary

    Other Parameters
    ----------------
    **kwargs
        Keyword arguments passed to plot_plate_heatmap()

    Returns
    -------
    df_summary : pandas DataFrame
        DataFrame used for plotting
        optional output, only returns if return_summary=True

    axes : np.array of matplotlib Axes objects
    """
    df_cells.loc[:,['mapped_0','mapped_1']] = df_cells[['cell_barcode_0','cell_barcode_1']].isin(barcodes).values

    df = (df_sbs_info[['well','tile','cell']]
           .merge(df_cells[['well','tile','cell','mapped_0','mapped_1']],
                  how='left',
                  on=['well','tile','cell']
                 )
          )

    if mapping_to == 'one':
        metric = 'fraction of cells mapping to 1 barcode'
        df = df.assign(mapped = lambda x: x[['mapped_0','mapped_1']].sum(axis=1)==1)
    elif mapping_to == 'any':
        metric = 'fraction of cells mapping to >=1 barcode'
        df = df.assign(mapped = lambda x: x[['mapped_0','mapped_1']].sum(axis=1)>0)
    else:
        raise ValueError(f'mapping_to={mapping_to} not implemented')

    df_summary = (df
                  .groupby(['well','tile'])
                  ['mapped']
                  .value_counts(normalize=True)
                  .rename(metric)
                  .to_frame()
                  .reset_index()
                  .query('mapped')
                  .drop(columns='mapped')
                 )

    axes = plot_plate_heatmap(df_summary,shape=shape,plate=plate,**kwargs)

    if return_summary:
        return df_summary,axes
    return axes

def plot_read_mapping_heatmap(df_reads, barcodes, shape='square', plate='6W', return_summary=False, **kwargs):
    """Plot the mapping rate of reads by well and tile in a convenient plate layout.

    Parameters
    ----------
    df_reads: pandas DataFrame
        DataFrame of all reads output from sbs mapping pipeline, e.g., concatenated outputs for all tiles and wells 
        of Snake.call_reads().

    barcodes : list or set of strings
        Expected barcodes from the pool library design.

    shape : str, default 'square'
        Shape of subplot for each well used in plot_plate_heatmap

    plate : {'6W','24W','96W'}
        Plate type for plot_plate_heatmap

    return_summary : boolean, default False
        If true, returns df_summary

    Other Parameters
    ----------------
    **kwargs
        Keyword arguments passed to plot_plate_heatmap()

    Returns
    -------
    df_summary : pandas DataFrame
        DataFrame used for plotting
        optional output, only returns if return_summary=True

    axes : np.array of matplotlib Axes objects
    """

    df_reads.loc[:,'mapped'] = df_reads['barcode'].isin(barcodes)

    df_summary  = (df_reads
                   .groupby(['well','tile'])
                   ['mapped']
                   .value_counts(normalize=True)
                   .rename('fraction of reads mapping')
                   .to_frame()
                   .reset_index()
                   .query('mapped')
                   .drop(columns='mapped')
                  )

    axes = plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)

    if return_summary:
        return df_summary,axes
    return axes

def plot_sbs_ph_matching_heatmap(df_merge, df_info, target='sbs', shape='square', plate='6W',
    return_summary=False, **kwargs):
    """Plot the rate of matching segmented cells between phenotype and SBS datasets by well and tile 
    in a convenient plate layout.

    Parameters
    ----------
    df_merge: pandas DataFrame
        DataFrame of all matched cells, e.g., concatenated outputs for all tiles and wells 
        of Snake.merge_triangle_hash(). Expects 'tile' and 'cell_0' columns to correspond to phenotype data and
        'site', 'cell_1' columns to correspond to sbs data.

    df_info : pandas DataFrame
        DataFrame of all cells segmented from either phenotype or sbs images, e.g., concatenated outputs for all tiles and wells of 
        Snake.extract_phenotype_minimal(data_phenotype=nulcei,nuclei=nuclei) often used as sbs_cell_info rule in 
        Snakemake.

    target : {'sbs','phenotype'}
        Which dataset to use as the target, e.g., if target='sbs' plots fraction of cells in each sbs tile that match to 
        a phenotype cell. Should match the information stored in df_info; if df_info is a table of all segmented cells from 
        sbs tiles then target should be set as 'sbs'.

    shape : str, default 'square'
        Shape of subplot for each well used in plot_plate_heatmap. Default infers shape based on value of target.

    plate : {'6W','24W','96W'}
        Plate type for plot_plate_heatmap

    return_summary : boolean, default False
        If true, returns df_summary

    Other Parameters
    ----------------
    **kwargs
        Keyword arguments passed to plot_plate_heatmap()

    Returns
    -------
    df_summary : pandas DataFrame
        DataFrame used for plotting
        optional output, only returns if return_summary=True

    axes : np.array of matplotlib Axes objects
    """
    if target == 'sbs':
        merge_cols = ['site','cell_1']
        source = 'phenotype'
        if not shape:
            shape = '6W_sbs'
    elif target == 'phenotype':
        merge_cols = ['tile','cell_0']
        source = 'sbs'
        if not shape:
            shape = '6W_ph'
    else:
        raise ValueError('target = {} not implemented'.format(target))

    df_summary = (df_info
                   .rename(columns={'tile':merge_cols[0],'cell':merge_cols[1]})
                   [['well']+merge_cols]
                   .merge(df_merge[['well']+merge_cols+['distance']],
                        how='left',on=['well']+merge_cols)
                   .assign(matched=lambda x: x['distance'].notna())
                   .groupby(['well']+merge_cols[:1])
                   ['matched']
                   .value_counts(normalize=True)
                   .rename('fraction of {} cells matched to {} cells'.format(target,source))
                   .to_frame()
                   .reset_index()
                   .query('matched==True')
                   .drop(columns='matched')
                   .rename(columns={merge_cols[0]:'tile'})
                  )
    axes = plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)

    if return_summary:
        return df_summary,axes
    return axes

def plot_plate_heatmap(df, metric=None, shape='square', plate='6W', snake_sites=True, **kwargs): 
    """Plot the rate of matching segmented cells between phenotype and SBS datasets by well and tile 
    in a convenient plate layout.

    Parameters
    ----------
    df: pandas DataFrame
        Summary DataFrame of values to plot, expects one row for each (well, tile) combination.

    metric : str, default None
        Column of `df` to use for plotting the heatmap. If None, attempts to infer based on column names.

    shape : {'square','6W_ph','6W_sbs',list}, default 'square'
        Shape of subplot for each well. 
            'square' infers dimensions of the smallest square that fits the number of sites.
            '6W_ph' and '6W_sbs' use a common  6 well tile map from a Nikon Ti2/Elements set-up with 20X and 10X objectives,
                respectively. 
            Alternatively, a list can be passed containing the number of sites in each row of a tile layout. This is mapped
                into a centered shape within a rectangle. Unused corners of this rectangle are plotted as NaN. The summation
                of this list should equal the total number of sites.

    plate : {'6W','24W','96W'}
        Plate type for plot_plate_heatmap

    snake_sites : boolean, default True
        If true, plots tiles in a snake order similar to the order of sites acquired by many high throughput 
        microscope systems.

    Other Parameters
    ----------------
    **kwargs
        Keyword arguments passed to matplotlib.pyplot.imshow()

    Returns
    -------
    axes : np.array of matplotlib Axes objects
    """
    import string
    
    tiles = max(len(df['tile'].unique()),df['tile'].max())
    
    # define grid for plotting
    if shape == 'square':
        r = c = int(np.ceil(np.sqrt(tiles)))
        grid = np.empty(r*c)
        grid[:] = np.NaN
        grid[:tiles]=range(tiles)
        grid = grid.reshape(r,c)
    else:
        if shape == '6W_ph':
            rows = [7,13,17,21,25,27,29,31,33,33,35,35,37,37,39,39,39,41,41,41,41,
                    41,41,41,39,39,39,37,37,35,35,33,33,31,29,27,25,21,17,13,7]
        elif shape == '6W_sbs':
            rows = [5,9,13,15,17,17,19,19,21,21,21,21,21,19,19,17,17,15,13,9,5]
        elif isinstance(shape,list):
            rows = shape
        else:
            raise ValueError('{} shape not implemented, can pass custom shape as a' 
                  'list specifying number of sites per row'.format(shape))
        
        r,c = len(rows),max(rows)
        grid = np.empty((r,c))
        grid[:] = np.NaN

        next_site = 0
        for row,row_sites in enumerate(rows):
            start = int((c-row_sites)/2)
            grid[row,start:start+row_sites] = range(next_site,next_site+row_sites)
            next_site += row_sites
            
    if snake_sites:
        grid[1::2] = grid[1::2, ::-1]
    
    # infer metric to plot if necessary
    if not metric:
        metric = [col for col in df.columns if col not in ['plate','well','tile']]
        if len(metric)!=1:
            raise ValueError('Cannot infer metric to plot, can pass metric column name explicitly to metric kwarg')
        metric = metric[0]
    
    # define subplots layout
    if df['well'].nunique()==1:
        wells = df['well'].unique()
        fig,axes = plt.subplots(1,1,figsize=(10,10))
        axes = np.array([axes])
    elif plate=='6W':
        wells = [f'{r}{c}' for r in string.ascii_uppercase[:2] for c in range(1,4)]
        fig,axes = plt.subplots(2,3,figsize=(15,10))
    elif plate=='24W':
        wells = [f'{r}{c}' for r in string.ascii_uppercase[:4] for c in range(1,7)]
        fig,axes = plt.subplots(4,6,figsize=(15,10))
    elif plate=='96W':
        wells = [f'{r}{c}' for r in string.ascii_uppercase[:8] for c in range(1,13)]
        fig,axes = plt.subplots(8,12,figsize=(15,10))
    else:
        wells = sorted(df['well'].unique())
        nr = nc = int(np.ceil(np.sqrt(len(wells))))
        if (nr - 1) * nc >= len(wells):
            nr -= 1
        fig,axes = plt.subplots(nr,nc,figsize=(15,15))
    
    # define colorbar min and max    
    cmin,cmax = (df[metric].min(),df[metric].max())

    # plot wells
    for ax,well in zip(axes.reshape(-1),wells):
        values = grid.copy()
        df_well = df.query('well==@well')
        if df_well.pipe(len)>0:
            for tile in range(tiles):
                try:
                    values[grid==tile] = df_well.loc[df_well.tile==tile,metric].values[0]
                except:
                    values[grid==tile] = np.nan
            plot = ax.imshow(values,vmin=cmin,vmax=cmax,**kwargs)
        ax.set_title('Well {}'.format(well),fontsize=24)
        ax.axis('off')
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.025, 0.7])
    try:
        cbar = fig.colorbar(plot,cax=cbar_ax)
    except:
        # plot variable empty, no data plotted
        raise ValueError('No data to plot')
    cbar.set_label(metric,fontsize=18)
    cbar_ax.yaxis.set_ticks_position('left')
    
    return axes, cbar
    