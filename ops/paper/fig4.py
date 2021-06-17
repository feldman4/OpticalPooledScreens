import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import to_rgba
from skimage.morphology import closing
from skimage.measure import regionprops, find_contours
import itertools

from ..utils import add_fstrings, subimage
from ..in_situ import add_clusters, index_singleton_clusters
from ..plates import add_global_xy
from ..imports import read, save, rename, GRAY, GREEN
from ..annotate import annotate_labels, relabel_array, outline_mask
from ..screen_stats import process_rep

from .fig3 import default_rc_params

labels = {
    'dapi_gfp_corr_nuclear': 'DAPI:p65 correlation',
    'dapi_gfp_corr': 'DAPI:p65 correlation',
    't_hr': 'Time after\nstimulation (hrs)',
}


combined_csv = 'experimentC/process/combined.csv'
bases_csv_search = 'experimentC/process/*A1*bases.csv'
reads_csv_search = 'experimentC/process/*A1*reads.csv'
barcode_table = 'experimentC/barcodes.csv'
snakemake_config = 'experimentC/config_small_fig3.yaml'


class PanelA:
    """Example results and analysis for one image field of SBS and phenotype data.
    """
    wells = ['A1', 'A2', 'A3']
    well, tile, cluster = 'A1', 456, 30
    gene = 'TNFRSF1A'
    window = 100
    min_cluster_size = 5
    process_example = 'experimentC/process_debug/10X_A1_Tile-456.aligned.tif'
    display_ranges = {
        'p65': [400, 3000],
        'DAPI': [100, 3000],
        'SBS_G': [700, 5000],
        'SBS_T': [600, 3000],
        'SBS_A': [600, 4000],
        'SBS_C': [600, 3000],
    }
    simple_colors = ['red', 'gray']

    rc_params = default_rc_params.copy()
    rc_params.update({
        'legend.facecolor': 'white',
        'legend.framealpha': 1,
        'legend.fontsize': 10,
    })

    
    def load_combined():
        """Cluster IDs are only unique within a specific (well, tile).
        """
        return (pd.read_csv(combined_csv)
         .pipe(add_global_xy, '6w', (25, 25), ij=('i_nucleus', 'j_nucleus'))
         .groupby(['well', 'tile'])
         .progress_apply(add_clusters, ij=('global_x', 'global_y'), verbose=False)
         .reset_index(drop=True)
        )

    def find_example(df_combined):
        """Search for candidate field of view containing a cluster of cells mapped to sgRNA
        for gene of interest.
        """

        df_filtered = (df_combined
         .query('gene_symbol == @PanelA.gene & well == @PanelA.wells')
         .query('cell_barcode_1 != cell_barcode_1')
         .query('cluster > -1')
        )

        barcode_counts = (df_filtered
         .groupby(['well', 'tile', 'cluster', 'cluster_size'])
         [['cell_barcode_count_0', 'dapi_gfp_corr_nucleus']].mean()
         .reset_index()
        )

        return df_filtered, barcode_counts

    def filter_examples(barcode_counts):
        return (barcode_counts
         .query('cluster_size >= @PanelA.min_cluster_size')
         .sort_values('cell_barcode_count_0')
        )

    def save_example_crop(df_filtered, 
                          save_prefix='experimentC/fig4/example2_',
                          well=None, tile=None, cluster=None):
        """Cell mask missing some, could fix later.
        """
        
        well = PanelA.well if well is None else well
        tile = PanelA.tile if tile is None else tile
        cluster = PanelA.cluster if cluster is None else cluster
        df = (df_filtered
         .query('well == @well & tile == @tile & cluster == @cluster')
        )
        # return df
        i, j = df[['i_cell', 'j_cell']].mean().astype(int)
        
        # load data
        d = dict(well=well, tile=tile)
        f = PanelA.process_example
        data_sbs = read(rename(f, tag='aligned', **d))
        data_ph = read(rename(f, tag='phenotype_aligned', **d))
        cells = read(rename(f, tag='cells', **d))
        # load labels
        labeled, label_key = (df
         .pipe(annotate_labels, 'cell', 'gene_symbol', label_mask=cells, return_key=True)
        )

        f = f'{save_prefix}{well}_{tile}.tif'
        luts = GRAY, GREEN, GRAY
        annotated = np.array(list(data_ph) + [labeled])
        crop = lambda x: subimage(x, [i, j, i, j], pad=PanelA.window)

        save(f, crop(annotated), luts=luts)

    def load_reprocessed_data(df_combined, well, tile, cluster):
        """Load data associated with cluster.
        """
        window = PanelA.window

        # center crop on cluster with gene of interest
        i, j = (df_combined
         .query('well == @well & tile == @tile & cluster == @cluster')
        [['i_nucleus', 'j_nucleus']].mean().astype(int)
        )
        crop = lambda x: subimage(x, [i, j, i, j], pad=window)

        # load data for this well,tile
        f = PanelA.process_example
        d = dict(well=well, tile=tile)
        dapi, p65 = crop(read(rename(f, tag='phenotype_aligned', **d)))
        sbs_stack = crop(read(rename(f, tag='aligned', **d)))
        cells = crop(read(rename(f, tag='cells', **d)))

        selem = np.ones((3, 3))
        cells = closing(cells, selem=selem)

        # filter to just the cells in the cropped area, clustered by sgRNA
        df_window = (df_combined
        .query('well == @well & tile == @tile')
        # prefixes not sequenced as designed
        .drop_duplicates(['well', 'tile', 'cell', 'cell_barcode_0'], keep=False)
        .pipe(add_clusters, 'sgRNA', 
            ij=('i_nucleus', 'j_nucleus'), verbose=False)
        .query('@i - @window < i_nucleus < @i + @window')
        .query('@j - @window < j_nucleus < @j + @window')
        )

        return df_window, p65, sbs_stack, cells, (i, j)
        
    def make_cell_outline(df_window, cells, outline_method):
        gene_symbols = df_window['gene_symbol'].drop_duplicates().dropna().pipe(list)
        if outline_method == 'cluster':
            outline, key = (df_window
            .pipe(annotate_labels, 'cell', 'gene_symbol', 
                label_mask=cells, outline=None, return_key=True))
            outline = outline_mask(outline, direction='inner')
        elif outline_method == 'cell':
            outline, key = (df_window
            .pipe(annotate_labels, 'cell', 'gene_symbol', 
                label_mask=cells, outline='label', return_key=True))
        elif outline_method == 'target_only':
            # outline the cells matching the target cluster
            target_cluster = (df_window
             .query('cluster == @PanelA.cluster')
             .pipe(annotate_labels, 'cell', 'gene_symbol', 
                label_mask=cells, outline='label')
            )
            # mask the cells that don't map
            unmapped_id = 2
            mapped_cells = df_window.dropna(subset=['sgRNA'])['cell'].values
            
            unmapped_id = len(gene_symbols) - 1
            mask = (cells.flat[:] > 0) & ~np.in1d(cells, mapped_cells)
            print(mask.sum(), unmapped_id)
            outline.flat[mask] = unmapped_id

        else:
            raise ValueError('outline_method must be one of (cell, cluster)')

        # annotate_labels returns key dictionary of image label => gene_symbol 

        
        return outline, gene_symbols

    def simple_plot_phenotype(df_combined, well=None, tile=None, cluster=None, figsize=(6, 6)):
        well = PanelA.well if well is None else well
        tile = PanelA.tile if tile is None else tile
        cluster = PanelA.cluster if cluster is None else cluster

        df_window, p65, sbs_stack, cells, ij = (
            PanelA.load_reprocessed_data(df_combined, well, tile, cluster))

        # prepare the integer-coded mask
        coded_mask = (df_combined
        .query('well == @well & tile == @tile & cluster == @cluster')
        .pipe(annotate_labels, 'cell', 'gene_symbol', 
            label_mask=cells)
        )
        coded_mask = outline_mask(coded_mask, direction='inner')
        
        # label unmapped cells
        mapped_cells = (df_combined
        .query('well == @well & tile == @tile')
        .dropna(subset=['sgRNA'])
        ['cell']
        )
        not_mapped = ~np.isin(cells, mapped_cells)
        not_bkgd = cells > 0
        coded_mask[not_mapped & not_bkgd] = 2

        palette = np.array(sns.color_palette(['black'] + PanelA.simple_colors))

        rgb_mask = palette[coded_mask]
        p65_min, p65_max = PanelA.display_ranges['p65']
        rgb = np.zeros(p65.shape + (3,))
        rgb[..., 1] = (p65.astype(float) - p65_min) / (p65_max - p65_min)
        rgb += rgb_mask
        rgb = rgb.clip(min=0, max=1)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(rgb)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    def plot_phenotype(df_window, p65, outline, gene_symbols, ij, figsize=(6, 6)):
        window = PanelA.window
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = 'red', 'lavender', 'cyan', 'yellow', 'puce', 'orange'
        palette = [to_rgba(sns.xkcd_rgb[x])[:3] 
                for x,_ in zip(itertools.cycle(colors), gene_symbols)]

        # replace non-targeting with white
        for i_, gene_symbol in enumerate(gene_symbols):
            if gene_symbol == 'non-targeting':
                palette[i_] = [1, 1, 1]
        
        # include [0, 0, 0] for background pixels
        outline_palette = np.array([[0, 0, 0]] + palette)
        outline_rgb = outline_palette[outline]
        colors = dict(zip(gene_symbols, palette))

        # combine green phenotype and color-coded outlines
        p65_min, p65_max = PanelA.display_ranges['p65']
        rgb = np.zeros(p65.shape + (3,))
        rgb[..., 1] = (p65.astype(float) - p65_min) / (p65_max - p65_min)
        rgb += outline_rgb
        rgb = rgb.clip(min=0, max=1)

        i, j = ij
        ax.imshow(rgb, extent=(j - window, j + window, i - window, i + window))
        
        # display labels
        for _, df in df_window.groupby('cluster'):
            gene_symbol = df['gene_symbol'].iloc[0]
            if gene_symbol != gene_symbol:
                continue
            color = colors[gene_symbol]
            if gene_symbol == 'non-targeting':
                gene_symbol = 'control'
            

            i_cl, j_cl = df[['i_nucleus', 'j_nucleus']].mean()
            txt = ax.text(j_cl, i_cl - 10, gene_symbol,
                color=color, fontweight='bold', ha='center', fontsize=14)
            effects = [patheffects.withStroke(linewidth=3, foreground=(0, 0, 0, 0.5))]
            txt.set_path_effects(effects)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        return fig, outline_rgb

    def plot_phenotype2(df_combined, well=None, tile=None, cluster=None, 
                        window_view=100, window_label=1000, outline_closing=5,
                        num_clusters=7, figsize=(4, 4), sns_palette='pastel'):
        """Rewrote to use contour plots, mask unmapped cells, and use a normal legend.
        """
        well = PanelA.well if well is None else well
        tile = PanelA.tile if tile is None else tile
        cluster = PanelA.cluster if cluster is None else cluster

        # load data for this well,tile
        f = PanelA.process_example
        d = dict(well=well, tile=tile)
        dapi, p65 = read(rename(f, tag='phenotype_aligned', **d))
        sbs_stack = read(rename(f, tag='aligned', **d))
        cells = read(rename(f, tag='cells', **d))

        df_tile = (df_combined
        .query('well == @well & tile == @tile')
        # prefixes not sequenced as designed
        .drop_duplicates(['well', 'tile', 'cell', 'cell_barcode_0'], keep=False)         
        .query('sgRNA == sgRNA')
        )
        
        df_tile['cluster'] = index_singleton_clusters(df_tile['cluster'])

        df_target = df_tile.query('cluster == @cluster')
        # center of target cluster
        i0, j0 = df_target[['i_nucleus', 'j_nucleus']].mean()
        
        # prepare mNeonGreen layer
        p65_min, p65_max = PanelA.display_ranges['p65']
        p65_rgb = np.zeros(p65.shape + (3,))
        p65_rgb[..., 1] = (p65.astype(float) - p65_min) / (p65_max - p65_min)
        p65_rgb = p65_rgb.clip(0, 1)

        # prepare unmapped cells
        mapped_cells = (df_tile
        .dropna(subset=['sgRNA'])
        ['cell']
        )
        cell_regions = np.array(regionprops(cells))
        cell_centroids = np.array([r.centroid for r in cell_regions])
        nearby_index = np.abs((cell_centroids - (i0, j0)).max(axis=1)) < window_label
        nearby_cells = [r.label for r in cell_regions[nearby_index]]

        not_mapped = ~np.isin(cells, mapped_cells)
        # nearby = np.isin(cells, nearby_cells)
        not_bkgd = cells > 0
        unmapped = 1 * (not_mapped & not_bkgd)
        selem = np.ones((outline_closing, outline_closing))
        unmapped = closing(unmapped, selem=selem)
        colors = np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 0.5]
        ])
        unmapped = colors[unmapped]

        # this function plots the overlay for a specific cluster
        # the same color/legend entry is used for each gene symbol
        seen_before = set()
        def plot_one_cluster(df_tile, ax, cluster, **kwargs):
            # global seen_before
            df = df_tile.query('cluster == @cluster')
            gene_symbol = df['gene_symbol'].iloc[0]
            mask = annotate_labels(df, 'cell', 'gene_symbol', label_mask=cells)
            selem = np.ones((outline_closing, outline_closing))
            mask = closing(mask, selem=selem)
            contours = find_contours(mask, 0.5)
            for contour in contours:
                if gene_symbol in seen_before:
                    label = None
                else:
                    label = gene_symbol
                    seen_before.add(gene_symbol)
                ax.plot(contour[:, 1], contour[:, 0], color=palette[gene_symbol], **kwargs)


        cluster_coords = df_tile.groupby('cluster')[['i_nucleus', 'j_nucleus']].mean()
        distance = ((cluster_coords - (i0, j0))**2).sum(axis=1)**0.5
        neighbors = distance.sort_values()[1:num_clusters + 1].index

        with plt.rc_context(PanelA.rc_params):
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(p65_rgb)
            ax.imshow(unmapped)
            ax.set_xlim([j0 - window_view, j0 + window_view])
            ax.set_ylim([i0 - window_view, i0 + window_view])

            # make sure each gene symbol has a unique plot color
            clusters_to_plot = [cluster] + list(neighbors)
            gene_symbols = (df_tile
             .query('cluster == @clusters_to_plot')
             ['gene_symbol'].drop_duplicates()
            )
            palette = sns.color_palette(palette=sns_palette, n_colors=len(gene_symbols))
            palette = {k: v for k,v in zip(gene_symbols, palette)}
            palette[PanelA.gene] = 'white'
            palette['non-targeting'] = 'dodgerblue'

            target_kws = dict(lw=4, zorder=10)
            plot_one_cluster(df_tile, ax, cluster, lw=4, zorder=10)
            for n in neighbors:
                plot_one_cluster(df_tile, ax, n, lw=3)

            # make the legend
            from matplotlib import patheffects as pe
            from matplotlib.lines import Line2D
            opts = dict(lw=4, path_effects=[pe.Stroke(linewidth=8, foreground='black'), 
                                            pe.Normal()])
            gene_order = [PanelA.gene]
            if 'non-targeting' in seen_before:
                gene_order += ['non-targeting']
            gene_order += [x for x in seen_before if x not in gene_order]
            lines = []
            for gene_symbol in gene_order:
                lines += [Line2D([0], [0], color=palette[gene_symbol], **opts)]

            ax.legend(lines, gene_order, loc='best', handlelength=1.5)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        
        return fig, cells, sbs_stack, (i0, j0)

    def plot_sbs_cycle(data, outline_rgb=None, cells=None, figsize=(4, 4)):
        colors = 'white', (0, 1, 0), 'red', 'magenta', 'cyan'
        channels = 'DAPI', 'SBS_G', 'SBS_T', 'SBS_A', 'SBS_C'
        display_ranges = np.array([PanelA.display_ranges[x] for x in channels])

        rgb = colorize(data, colors, display_ranges)
        if outline_rgb is not None:
            mask = outline_rgb > 0
            rgb[mask] = 0
            rgb += outline_rgb

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(rgb)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        if cells is not None:
            from skimage.measure import regionprops, find_contours
            from skimage.morphology import dilation
            cells_simple = closing(cells, selem=np.ones((5, 5)))
            cells_simple[dilation(cells_simple) != cells_simple] = 0
            contours = find_contours(cells_simple, 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color='white')
        
        return fig

    def outline_rgb_to_gray(outline_rgb, shade=0.5):
        outline_gray = (outline_rgb > 0).any(axis=-1) * shade
        outline_gray = np.tile(outline_gray, (3, 1, 1)).transpose([1, 2, 0])
        return outline_gray

    def plot_base_calls(outline_rgb, ij, figsize=(4, 4)):
        i,j = ij
        crop = lambda x: subimage(x, [i, j, i, j], pad=PanelA.window)

        data = read(rename(PanelA.process_example, tag='aligned'))[0] 
        data = crop(data)
        colors = 'white', (0, 1, 0), 'red', 'magenta', 'cyan'
        channels = 'DAPI', 'SBS_G', 'SBS_T', 'SBS_A', 'SBS_C'
        display_ranges = np.array([PanelA.display_ranges[x] for x in channels])
        sbs_rgb = colorize(data, colors, display_ranges)

        outline_gray = PanelA.outline_rgb_to_gray(outline_rgb, 1)
        f = rename(PanelA.process_example, tag='annotate_SBS')
        base_calls = crop(read(f))[0, -1]
        print(base_calls.shape)
        
        colors = 'black', (0, 1, 0), 'red', 'magenta', 'cyan'
        grmc = np.array([to_rgba(c)[:3] for c in colors])
        grmc_rgb = grmc[base_calls]

        # a bit different
        # f = rename(PanelA.process_example, tag='aligned')
        # dapi = crop(read(f))[0, 0]
        # dr = np.array([PanelA.display_ranges['DAPI']])
        # dapi_rgb = colorize(dapi[None], [[1, 1, 1]], dr)

        rgb = grmc_rgb + sbs_rgb
        rgb[outline_gray > 0] = 0
        rgb += outline_gray
        rgb = rgb.clip(min=0, max=1)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(rgb)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        return fig

    def plot_base_calls2(sbs_stack, cells, figsize=(4, 4)):
        well = PanelA.well
        tile = PanelA.tile
        colors = 'white', 
        channels = 'DAPI',
        display_ranges = np.array([PanelA.display_ranges[x] for x in channels])
        sbs_rgb = colorize(sbs_stack[0][[0]], colors, display_ranges)

        f = rename(PanelA.process_example, well=well, tile=tile, tag='annotate_SBS')
        base_calls = read(f)[0, -1]

        colors = 'black', (0, 1, 0), 'red', 'magenta', 'cyan'
        grmc = np.array([to_rgba(c)[:3] for c in colors])
        grmc_rgb = grmc[base_calls]

        rgb = grmc_rgb + sbs_rgb*1.1
        rgb = rgb.clip(min=0, max=1)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(rgb)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        if cells is not None:
            from skimage.measure import regionprops, find_contours
            from skimage.morphology import dilation
            cells_simple = closing(cells, selem=np.ones((5, 5)))
            cells_simple[dilation(cells_simple) != cells_simple] = 0
            contours = find_contours(cells_simple, 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color='white')

        return fig


class PanelB:
    """Activation screen data presented as per-sgRNA distributions.
    """
    gene_order = ['Non-targeting', 'TNFRSF1A', 'MAP3K7', 'IL1R1']
    wells = ['A1', 'A2', 'A3']
    cells_per_sgRNA = 100
    rc_params = default_rc_params.copy()
    rc_params.update({
    })

    def load_data():
        # TODO: update to process_fig4_all result
        source = 'lasagna/20190403_6W-277A/combined.hdf'
        pd.read_hdf('')

    def prepare_data(df_combined, cells_per_sgRNA=None):
        if cells_per_sgRNA is None:
            cells_per_sgRNA = PanelB.cells_per_sgRNA
            
        df_targeting = (df_combined
         .query('well == @PanelB.wells')
         .query('gene_symbol == @PanelB.gene_order')
         .groupby('sgRNA_name').apply(lambda x: 
           x.sample(cells_per_sgRNA, replace=False, random_state=0) 
              if len(x) > cells_per_sgRNA else x)
         .reset_index(drop=True)
        )

        nontargeting = ['sg_nt213', 'sg_nt393', 'sg_nt8', 'sg_nt562', 'sg_nt759',]
        df_nt = (df_combined
         .query('well == @PanelB.wells')
         .query('sgRNA_name == @nontargeting')
         .assign(gene_symbol='Non-targeting')
         .groupby('sgRNA_name').apply(lambda x: 
           x.sample(cells_per_sgRNA, replace=False, random_state=0))
         .reset_index(drop=True)
        )

        df_plot = (pd.concat([df_nt, df_targeting])
         .pipe(PanelB.add_sgRNA_numbers)
         .sort_values('gene_symbol', key=lambda x: x.apply(PanelB.gene_order.index))
        )
        for k,v in labels.items():
            if k in df_plot:
                df_plot[v] = df_plot[k]

        return df_plot

    def add_sgRNA_numbers(df_plot):
        sgRNA_numbers = (df_plot
        .groupby(['gene_symbol', 'sgRNA_name'])
        ['dapi_gfp_corr_nuclear'].mean().rename('mean')
        .groupby(['gene_symbol']).rank(ascending=False).astype(int)
        .reset_index().set_index('sgRNA_name')['mean'].to_dict())
        return df_plot.assign(sgRNA_number=df_plot['sgRNA_name'].map(sgRNA_numbers))

    def plot(df_plot, figsize=(8, 4)):
        with plt.rc_context(PanelB.rc_params):
            fig, axs = plt.subplots(figsize=figsize, ncols=len(PanelB.gene_order), sharey=True)
            palette = 'Greens_r'
            for ax, (gene, df) in zip(axs, df_plot.groupby('gene_symbol', sort=False)):
                sns.boxplot(data=df, 
                                x='sgRNA_number',
                                y=labels['dapi_gfp_corr_nuclear'],
                                ax=ax,
                                palette=palette,
                                fliersize=2,
                            )
                ax.set_xlabel(gene + '\nsgRNAs')

            [ax.set_ylabel('') for ax in axs[1:]]
            [ax.yaxis.set_visible(False) for ax in axs[1:]]
            sns.despine(fig=fig, left=True)
            sns.despine(ax=axs[0], left=False)
        return fig


class PanelC:
    """Kinetic data presented as per-cell traces, per-sgRNA averages, and per-gene distributions.
    """
    gene_order = ['TNFRSF1A', 'RIPK1', 'TNFAIP3']
    num_cells = 100 # number of individual cell traces
    num_sg_cells = 500 # sampling for per-sgRNA and per-gene plot
    rc_params = default_rc_params.copy()
    rc_params.update({
        'axes.labelsize': 14,
        'axes.titlesize': 16,
    })

    def prepare_data(df_combined_kinetic):

        df_plot = (df_combined_kinetic
         .assign(t_hr=lambda x: x['t_ms'] / 1000 / 60 / 60)
         .query('gene_symbol == @PanelC.gene_order')
         .pipe(add_fstrings, wtc='{well}_{tile}_{cell_ph_tracked}')
        )

        per_gene = (df_combined_kinetic
         .assign(t_hr=lambda x: x['t_ms'] / 1000 / 60 / 60)
         .groupby(['gene_symbol', 'frame'])
        [['t_hr', 'dapi_gfp_corr']].mean())

        sg_cells = (df_plot
        .query('frame == 0')
        .groupby('gene_symbol')['wtc']
        .apply(lambda x: x.sample(PanelC.num_sg_cells, replace=False, random_state=0))
        .pipe(list)
        )

        per_gene_sgRNA = (df_plot
        .query('wtc == @sg_cells')
        .groupby(['gene_symbol', 'sgRNA', 'frame'])
        .pipe(lambda x:
            pd.concat([
            x['dapi_gfp_corr'].mean(),
            x['dapi_gfp_corr'].sem().rename('dapi_gfp_corr_sem'),
            x['t_hr'].mean(),
            ], axis=1)
        ))

        df_nt = df_combined_kinetic.query('gene_symbol == "nontargeting"')

        return df_plot, df_nt, per_gene, per_gene_sgRNA

    def plot(df_plot, df_nt, per_gene, per_gene_sgRNA, figsize=(7.5, 6)):
        nt_kws = dict(color='gray', lw=3)
        cell_kws = dict(color='green', alpha=0.2)
        sg_kws = dict(color='green', lw=1)
        violin_kws = dict(width=0.95, scale='width', inner=None, palette=['gray', 'green'])
        with plt.rc_context(PanelC.rc_params):
            fig, axes = plt.subplots(figsize=figsize, ncols=3, nrows=3, sharey=True, sharex=True)


            for (gene, df) in df_plot.groupby('gene_symbol'):
                ax0, ax1, ax2 = axes[:, PanelC.gene_order.index(gene)]
                ax0.set_title(gene)
                
                cell_ix = df.groupby('wtc').ngroup()
                cell_ix_plot = (cell_ix.drop_duplicates()
                .sample(PanelC.num_cells, random_state=0, replace=False))
                keep = cell_ix.isin(cell_ix_plot)

                for _, df_ in df[keep].groupby(['well', 'tile', 'cell']):
                    ax0.plot(df_['t_hr'], df_['dapi_gfp_corr'], **cell_kws)
                    
                it = per_gene_sgRNA.loc[gene].groupby('sgRNA')
                for _, df_ in it:
                    ax1.errorbar(df_['t_hr'].values, 
                                df_['dapi_gfp_corr'].values, 
                                df_['dapi_gfp_corr_sem'].values,
                                **sg_kws)
                
                # plot non-targeting
                nt_data = per_gene.loc['nontargeting']    
                ax0.plot(nt_data['t_hr'], nt_data['dapi_gfp_corr'], **nt_kws)
                ax1.plot(nt_data['t_hr'], nt_data['dapi_gfp_corr'], **nt_kws)
                
                df_v = pd.concat([df, df_nt]).query('frame % 2 == 0')
                df_v['t_hr_frame'] = df_v['frame'].map(nt_data['t_hr'])
                order = df_v.groupby('frame')['t_hr'].mean()
                sns.violinplot(data=df_v, x='t_hr_frame', y='dapi_gfp_corr', hue='gene_symbol', 
                            split=True, hue_order=['nontargeting', gene], ax=ax2, **violin_kws)
                
            [ax.set_xlabel(labels['t_hr']) for ax in axes[-1, :]]
            label = 'DAPI:p65\ncorrelation'
            [ax.set_ylabel('') for ax in axes.flat[:]]
            labelpad = -8
            axes[0, 0].set_ylabel('Per-cell', labelpad=labelpad)
            axes[1, 0].set_ylabel('Per-sgRNA', labelpad=labelpad)
            axes[2, 0].set_ylabel('Per-gene', labelpad=labelpad)
            [ax.legend().remove() for ax in axes[2, :]]
            axes[0, 0].set_ylim([-1.1, 1.1])
            axes[0, 0].set_xlim([-0.55, 5.5])
            axes[0, 2].set_xticks([0, 1, 2, 3, 4, 5])
            axes[0, 2].set_xticklabels([0, 1, 2, 3, 4, 5])
            sns.despine(fig)
            
        return fig


class PanelE:
    combined_csv = 'experimentC/process_fig4_all/combined.csv'
    per_sgRNA_rep_csv = 'experimentC/stats_per_sgRNA_rep.csv'
    df_conditions = pd.DataFrame([
            {'well': 'A1', 'stimulant': 'TNFa', 'replicate': 1},
            {'well': 'A2', 'stimulant': 'TNFa', 'replicate': 2},
            {'well': 'A3', 'stimulant': 'TNFa', 'replicate': 3},
    ])
    min_cells_per_sgRNA = 10

    def calculate_sgRNA_rep_stats():
        """Takes about 7 minutes for 3 wells. Slow step is t-tests?
        """
        import ops.screen_stats

        df_combined = (pd.read_csv(PanelE.combined_csv)
         .assign(sgRNA_name=lambda x: x['sgRNA'])
         .merge(PanelE.df_conditions)
         .groupby(['stimulant', 'replicate'])
         .progress_apply(process_rep)
         .reset_index().drop('level_2', axis=1)
         .to_csv(PanelE.per_sgRNA_rep_csv, index=None)
        )
    
    def calculate_gene_stats():
        df_stats = pd.read_csv(PanelE.per_sgRNA_rep_csv)
        df_gene_stats = (df_stats
        .query('count > @PanelE.min_cells_per_sgRNA')
        .sort_values('w_dist', ascending=False)
        .groupby(['stimulant', 'gene_symbol'])
        ['w_dist'].nth(1)
        .sort_values(ascending=False)
        .rename('gene_translocation_defect').reset_index()
        )
        return df_gene_stats


def colorize(data, colors, display_ranges):
    """Example:

    colors = 'white', (0, 1, 0), 'red', 'magenta', 'cyan'
    display_ranges = np.array([
        [100, 3000],
        [700, 5000],
        [600, 3000],
        [600, 4000],
        [600, 3000],
    ])
    rgb = fig4.colorize(data, colors, display_ranges)
    plt.imshow(rgb)
    """
    color_map = np.array([to_rgba(c)[:3] for c in colors])
    dr = display_ranges[..., None, None]
    
    normed = (data - dr[:, 0]) / (dr[:, 1] - dr[:, 0] )
    # there's probably a nicer way to do this
    rgb = (color_map.T[..., None, None] * normed[None, ...]).sum(axis=1)
    return rgb.clip(min=0, max=1).transpose([1, 2, 0])

