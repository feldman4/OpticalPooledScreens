import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import to_rgba
from skimage.morphology import closing
import itertools

from ..utils import add_fstrings, subimage
from ..in_situ import add_clusters
from ..plates import add_global_xy
from ..imports import read, save, rename, GRAY, GREEN
from ..annotate import annotate_labels, relabel_array
from ..screen_stats import process_rep

labels = {
    'dapi_gfp_corr_nuclear': 'DAPI:p65 nuclear correlation',
    'dapi_gfp_corr': 'DAPI:p65 nuclear correlation',
    't_hr': 'Time post-stimulation (hrs)',
}

def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(f'figures/{name}.png', dpi=200)
    fig.savefig(f'figures/{name}.svg')


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


class PanelA:
    """Activation screen data presented as per-sgRNA distributions.
    """
    gene_order = ['Non-targeting', 'TNFRSF1A', 'MAP3K7', 'IL1R1']
    wells = ['A1', 'A2', 'A3']

    def prepare_data(df_combined, cells_per_sgRNA=100):
            
        df_targeting = (df_combined
         .query('well == @PanelA.wells')
         .query('gene_symbol == @PanelA.gene_order')
         .groupby('sgRNA_name').apply(lambda x: 
           x.sample(cells_per_sgRNA, replace=False, random_state=0) 
              if len(x) > cells_per_sgRNA else x)
         .reset_index(drop=True)
        )

        nontargeting = ['sg_nt213', 'sg_nt393', 'sg_nt8', 'sg_nt562', 'sg_nt759',]
        df_nt = (df_combined
         .query('well == @PanelA.wells')
         .query('sgRNA_name == @nontargeting')
         .assign(gene_symbol='Non-targeting')
         .groupby('sgRNA_name').apply(lambda x: 
           x.sample(cells_per_sgRNA, replace=False, random_state=0))
         .reset_index(drop=True)
        )

        df_plot = (pd.concat([df_nt, df_targeting])
         .pipe(PanelA.add_sgRNA_numbers)
         .sort_values('gene_symbol', key=lambda x: x.apply(PanelA.gene_order.index))
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
        fig, axs = plt.subplots(figsize=figsize, ncols=len(PanelA.gene_order), sharey=True)
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

        [ax.set_ylabel('') for ax in axs[1:]];
        [ax.yaxis.set_visible(False) for ax in axs[1:]];
        sns.despine(fig=fig, left=True)
        sns.despine(ax=axs[0], left=False)
        return fig


class PanelB:
    """Kinetic data presented as per-cell traces, per-sgRNA averages, and per-gene distributions.
    """
    gene_order = ['TNFRSF1A', 'RIPK1', 'TNFAIP3']
    num_cells = 100 # number of individual cell traces
    num_sg_cells = 500 # sampling for per-sgRNA and per-gene plot
    
    def prepare_data(df_combined_kinetic):

        df_plot = (df_combined_kinetic
         .assign(t_hr=lambda x: x['t_ms'] / 1000 / 60 / 60)
         .query('gene_symbol == @PanelB.gene_order')
         .pipe(add_fstrings, wtc='{well}_{tile}_{cell_ph_tracked}')
        )

        per_gene = (df_combined_kinetic
         .assign(t_hr=lambda x: x['t_ms'] / 1000 / 60 / 60)
         .groupby(['gene_symbol', 'frame'])
        [['t_hr', 'dapi_gfp_corr']].mean())

        sg_cells = (df_plot
        .query('frame == 0')
        .groupby('gene_symbol')['wtc']
        .apply(lambda x: x.sample(PanelB.num_sg_cells, replace=False, random_state=0))
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

    def plot(df_plot, df_nt, per_gene, per_gene_sgRNA, figsize=(8, 6)):
        nt_kws = dict(color='gray', lw=3)
        cell_kws = dict(color='green', alpha=0.2)
        sg_kws = dict(color='green', lw=1)
        violin_kws = dict(width=0.95, scale='width', inner=None, palette=['gray', 'green'])

        fig, axes = plt.subplots(figsize=figsize, ncols=3, nrows=3, sharey=True, sharex=True)


        for (gene, df) in df_plot.groupby('gene_symbol'):
            ax0, ax1, ax2 = axes[:, PanelB.gene_order.index(gene)]
            ax0.set_title(gene)
            
            cell_ix = df.groupby('wtc').ngroup()
            cell_ix_plot = (cell_ix.drop_duplicates()
             .sample(PanelB.num_cells, random_state=0, replace=False))
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
        label = 'DAPI:p65 correlation'
        [ax.set_ylabel('') for ax in axes.flat[:]]
        axes[0, 0].set_ylabel('Per-cell\n' + label)
        axes[1, 0].set_ylabel('Per-sgRNA\n' + label)
        axes[2, 0].set_ylabel('Per-gene\n' + label)
        [ax.legend().remove() for ax in axes[2, :]];
        axes[0, 0].set_ylim([-1.1, 1.1])
        axes[0, 0].set_xlim([-0.55, 5.5])
        axes[0, 2].set_xticks([0, 1, 2, 3, 4, 5])
        axes[0, 2].set_xticklabels([0, 1, 2, 3, 4, 5])
        sns.despine(fig)
        return fig


class PanelC:
    """Example results and analysis for one image field of SBS and phenotype data.
    """
    wells = ['A1', 'A2', 'A3']
    well, tile, cluster = 'A1', '456', '69'
    gene = 'TNFRSF1A'
    window = 100
    process_example = 'experimentC/process_fig4/10X_A1_Tile-456.aligned.tif'
    display_ranges = {
        'p65': [400, 3000],
        'DAPI': [100, 3000],
        'SBS_G': [700, 5000],
        'SBS_T': [600, 3000],
        'SBS_A': [600, 4000],
        'SBS_C': [600, 3000],
    }
    
    def find_example(df_combined, gene='TNFRSF1A'):
        """Search for candidate field of view containing a cluster of cells with given gene.
        """
        cluster_size = 5

        df_filtered = (df_combined.query('gene_symbol == @gene & well == @PanelC.wells')
         .pipe(add_global_xy, '6w', (25, 25))
         .query('cell_barcode_1 != cell_barcode_1')
         .pipe(add_clusters, verbose=False)
         .query('cluster > -1')
        )

        barcode_counts = (df_filtered
         .groupby(['well', 'tile', 'cluster', 'cluster_size'])
         ['cell_barcode_count_0'].mean()
         .reset_index()
        )

        return df_filtered, barcode_counts

    def filter_examples(barcode_counts):
        return (barcode_counts
         .query('cluster_size >= 5').sort_values('cell_barcode_count_0')
        )

    def save_example_crop(df_filtered, 
                          save_prefix='experimentC/fig4/example2_',
                          well=None, tile=None, cluster=None):
        """Cell mask missing some, could fix later.
        """
        
        well = PanelC.well if well is None else well
        tile = PanelC.tile if tile is None else tile
        cluster = PanelC.cluster if cluster is None else cluster
        df = (df_filtered
         .query('well == @well & tile == @tile & cluster == @cluster')
        )
        i, j = df[['i', 'j']].mean().astype(int)
        
        # load data
        d = dict(well=well, tile=tile)
        data_sbs = read(rename(process_example, tag='aligned', **d))
        data_ph = read(rename(process_example, tag='phenotype_aligned', **d))
        cells = read(rename(process_example, tag='cells', **d))
        # load labels
        labeled, label_key = (df
         .pipe(annotate_labels, 'cell', 'gene_symbol', label_mask=cells, return_key=True)
        )

        f = f'{save_prefix}{well}_{tile}.tif'
        luts = GRAY, GREEN, GRAY
        annotated = np.array(list(data_ph) + [labeled])
        crop = lambda x: subimage(x, [i, j, i, j], pad=PanelC.window)

        save(f, crop(annotated), luts=luts)

    def load_reprocessed_data(outline_type='label'):
        well = PanelC.well
        tile = PanelC.tile
        window = PanelC.window

        f = 'experimentC/process_fig4/combined.csv'
        df_combined = (pd.read_csv(f)
        .pipe(add_global_xy, '6w', (25, 25))
        .query('well == @well & tile == @tile')
        # prefixes not sequenced as designed
        .drop_duplicates(['cell', 'cell_barcode_0'], keep=False)
        .pipe(add_clusters, 'gene_symbol', 
            ij=('i_nucleus', 'j_nucleus'), verbose=False)
        )

        i, j = (df_combined
        .query('gene_symbol == @PanelC.gene')
        [['i_nucleus', 'j_nucleus']].mean().astype(int)
        )
        crop = lambda x: subimage(x, [i, j, i, j], pad=window)

        f = PanelC.process_example
        d = dict(well=well, tile=tile)
        dapi, p65 = crop(read(rename(f, tag='phenotype_aligned', **d)))
        sbs_stack = crop(read(rename(f, tag='aligned', **d)))
        cells = crop(read(rename(f, tag='cells', **d)))

        selem = np.ones((3, 3))
        cells = closing(cells, selem=selem)

        df_window = (df_combined
        .query('@i - @window < i_nucleus < @i + @window')
        .query('@j - @window < j_nucleus < @j + @window')
        )

        gene_symbols = df_window['gene_symbol'].drop_duplicates().dropna().pipe(list)

        outline, key = (df_combined
        .pipe(annotate_labels, 'cell', 'gene_symbol', 
            label_mask=cells, outline=outline_type, return_key=True))

        reverse_key = {v: k for k,v in key.items()}
        new_labels = {reverse_key[x]: i + 1 for i, x in enumerate(gene_symbols)}
        outline_adj = relabel_array(outline, new_labels).astype(int)

        return df_window, p65, outline_adj, gene_symbols, (i, j), sbs_stack

    def plot_phenotype(df_window, p65, outline_adj, gene_symbols, ij, figsize=(6, 6)):
        i, j = ij
        window = PanelC.window
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = 'red', 'lavender', 'cyan', 'yellow', 'puce', 'orange'
        palette = [to_rgba(sns.xkcd_rgb[x])[:3] 
                for x,_ in zip(itertools.cycle(colors), gene_symbols)]
        palette = [[1, 1, 1]] + palette[:-1]
        outline_palette = np.array([[0, 0, 0]] + palette)
        outline_rgb = outline_palette[outline_adj]
        colors = dict(zip(gene_symbols, palette))

        p65_min, p65_max = PanelC.display_ranges['p65']
        rgb = np.zeros(p65.shape + (3,))
        rgb[..., 1] = (p65.astype(float) - p65_min) / (p65_max - p65_min)
        rgb += outline_rgb
        rgb = rgb.clip(min=0, max=1)

        ax.imshow(rgb)

        for _, df in df_window.groupby('cluster'):
            gene_symbol = df['gene_symbol'].iloc[0]
            if gene_symbol != gene_symbol:
                continue
            color = colors[gene_symbol]
            if gene_symbol == 'non-targeting':
                gene_symbol = 'control'
            

            i_cl, j_cl = df[['i_nucleus', 'j_nucleus']].mean()
            txt = ax.text(j_cl - j + window, -10 + i_cl - i + window, gene_symbol,
                color=color, fontweight='bold', ha='center', fontsize=14)
            effects = [patheffects.withStroke(linewidth=3, foreground=(0, 0, 0, 0.5))]
            txt.set_path_effects(effects)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        return fig, outline_rgb

    def plot_sbs_cycle(data, outline_rgb=None):
        colors = 'white', (0, 1, 0), 'red', 'magenta', 'cyan'
        channels = 'DAPI', 'SBS_G', 'SBS_T', 'SBS_A', 'SBS_C'
        display_ranges = np.array([PanelC.display_ranges[x] for x in channels])

        rgb = colorize(data, colors, display_ranges)
        if outline_rgb is not None:
            mask = outline_rgb > 0
            rgb[mask] = 0
            rgb += outline_rgb

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(rgb)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        return fig

    def outline_rgb_to_gray(outline_rgb, shade=0.5):
        outline_gray = (outline_rgb > 0).any(axis=-1) * shade
        outline_gray = np.tile(outline_gray, (3, 1, 1)).transpose([1, 2, 0])
        return outline_gray

    def plot_base_calls(outline_rgb, ij, figsize=(6, 6)):
        i,j = ij
        crop = lambda x: subimage(x, [i, j, i, j], pad=PanelC.window)

        data = read(rename(PanelC.process_example, tag='aligned'))[0] 
        data = crop(data)
        colors = 'white', (0, 1, 0), 'red', 'magenta', 'cyan'
        channels = 'DAPI', 'SBS_G', 'SBS_T', 'SBS_A', 'SBS_C'
        display_ranges = np.array([PanelC.display_ranges[x] for x in channels])
        sbs_rgb = colorize(data, colors, display_ranges)

        outline_gray = PanelC.outline_rgb_to_gray(outline_rgb, 1)
        f = rename(PanelC.process_example, tag='annotate_SBS')
        base_calls = crop(read(f))[0, -1]
        print(base_calls.shape)
        
        colors = 'black', (0, 1, 0), 'red', 'magenta', 'cyan'
        grmc = np.array([to_rgba(c)[:3] for c in colors])
        grmc_rgb = grmc[base_calls]

        # a bit different
        # f = rename(PanelC.process_example, tag='aligned')
        # dapi = crop(read(f))[0, 0]
        # dr = np.array([PanelC.display_ranges['DAPI']])
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

class PanelD:
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

        df_combined = (pd.read_csv(PanelD.combined_csv)
         .assign(sgRNA_name=lambda x: x['sgRNA'])
         .merge(PanelD.df_conditions)
         .groupby(['stimulant', 'replicate'])
         .progress_apply(process_rep)
         .reset_index().drop('level_2', axis=1)
         .to_csv(PanelD.per_sgRNA_rep_csv, index=None)
        )
    
    def calculate_gene_stats():
        df_stats = pd.read_csv(PanelD.per_sgRNA_rep_csv)
        df_gene_stats = (df_stats
        .query('count > @PanelD.min_cells_per_sgRNA')
        .sort_values('w_dist', ascending=False)
        .groupby(['stimulant', 'gene_symbol'])
        ['w_dist'].nth(1)
        .sort_values(ascending=False)
        .rename('gene_translocation_defect').reset_index()
        )
        return df_gene_stats