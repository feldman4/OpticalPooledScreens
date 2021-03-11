import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from ..utils import csv_frame
from ..in_situ import dataframe_to_values, transform_medians
from ..plates import add_global_xy

channel_info = [
    ('G', '572 nm', '#0F8040'), 
    ('T', '615 nm', '#ED1E24'), 
    ('A', '680 nm', '#B9519F'), 
    ('C', '732 nm', '#6ECDDD'),
    ]

bases_csv_search = 'experimentC/process_fig4/*bases.csv'
reads_csv_search = 'experimentC/process_fig4/*reads.csv'
barcode_table = 'experimentC/barcodes.csv'
snakemake_config = 'experimentC/config_small_fig3.yaml'

blue = (0.3, 0.3, 0.9)
hex_colors = '#0F8040', '#ED1E24', '#B9519F', '#6ECDDD'

default_rc_params = {
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'font.sans-serif': 'Arial',
}


def load_prefixes():
    """Load sequenced prefixes of sgRNAs. In experiment C one cycle of SBS was skipped.
    """
    df_barcodes = pd.read_csv(barcode_table)
    with open(snakemake_config, 'r') as fh:
        sbs_cycles = yaml.safe_load(fh)['SBS_CYCLES']
    barcode_to_prefix = lambda x: ''.join(x[c - 1] for c in sbs_cycles)
    return df_barcodes['barcode'].apply(barcode_to_prefix).pipe(list)


class PanelA():
    rc_params = default_rc_params.copy()
    rc_params.update({
      'axes.labelpad': 10,
      'font.size': 14,
    })
    colorbar_tick_size = 12

    def plot(figsize=(4.5, 5)):
        df_bases = csv_frame(bases_csv_search)
        order = [2, 3, 0, 1]
        compensation = (df_bases
         .query('cell > 0')
         .pipe(PanelA.get_compensation_matrix)
         .iloc[order, order]
        )

        from matplotlib.colors import to_rgb
        label_colors = [to_rgb(row[2]) for row in channel_info]

        fig = PanelA.plot_compensation_notebook(compensation.values, label_colors, figsize)
        return fig

    def plot_compensation_notebook(compensation, label_colors, figsize):
        """Luke's original colab function
        """
        
        with plt.rc_context(PanelA.rc_params):
            fig, _ = plt.subplots(figsize=figsize)
            im = plt.imshow(compensation,vmin=0, vmax=0.5,cmap='inferno')
            ax = im.axes
            cbar = plt.colorbar(ax=ax,use_gridspec=True,shrink=0.8)
            cbar.ax.tick_params(labelsize=PanelA.colorbar_tick_size)
            for i,row in enumerate(compensation):
                for j,col in enumerate(row):
                    color = 'w' if col <0.4 else 'k'
                    text = ax.text(j, i, "%.3f"%col,
                                    ha="center", va="center", color=color)
            ax.set_xticks([])
            ax.set_yticks([])
                
            xlabel_ax = plt.axes([0.125,0.1675,0.62,0.015])
            xlabel_ax.set_xticks([0,1,2,3])
            xlabel_ax.set_xticklabels(['G','T','A','C'])
            xlabel_ax.set_yticks([])
            xlabel_ax.imshow(np.array(label_colors)[None],aspect='auto')

            ax.set_yticks([0,1,2,3])
            ax.set_yticklabels(['572 nm','615 nm','680 nm','732 nm'])
        return fig

    def get_compensation_matrix(df_bases):    
        channels = df_bases['channel'].value_counts().shape[0]
        X_ = dataframe_to_values(df_bases.query('cell > 0'))
        _, W = transform_medians(X_.reshape(-1, channels))
        compensation = np.linalg.inv(W).T
        df_compensation = pd.DataFrame(compensation)
        channels = df_bases['channel'][:channels].pipe(list)
        df_compensation.columns = channels
        df_compensation.index = [dict([row[:2] for row in channel_info])[x] for x in channels]
        return df_compensation


class PanelB():
    rc_params = default_rc_params.copy()
    rc_params.update({
        'axes.labelpad': 10,
        'legend.fontsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })
    title_fontsize = 16 # axes.titlesize doesn't work with FacetGrid?

    legend_items = {
        'G': '"G" base call',
        'T': '"T" base call',
        'A': '"A" base call',
        'C': '"C" base call',
        'unmapped': 'unmapped reads', 
    }


    def load_intensity_table():
        """Load pre- and post-correction base intensities, annotated with base call and 
        read mapping.
        """

        prefixes = load_prefixes()

        df_bases = csv_frame(bases_csv_search)
        df_reads = (csv_frame(reads_csv_search)
        .assign(mapped=lambda x: x['barcode'].isin(prefixes))
        )
        read_info = df_reads[['well', 'tile', 'read', 'mapped']]

        df_raw = (df_bases
        .pivot_table(index = ['well', 'tile', 'cell', 'read', 'cycle'], 
                    columns='channel', values='intensity')
        .assign(correction='before')
        )

        # estimate spectral correction from reads in cells
        X_ = dataframe_to_values(df_bases.query('cell > 0'))
        _, W = transform_medians(X_.reshape(-1, 4))

        # then apply to all data
        X = dataframe_to_values(df_bases)
        Y = W.dot(X.reshape(-1, 4).T).T.astype(int)

        df_corrected = (pd.DataFrame(Y, columns=list('ACGT'), index=df_raw.index)
        .assign(call=lambda x: x.columns[np.argmax(x.values, axis=1)])
        .assign(correction='after')
        )

        df_plot = (df_raw
        .assign(call=df_corrected['call'])
        .pipe(lambda x: pd.concat([x, df_corrected]))
        .reset_index()
        .merge(read_info)
        .assign(coloring=lambda x: 
          [c if m else 'unmapped' for c,m in x[['call', 'mapped']].values])
        )

        return df_plot

    def scatter_calls(df_plot, calls, cycles=[1, 9], 
                      xlim=(-1000, 7000), ylim=(-1000, 7000), height=2.5):
        """
        """
        colors = {k: v for k,_,v in channel_info}
        palette = ['gray'] + [colors[x] for x in calls]
        assert len(calls) == 2

        with plt.rc_context(PanelB.rc_params):
            fg = (df_plot
            .query('cycle == @cycles & (call == @calls | mapped==False)')
            .pipe(sns.FacetGrid, col='cycle', row='correction',
                hue='coloring', hue_order=['unmapped'] + list(calls), palette=palette,
                height=height,
                )
            .map_dataframe(sns.scatterplot, calls[0], calls[1], style='mapped', style_order=[True,False])
            .set_ylabels(rotation=0)
            .set_titles(template="Cycle {col_name}", size=PanelB.title_fontsize)
            )

        for ax in fg.axes[1, :]:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_title('')
            ax.plot(xlim, ylim, scalex=False, scaley=False, color='k', linestyle='--')

        for ax in fg.axes.flat[:]:
            ax.set_xticks([x for x in ax.get_xticks() if x >= 0])
            ax.set_yticks([x for x in ax.get_yticks() if x >= 0])

        fg.fig.tight_layout()
            
        return fg

    def make_legend(facetgrids, figsize=(2.5, 3)):
        """Collect info from list of FacetGrid objects to make a legend.
        """
        legend_items = PanelB.legend_items
        legend_info = {}
        for fg in facetgrids:
            handles, labels = fg.axes.flat[0].get_legend_handles_labels()
            legend_info.update(dict(zip(labels, handles)))

        handles = [legend_info[x] for x in legend_items]

        with plt.rc_context(PanelB.rc_params):
            fig, ax = plt.subplots()
            leg = fig.legend(handles, legend_items.values(),
                            title='', markerscale=3, frameon=False,
                            bbox_to_anchor=(0,0,1,1), ncol=3)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            sns.despine(ax=ax, left=True, bottom=True)
            fig.tight_layout()
        return fig

    def plot():
        df_plot = PanelB.load_intensity_table()
        fg1 = PanelB.scatter_calls(df_plot, ['A', 'C'], 
                                        xlim=(-1000, 6000), ylim=(-1000, 6000))
        fg2 = PanelB.scatter_calls(df_plot, ['G', 'T'], 
                                        xlim=(-1000, 8000), ylim=(-1000, 13000))
        fig_legend = PanelB.make_legend([fg1, fg2])
        # make these just right
        ax1 = fg1.axes.flat[0]
        ax2 = fg2.axes.flat[0]
        ax1.set_xticks([0, 3000, 6000])
        ax1.set_yticks([0, 2000, 4000, 6000])
        ax2.set_xticks([0, 4000, 8000])
        ax2.set_yticks([0, 4000, 8000, 12000])
        ax2.set_xlim([-1000, 8000])
        ax2.set_ylim([-1000, 12300])
        return fg1, fg2, fig_legend


class PanelC():
    rc_params = {
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.labelpad': 10,
        'legend.fontsize': 14,
        'legend.frameon': False,
        'legend.loc': 'lower center',
        'lines.linewidth': 3,
    }
    labels = {
        'cumulative_mapped': ('Read mapping rate', 'green'), 
        'cumulative_per_cell': ('Mapped reads per cell', blue)
        }

    def plot():
        max_x = 400

        fig = PanelC.plot_mapping()
        ax0, ax1 = fig.axes
        ax0.set_yticks([0.7, 0.75, 0.8, 0.85])
        ax1.set_yticks([0, 1, 2, 3, 4])
        ax0.set_xticks([25, 100, 200, 300, 400])
        ax0.set_xlim([25, max_x])
        
        return fig

    def plot_mapping(figsize=(4.15, 3.2)):
        prefixes = load_prefixes()
        
        search = 'experimentC/process_fig3/*reads.csv'

        key = 'peak'
        xlabel = 'Peak threshold'
        count_cells = lambda x: x.groupby(['well', 'tile'])['cell'].max().sum()

        df_reads = (csv_frame(search)
        .sort_values(key, ascending=False)
        .reset_index(drop=True)
        .assign(mapped=lambda x: x['barcode'].isin(prefixes))           
        .assign(cumulative_mapped=lambda x: x['mapped'].cumsum() / (1 + np.arange(len(x))))
        .assign(cumulative_per_cell=lambda x: x['mapped'].cumsum() / count_cells(x))
        )

        with plt.rc_context(PanelC.rc_params):
            k0 = 'cumulative_mapped'
            k1 = 'cumulative_per_cell'

            fig, ax0 = plt.subplots(figsize=figsize)
            ax1 = ax0.twinx()
            ax0.set_zorder(1)
            ax1.set_zorder(0)
            ax0.patch.set_visible(False)

            df_reads.plot(x=key, y=k0, color=PanelC.labels[k0][1], ax=ax0)
            df_reads.plot(x=key, y=k1, ax=ax1, color=PanelC.labels[k1][1])
            ax0.set_xlim([0, 400])
            ax0.set_ylim([0.7, 0.85])
            ax1.set_ylim([0, 4])

            h0, l0 = ax0.get_legend_handles_labels()
            h1, l1 = ax1.get_legend_handles_labels()
            ax0.get_legend().remove()
            ax1.get_legend().remove()
            new_labels = [PanelC.labels[x][0] for x in l0 + l1]
            ax0.legend(h0 + h1, new_labels, 
                    bbox_to_anchor=(0.04, 0.1, 1, 1),
                    )
            ax0.set_ylabel(PanelC.labels[k0][0])
            ax1.set_ylabel(PanelC.labels[k1][0])

        ax0.set_xlabel(xlabel)
        return ax0.figure


class PanelD():
    rc_params = {
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.labelpad': 10,
        'legend.fontsize': 14,
        'legend.frameon': False,
        'legend.loc': 'lower left',
        'lines.linewidth': 3,
    }

    def plot(figsize=(4.15, 3.2)):

        prefixes = load_prefixes()
        df_reads = (csv_frame(reads_csv_search)
        .assign(mapped=lambda x: x['barcode'].isin(prefixes))           
        .sort_values('peak')
        )
        num_cycles = len(list(prefixes)[0])

        arr = []
        for i in range(1, 10):
            reads = df_reads['barcode'].str[:i]
            reference = set(x[:i] for x in prefixes)
            mapped = reads.isin(reference).mean()
            null = min(len(reference) / 4**i, 1)
            arr += [{'Cycle': i, 'Experiment': mapped, 'Random sequences': null}]

        df_mapping = pd.DataFrame(arr).set_index('Cycle')
        labels = {'Experiment': 'green', 'Random sequences': 'gray'}
        with plt.rc_context(PanelD.rc_params):
            fig, ax = plt.subplots(figsize=figsize)
            for i, label in enumerate(labels):
                df_mapping[label].plot(ax=ax, color=labels[label], x_compat=True, zorder=-i)
            ax.set_xticks(np.arange(1, 10))
            ax.legend(handlelength=1.5)
            ax.set_ylabel('Read mapping rate')
        ax.set_ylim([0, 1.04])
        return fig

class PanelE():
    well_spacing = 38000 # decrease spacing between wells
    label_size_6w = 14
    cmap = 'inferno'

    def label_6w(df_stats, fig, label_pad=9000):
        ax = fig.axes[0]
        x0 = df_stats['global_x'].min() - label_pad
        y0 = df_stats['global_y'].max() + label_pad*0.7

        well_spacing = PanelE.well_spacing
        opts = dict(size=PanelE.label_size_6w, ha='center', va='center')
        ax.text(x0, 0, 'A', **opts)
        ax.text(x0, -well_spacing, 'B', **opts)

        ax.text(0, y0, '1', **opts)
        ax.text(well_spacing, y0, '2', **opts)
        ax.text(well_spacing*2, y0, '3', **opts)

        ax.scatter(0, 0, s=4000, zorder=-1, color='lightgray', marker='s')


    def plot_well(df_stats, col, vmin, vmax, figsize=(4, 3), cbar_pad=0.73, s=5, cbar=True):
        cols = ['well', 'tile', 'global_x', 'global_y']

        fig, ax = plt.subplots(figsize=figsize)
        df_stats.plot(kind='scatter', x='global_x', y='global_y', c=col, edgecolors='none',
                    marker='s', ax=ax, s=s, cmap=PanelE.cmap, vmin=vmin, vmax=vmax)

        ax.axis('equal')
        ax.axis('off')
        ax_cbar = fig.axes[1]
            
        if cbar:
            ax_cbar.set_ylabel('')
            bbox = list(ax_cbar.get_position().bounds)
            bbox[0] = cbar_pad
            bbox[2] *= 0.9
            ax_cbar.set_position(bbox)
        else:
            ax_cbar.remove()

        return fig

    def calculate_stats(df_combined, df_reads):
        cols = ['well', 'tile']
        A = df_combined.groupby(cols).size().rename('cell_count')
        B = df_combined.groupby(cols)['single_mapped'].mean().rename('mapped_to_one')
        C = df_reads.groupby(cols)['mapped'].mean().rename('read_mapping')
        return (pd.concat([A, B, C], axis=1).reset_index()
                .pipe(add_global_xy, well_spacing=PanelE.well_spacing, grid_shape=(25, 25))
                )


