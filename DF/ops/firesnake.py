import inspect
import functools
import os
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='regionprops and image moments')
warnings.filterwarnings('ignore', message='non-tuple sequence for multi')
warnings.filterwarnings('ignore', message='precision loss when converting')

import numpy as np
import pandas as pd
import skimage
import skimage.morphology
import ops.annotate
import ops.features
import ops.process
import ops.io
import ops.in_situ
from .process import Align


class Snake():
    """Container class for methods that act directly on data (names start with
    underscore) and methods that act on arguments from snakemake (e.g., filenames
    provided instead of image and table data). The snakemake methods (no underscore)
    are automatically loaded by `Snake.load_methods`.
    """

    # ALIGNMENT AND SEGMENTATION

    @staticmethod
    def _align_SBS(data, method='DAPI', upsample_factor=2, window=2, cutoff=1,
        align_channels=slice(1, None), keep_trailing=False):
        """Rigid alignment of sequencing cycles and channels. 

        Parameters
        ----------

        data : numpy array
            Image data, expected dimensions of (CYCLE, CHANNEL, I, J).

        method : {'DAPI','SBS_mean'}, default 'DAPI'
            Method for aligning 'data' across cycles. 'DAPI' uses cross-correlation between subsequent cycles
            of DAPI images, assumes sequencing channels are aligned to DAPI images. 'SBS_mean' uses the
            mean background signal from the SBS channels to determine image offsets between cycles of imaging,
            again using cross-correlation.

        upsample_factor : int, default 2
            Subpixel alignment is done if `upsample_factor` is greater than one (can be slow).
            Parameter passed to skimage.feature.register_translation.

        window : int, default 2
            A centered subset of data is used if `window` is greater than one. The size of the removed border is
            int((x/2.) * (1 - 1/float(window))).

        cutoff : float, default 1
            Threshold for removing extreme values from SBS channels when using method='SBS_mean'. Channels are normalized
            to the 70th percentile, and normalized values greater than `cutoff` are replaced by `cutoff`.

        align_channels : slice object or None, default slice(1,None)
            If not None, aligns channels (defined by the passed slice object) to each other within each cycle. If
            None, does not align channels within cycles. Useful in particular for cases where images for all stage
            positions are acquired for one SBS channel at a time, i.e., acquisition order of channels(positions).

        keep_trailing : boolean, default True
            If True, keeps only the minimum number of trailing channels across cycles. E.g., if one cycle contains 6 channels,
            but all others have 5, only uses trailing 5 channels for alignment.

        n : int, default 1
            The first SBS channel in `data`.

        Returns
        -------

        aligned : numpy array
            Aligned image data, same dimensions as `data` unless `data` contained different numbers of channels between cycles
            and keep_trailing=True.
        """
        data = np.array(data)
        if keep_trailing:
            valid_channels = min([len(x) for x in data])
            data = np.array([x[-valid_channels:] for x in data])

        assert data.ndim == 4, 'Input data must have dimensions CYCLE, CHANNEL, I, J'

        # align SBS channels for each cycle
        aligned = data.copy()

        if align_channels is not None:
            align_it = lambda x: Align.align_within_cycle(
                x, window=window, upsample_factor=upsample_factor)
            aligned[:, align_channels] = np.array(
                [align_it(x) for x in aligned[:, align_channels]])
            

        if method == 'DAPI':
            # align cycles using the DAPI channel
            aligned = Align.align_between_cycles(aligned, channel_index=0, 
                                window=window, upsample_factor=upsample_factor)
        elif method == 'SBS_mean':
            # calculate cycle offsets using the average of SBS channels
            target = Align.apply_window(aligned[:, n:], window=window).max(axis=1)
            normed = Align.normalize_by_percentile(target)
            normed[normed > cutoff] = cutoff
            offsets = Align.calculate_offsets(normed, upsample_factor=upsample_factor)
            # apply cycle offsets to each channel
            for channel in range(aligned.shape[1]):
                aligned[:, channel] = Align.apply_offsets(aligned[:, channel], offsets)

        return aligned

    @staticmethod
    def _align_by_DAPI(data_1, data_2, channel_index=0, upsample_factor=2):
        """Align the second image to the first, using the channel at position 
        `channel_index`. The first channel is usually DAPI.

        Parameters
        ----------

        data_1 : numpy array
            Image data to align to, expected dimensions of (CHANNEL, I, J).

        data_2 : numpy array
            Image data to align, expected dimensions of (CHANNEL, I, J).

        channel_index : int, default 0
            DAPI channel index

        upsample_factor : int, default 2
            Subpixel alignment is done if `upsample_factor` is greater than one (can be slow).
            Parameter passed to skimage.feature.register_translation.

        Returns
        -------

        aligned : numpy array
            `data_2` with calculated offsets applied to all channels.
        """
        images = data_1[channel_index], data_2[channel_index]
        _, offset = ops.process.Align.calculate_offsets(images, upsample_factor=upsample_factor)
        offsets = [offset] * len(data_2)
        aligned = ops.process.Align.apply_offsets(data_2, offsets)
        return aligned
        
    @staticmethod
    def _segment_nuclei(data, threshold, area_min, area_max, smooth=1.35, radius=15):
        """Find nuclei from DAPI. Uses local mean filtering to find cell foreground from aligned
        but unfiltered data, then filters identified regions by mean intensity threshold and area ranges.

        Parameters
        ----------

        data : numpy array
            Image data, expected dimensions of (CHANNEL, I, J) with the DAPI channel in channel index 0.
            Can also be a single-channel DAPI image of dimensions (I,J).

        threshold : float
            Foreground regions with mean DAPI intensity greater than `threshold` are labeled
            as nuclei.

        area_min, area_max : floats
            After individual nuclei are segmented from foreground using watershed algorithm, nuclei with
            `area_min` < area < `area_max` are retained.

        smooth : float, default 1.35
            Size of gaussian kernel used to smooth the distance map to foreground prior to watershedding.

        radius : float, default 15
            Radius of disk used in local mean thresholding to identify foreground.

        Returns
        -------

        nuclei : numpy array, dtype uint16
            Labeled segmentation mask of nuclei, dimensions are same as trailing two dimensions of `data`.
        """
        if isinstance(data, list):
            dapi = data[0]
        elif data.ndim == 3:
            dapi = data[0]
        else:
            dapi = data

        kwargs = dict(threshold=lambda x: threshold, 
            area_min=area_min, area_max=area_max,
            smooth=smooth, radius=radius)

        # skimage precision warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nuclei = ops.process.find_nuclei(dapi, **kwargs)
        return nuclei.astype(np.uint16)

    @staticmethod
    def _segment_nuclei_stack(dapi, threshold, area_min, area_max, smooth=1.35, radius=15):
        """Find nuclei from a nuclear stain (e.g., DAPI). Expects data to have shape (I, J) 
        (segments one image) or (N, I, J) (segments a series of DAPI images).
        """
        kwargs = dict(threshold=lambda x: threshold, 
            area_min=area_min, area_max=area_max,
            smooth=smooth, radius=radius)

        find_nuclei = ops.utils.applyIJ(ops.process.find_nuclei)
        # skimage precision warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nuclei = find_nuclei(dapi, **kwargs)
        return nuclei.astype(np.uint16)

    @staticmethod
    def _segment_cells(data, nuclei, threshold):
        """Segment cells from aligned data. Matches cell labels to nuclei labels.
        Note that labels can be skipped, for example if cells are touching the 
        image boundary.

        Parameters
        ----------

        data : numpy array
            Image data to use for cell boundary segmentation, expected dimensions of (CYCLE, CHANNEL, I, J),
            (CHANNEL, I, J), or (I,J). Takes minimum intensity over cycles, followed by mean intensity over
            channels if both are present. If channels are present, but not cycles, takes median over channels.

        nuclei : numpy array, dtype uint16
            Labeled segmentation mask of nuclei, dimensions are same as trailing two dimensions of `data`. Uses
            nuclei as seeds for watershedding and matches cell labels to nuclei labels.

        threshold : float
            Threshold used on `data` after reduction to 2 dimensions to identify cell boundaries.

        Returns
        -------

        cells : numpy array, dtype uint16
            Labeled segmentation mask of cell boundaries, dimensions are same as trailing dimensions of `data`.
            Labels match `nuclei` labels.
        """
        if data.ndim == 4:
            # no DAPI, min over cycles, mean over channels
            mask = data[:, 1:].min(axis=0).mean(axis=0)
        elif data.ndim == 3:
            mask = np.median(data[1:], axis=0)
        elif data.ndim == 2:
            mask = data
        else:
            raise ValueError

        mask = mask > threshold
        try:
            # skimage precision warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cells = ops.process.find_cells(nuclei, mask)
        except ValueError:
            print('segment_cells error -- no cells')
            cells = nuclei

        return cells

    # IN SITU

    @staticmethod
    def _transform_log(data, sigma=1, skip_index=None):
        """Apply Laplacian-of-Gaussian filter from scipy.ndimage.

        Parameters
        ----------

        data : numpy array
            Aligned SBS image data, expected dimensions of (CYCLE, CHANNEL, I, J).

        sigma : float, default 1
            size of gaussian kernel used in Laplacian-of-Gaussian filter

        skip_index : None or int, default None
            If an int, skips transforming a channel (e.g., DAPI with `skip_index=0`).

        Returns
        -------

        loged : numpy array
            LoG-ed `data`
        """
        data = np.array(data)
        loged = ops.process.log_ndi(data, sigma=sigma)
        if skip_index is not None:
            loged[..., skip_index, :, :] = data[..., skip_index, :, :]
        return loged

    @staticmethod
    def _compute_std(data, remove_index=None):
        """Use standard deviation over cycles, followed by mean across channels
        to estimate sequencing read locations. If only 1 cycle is present, takes
        standard deviation across channels.

        Parameters
        ----------

        data : numpy array
            LoG-ed SBS image data, expected dimensions of (CYCLE, CHANNEL, I, J).

        remove_index : None or int, default None
            Index of `data` to remove from subsequent analysis, generally any non-SBS channels (e.g., DAPI)

        Returns
        -------

        consensus : numpy array
            Standard deviation score for each pixel, dimensions of (I,J).
        """
        if remove_index is not None:
            data = remove_channels(data, remove_index)

        # for 1-cycle experiments
        if len(data.shape)==3:
            data = data[:,None,...]

        # leading_dims = tuple(range(0, data.ndim - 2))
        # consensus = np.std(data, axis=leading_dims)
        consensus = np.std(data, axis=0).mean(axis=0)

        return consensus
    
    @staticmethod
    def _find_peaks(data, width=5, remove_index=None):
        """Find local maxima and label by difference to next-highest neighboring
        pixel. Conventionally this is used to estimate SBS read locations by inputting
        the standard deviation score as returned by Snake.compute_std().

        Parameters
        ----------

        data : numpy array
            2D image data

        width : int, default 5
            Neighborhood size for finding local maxima.

        remove_index : None or int, default None
            Index of `data` to remove from subsequent analysis, generally any non-SBS channels (e.g., DAPI)

        Returns
        -------

        peaks : numpy array
            Local maxima scores, dimensions same as `data`. At a maximum, the value is max - min in the defined
            neighborhood, elsewhere zero.
        """
        if remove_index is not None:
            data = remove_channels(data, remove_index)

        if data.ndim == 2:
            data = [data]

        peaks = [ops.process.find_peaks(x, n=width) 
                    if x.max() > 0 else x 
                    for x in data]
        peaks = np.array(peaks).squeeze()
        return peaks

    @staticmethod
    def _max_filter(data, width, remove_index=None):
        """Apply a maximum filter in a window of `width`. Conventionally operates on Laplacian-of-Gaussian
        filtered SBS data, dilating sequencing channels to compensate for single-pixel alignment error.

        Parameters
        ----------

        data : numpy array
            Image data, expected dimensions of (..., I, J) with up to 4 total dimenions.

        width : int
            Neighborhood size for max filtering

        remove_index : None or int, default None
            Index of `data` to remove from subsequent analysis, generally any non-SBS channels (e.g., DAPI)

        Returns
        -------

        maxed : numpy array
            Maxed `data` with preserved dimensions.
        """
        import scipy.ndimage.filters

        if data.ndim == 2:
            data = data[None, None]
        if data.ndim == 3:
            data = data[None]

        if remove_index is not None:
            data = remove_channels(data, remove_index)
        
        maxed = scipy.ndimage.filters.maximum_filter(data, size=(1, 1, width, width))
    
        return maxed

    @staticmethod
    def _extract_bases(maxed, peaks, cells, threshold_peaks, wildcards, bases='GTAC'):
        """Find the signal intensity from `maxed` at each point in `peaks` above 
        `threshold_peaks`. Output is labeled by `wildcards` (e.g., well and tile) and 
        label at that position in integer mask `cells`.

        Parameters
        ----------

        maxed : numpy array
            Base intensity at each point, output of Snake.max_filter(), expected dimenions
            of (CYCLE, CHANNEL, I, J).

        peaks : numpy array
            Peaks/local maxima score for each pixel, output of Snake.find_peaks().

        cells : numpy array, dtype uint16
            Labeled segmentation mask of cell boundaries for labeling reads.

        threshold_reads : float
            Threshold for `peaks` for identifying candidate sequencing reads.

        wildcards : dict
            Metadata to include in output table, e.g., well, tile, etc. In Snakemake, use wildcards
            object.

        bases : string, default 'GTAC'
            Order of bases corresponding to the order of acquired SBS channels in `maxed`.

        Returns
        -------

        df_bases : pandas DataFrame
            Table of all candidate sequencing reads with intensity of each base for every cycle,
            (I,J) position of read, and metadata from `wildcards`.
        """
        if maxed.ndim == 3:
            maxed = maxed[None]

        if len(bases) != maxed.shape[1]:
            error = 'Sequencing {0} bases {1} but maxed data had shape {2}'
            raise ValueError(error.format(len(bases), bases, maxed.shape))

        # "cycle 0" is reserved for phenotyping
        cycles = list(range(1, maxed.shape[0] + 1))
        bases = list(bases)

        values, labels, positions = (
            ops.in_situ.extract_base_intensity(maxed, peaks, cells, threshold_peaks))

        df_bases = ops.in_situ.format_bases(values, labels, positions, cycles, bases)

        for k,v in sorted(wildcards.items()):
            df_bases[k] = v

        return df_bases

    @staticmethod
    def _call_reads(df_bases, peaks=None, correction_only_in_cells=True):
        """Call reads by compensating for channel cross-talk and calling the base
        with highest corrected intensity for each cycle. This "median correction"
        is performed independently for each tile.

        Parameters
        ----------

        df_bases : pandas DataFrame
            Table of base intensity for all candidate reads, output of Snake.extract_bases()

        peaks : None or numpy array, default None
            Peaks/local maxima score for each pixel (output of Snake.find_peaks()) to be included
            in the df_reads table for downstream QC or other analysis. If None, does not include
            peaks scores in returned df_reads table.

        correction_only_in_cells : boolean, default True
            If true, restricts median correction/compensation step to account only for reads that
            are within a cell, as defined by the cell segmentation mask passed into
            Snake.extract_bases(). Often identified spots outside of cells are not true sequencing
            reads.

        Returns
        -------

        df_reads : pandas DataFrame
            Table of all reads with base calls resulting from SBS compensation and related metadata.
        """
        if df_bases is None:
            return
        if correction_only_in_cells:
            if len(df_bases.query('cell > 0')) == 0:
                return
        
        cycles = len(set(df_bases['cycle']))
        channels = len(set(df_bases['channel']))

        df_reads = (df_bases
            .pipe(ops.in_situ.clean_up_bases)
            .pipe(ops.in_situ.do_median_call, cycles, channels=channels,
                correction_only_in_cells=correction_only_in_cells)
            )

        if peaks is not None:
            i, j = df_reads[['i', 'j']].values.T
            df_reads['peak'] = peaks[i, j]

        return df_reads

    @staticmethod
    def _call_cells(df_reads, df_pool=None, q_min=0):
        """Call the most-common barcode reads for each cell. If df_pool is supplied,
        prioritizes reads mapping to expected sequences.

        Parameters
        ----------

        df_reads : pandas DataFrame
            Table of all called reads, output of Snake.call_reads()

        df_pool : None or pandas DataFrame, default None
            Table of designed barcode sequences for mapping reads to expected barcodes. Expected
            columns are 'sgRNA', 'gene_symbol', and 'gene_id'.

        q_min : float in the range [0,1)
            Minimum quality score for read inclusion in the cell calling process.

        Returns
        -------

        df_cells : pandas DataFrame
            Table of all cells containing sequencing reads, listing top two most common barcode
            sequences. If df_pool is supplied, prioritizes reads mapping to expected sequences.
        """
        if df_reads is None:
            return
        
        if df_pool is None:
            return (df_reads
                .query('Q_min >= @q_min')
                .pipe(ops.in_situ.call_cells))
        else:
            prefix_length = len(df_reads.iloc[0].barcode) # get the number of completed SBS cycles
            df_pool[PREFIX] = df_pool.apply(lambda x: x.sgRNA[:prefix_length],axis=1)
            return (df_reads
                .query('Q_min >= @q_min')
                .pipe(ops.in_situ.call_cells_mapping,df_pool))

    # PHENOTYPE FEATURE EXTRACTION

    @staticmethod
    def _annotate_SBS(log, df_reads):
        # convert reads to a stack of integer-encoded bases
        cycles, channels, height, width = log.shape
        base_labels = ops.annotate.annotate_bases(df_reads, width=3, shape=(height, width))
        annotated = np.zeros((cycles, channels + 1, height, width), 
                            dtype=np.uint16)

        annotated[:, :channels] = log
        annotated[:, channels] = base_labels
        return annotated

    @staticmethod
    def _annotate_SBS_extra(log, peaks, df_reads, barcode_table, sbs_cycles,
                            shape=(1024, 1024)):
        barcode_to_prefix = lambda x: ''.join(x[c - 1] for c in sbs_cycles)
        barcodes = [barcode_to_prefix(x) for x in barcode_table['barcode']]

        df_reads['mapped'] = df_reads['barcode'].isin(barcodes)
        # convert reads to a stack of integer-encoded bases
        plus = [[0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]]
        xcross = [[1, 0, 1],
                  [0, 1, 0],
                  [1, 0, 1]]
        notch = [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 0]]
        notch2 = [[1, 1, 1],
                 [1, 1, 1],
                 [0, 1, 0]]
        top_right = [[0, 0, 0],
                     [0, 0, 0],
                     [1, 0, 0]]

        f = ops.annotate.annotate_bases
        base_labels  = f(df_reads.query('mapped'), selem=notch)
        base_labels += f(df_reads.query('~mapped'), selem=plus)
        # Q_min converted to 30 point integer scale
        Q_min = ops.annotate.annotate_points(df_reads, 'Q_min', selem=top_right)
        Q_30 = (Q_min * 30).astype(int)
        # a "donut" around each peak indicating the peak intensity
        peaks_donut = skimage.morphology.dilation(peaks, selem=np.ones((3, 3)))
        peaks_donut[peaks > 0] = 0 
        # nibble some more
        peaks_donut[base_labels.sum(axis=0) > 0] = 0
        peaks_donut[Q_30 > 0] = 0

        cycles, channels, height, width = log.shape
        annotated = np.zeros((cycles, 
            channels + 2, 
            # channels + 3, 
            height, width), dtype=np.uint16)

        annotated[:, :channels] = log
        annotated[:, channels] = base_labels
        annotated[:, channels + 1] = peaks_donut
        # annotated[:, channels + 2] = Q_30

        return annotated[:, 1:]

    @staticmethod
    def _extract_features(data, labels, wildcards, features=None):
        """Extracts features in dictionary and combines with generic region
        features.

        Parameters
        ----------

        data : numpy array
            Image data of expected dimensions (CHANNEL, I, J)

        labels : numpy array
            Labeled segmentation mask defining objects to extract features from, dimensions mathcing
            trailing (I,J) dimensions of `data`.

        wildcards : dict
            Metadata to include in output table, e.g., well, tile, etc. In Snakemake, use wildcards
            object.

        features : None or dict of 'key':function, default None
            Features to extract from `data` within `labels` and their definining function calls on an
            skimage regionprops object. E.g., features={'max_intensity':lambda r: r.intensity_image[r.image].max()}.
            Many pre-defined feature functions and dictionaries are available in the features.py module.

        Returns
        -------

        df : pandas DataFrame
            Table of all labeled regions in `labels` and their corresponding `features` measurements from
            `data`.
        """
        from ops.process import feature_table
        from ops.features import features_basic
        features = features.copy() if features else dict()
        features.update(features_basic)

        df = feature_table(data, labels, features)

        for k,v in sorted(wildcards.items()):
            df[k] = v
        
        return df

    @staticmethod
    def _extract_named_features(data, labels, feature_names, wildcards):
        """Extracts features in dictionary and combines with generic region
        features.
        """
        features = ops.features.make_feature_dict(feature_names)
        return Snake._extract_features(data, labels, wildcards, features)

    @staticmethod
    def _extract_named_cell_nucleus_features(data, cells, nuclei, cell_features, nucleus_features,
                                             wildcards, join='inner'):
        """Extract named features for cell and nucleus labels and join the results.
        """
        assert 'label' in cell_features and 'label' in nucleus_features
        df_phenotype = pd.concat([
            Snake._extract_named_features(data, cells, cell_features, {})
                .set_index('label').rename(columns=lambda x: x + '_cell'),
            Snake._extract_named_features(data, nuclei, nucleus_features, {})
                .set_index('label').rename(columns=lambda x: x + '_nucleus'),
        ], join=join, axis=1).reset_index().rename(columns={'label': 'cell'})
        
        for k,v in sorted(wildcards.items()):
            df_phenotype[k] = v

        return df_phenotype

    @staticmethod
    def _extract_phenotype_FR(data_phenotype, nuclei, wildcards):
        """Features for frameshift reporter phenotyped in DAPI, HA channels.
        """
        from ops.features import features_frameshift
        return (Snake._extract_features(data_phenotype, nuclei, wildcards, features_frameshift)
             .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_FR_myc(data_phenotype, nuclei, wildcards):
        """Features for frameshift reporter phenotyped in DAPI, HA, myc channels.
        """
        from ops.features import features_frameshift_myc
        return (Snake._extract_features(data_phenotype, nuclei, wildcards, features_frameshift_myc)
            .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_translocation(data_phenotype, nuclei, cells, wildcards):
        if (nuclei.max() == 0) or (cells.max() == 0):
            return

        import ops.features

        features_n = ops.features.features_translocation_nuclear
        features_c = ops.features.features_translocation_cell

        features_n = {k + '_nuclear': v for k,v in features_n.items()}
        features_c = {k + '_cell': v    for k,v in features_c.items()}

        df_n = (Snake._extract_features(data_phenotype, nuclei, wildcards, features_n)
            .rename(columns={'area': 'area_nuclear'}))

        df_c =  (Snake._extract_features(data_phenotype, cells, wildcards, features_c)
            .drop(['i', 'j'], axis=1).rename(columns={'area': 'area_cell'}))


        # inner join discards nuclei without corresponding cells
        df = (pd.concat([df_n.set_index('label'), df_c.set_index('label')], axis=1, join='inner')
                .reset_index())

        return (df
            .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_translocation_live(data, nuclei, wildcards):
        def _extract_phenotype_translocation_simple(data, nuclei, wildcards):
            import ops.features
            features = ops.features.features_translocation_nuclear_simple
            
            return (Snake._extract_features(data, nuclei, wildcards, features)
                .rename(columns={'label': 'cell'}))

        extract = _extract_phenotype_translocation_simple
        arr = []
        for i, (frame, nuclei_frame) in enumerate(zip(data, nuclei)):
            arr += [extract(frame, nuclei_frame, wildcards).assign(frame=i)]

        return pd.concat(arr)

    @staticmethod
    def _extract_phenotype_translocation_ring(data_phenotype, nuclei, wildcards, width=3):
        selem = np.ones((width, width))
        perimeter = skimage.morphology.dilation(nuclei, selem)
        perimeter[nuclei > 0] = 0

        inside = skimage.morphology.erosion(nuclei, selem)
        inner_ring = nuclei.copy()
        inner_ring[inside > 0] = 0

        return (Snake._extract_phenotype_translocation(data_phenotype, inner_ring, perimeter, wildcards)
            .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_minimal(data_phenotype, nuclei, wildcards):
        return (Snake._extract_features(data_phenotype, nuclei, wildcards, dict())
            .rename(columns={'label': 'cell'}))

    @staticmethod
    def _extract_phenotype_geom(labels, wildcards):
        from ops.features import features_geom
        return Snake._extract_features(labels, labels, wildcards, features_geom)

    @staticmethod
    def _analyze_single(data, alignment_ref, cells, peaks, 
                        threshold_peaks, wildcards, channel_ix=1):
        if alignment_ref.ndim == 3:
            alignment_ref = alignment_ref[0]
        data = np.array([[alignment_ref, alignment_ref], 
                          data[[0, channel_ix]]])
        aligned = ops.process.Align.align_between_cycles(data, 0, window=2)
        loged = Snake._transform_log(aligned[1, 1])
        maxed = Snake._max_filter(loged, width=3)
        return (Snake._extract_bases(maxed, peaks, cells, bases=['-'],
                    threshold_peaks=threshold_peaks, wildcards=wildcards))

    @staticmethod
    def _track_live_nuclei(nuclei, tolerance_per_frame=5):
        
        # if there are no nuclei, we will have problems
        count = nuclei.max(axis=(-2, -1))
        if (count == 0).any():
            error = 'no nuclei detected in frames: {}'
            print(error.format(np.where(count == 0)))
            return np.zeros_like(nuclei)

        import ops.timelapse

        # nuclei coordinates
        arr = []
        for i, nuclei_frame in enumerate(nuclei):
            extract = Snake._extract_phenotype_minimal
            arr += [extract(nuclei_frame, nuclei_frame, {'frame': i})]
        df_nuclei = pd.concat(arr)

        # track nuclei
        motion_threshold = len(nuclei) * tolerance_per_frame
        G = (df_nuclei
          .rename(columns={'cell': 'label'})
          .pipe(ops.timelapse.initialize_graph)
        )

        cost, path = ops.timelapse.analyze_graph(G)
        relabel = ops.timelapse.filter_paths(cost, path, 
                                    threshold=motion_threshold)
        nuclei_tracked = ops.timelapse.relabel_nuclei(nuclei, relabel)

        return nuclei_tracked

    @staticmethod
    def _merge_sbs_phenotype(sbs_tables, phenotype_tables, barcode_table, sbs_cycles, 
                             join='outer'):
        if isinstance(sbs_tables, pd.DataFrame):
            sbs_tables = [sbs_tables]
        if isinstance(phenotype_tables, pd.DataFrame):
            phenotype_tables = [phenotype_tables]
        
        cols = ['well', 'tile', 'cell']
        df_sbs = pd.concat(sbs_tables).set_index(cols)
        df_phenotype = pd.concat(phenotype_tables).set_index(cols)
        df_combined = pd.concat([df_sbs, df_phenotype], join=join, axis=1).reset_index()
        
        barcode_to_prefix = lambda x: ''.join(x[c - 1] for c in sbs_cycles)
        df_barcodes = (barcode_table.assign(prefix=lambda x: 
                            x['barcode'].apply(barcode_to_prefix)))
        if 'barcode' in df_barcodes and 'sgRNA' in df_barcodes:
            df_barcodes = df_barcodes.drop('barcode', axis=1)
        
        barcode_info = df_barcodes.set_index('prefix')
        return (df_combined
                .join(barcode_info, on='cell_barcode_0')
                .join(barcode_info.rename(columns=lambda x: x + '_1'), 
                      on='cell_barcode_1')
                )

    @staticmethod
    def add_method(class_, name, f):
        f = staticmethod(f)
        exec('%s.%s = f' % (class_, name))

    @staticmethod
    def load_methods():
        methods = inspect.getmembers(Snake)
        for name, f in methods:
            if name not in ('__doc__', '__module__') and name.startswith('_'):
                Snake.add_method('Snake', name[1:], Snake.call_from_snakemake(f))

    @staticmethod
    def call_from_snakemake(f):
        """Turn a function that acts on a mix of image data, table data and other 
        arguments and may return image or table data into a function that acts on 
        filenames for image and table data, plus other arguments.

        If output filename is provided, saves return value of function.

        Supported input and output filetypes are .pkl, .csv, and .tif.
        """
        def g(**kwargs):

            # split keyword arguments into input (needed for function)
            # and output (needed to save result)
            input_kwargs, output_kwargs = restrict_kwargs(kwargs, f)

            # load arguments provided as filenames
            input_kwargs = {k: load_arg(v) for k,v in input_kwargs.items()}

            results = f(**input_kwargs)

            if 'output' in output_kwargs:
                outputs = output_kwargs['output']
                
                if len(outputs) == 1:
                    results = [results]

                if len(outputs) != len(results):
                    error = '{0} output filenames provided for {1} results'
                    raise ValueError(error.format(len(outputs), len(results)))

                for output, result in zip(outputs, results):
                    save_output(output, result, **output_kwargs)

        return functools.update_wrapper(g, f)


Snake.load_methods()


def remove_channels(data, remove_index):
    """Remove channel or list of channels from array of shape (..., CHANNELS, I, J).
    """
    channels_mask = np.ones(data.shape[-3], dtype=bool)
    channels_mask[remove_index] = False
    data = data[..., channels_mask, :, :]
    return data


# IO


def load_arg(x):
    """Try loading data from `x` if it is a filename or list of filenames.
    Otherwise just return `x`.
    """
    one_file = load_file
    many_files = lambda x: [load_file(f) for f in x]
    
    for f in one_file, many_files:
        try:
            return f(x)
        except (pd.errors.EmptyDataError, TypeError, IOError) as e:
            if isinstance(e, (TypeError, IOError)):
                # wasn't a file, probably a string arg
                pass
            elif isinstance(e, pd.errors.EmptyDataError):
                # failed to load file
                return None
            pass
    else:
        return x


def save_output(filename, data, **kwargs):
    """Saves `data` to `filename`. Guesses the save function based on the
    file extension. Saving as .tif passes on kwargs (luts, ...) from input.
    """
    filename = str(filename)
    if data is None:
        # need to save dummy output to satisfy Snakemake
        with open(filename, 'w') as fh:
            pass
        return
    if filename.endswith('.tif'):
        return save_tif(filename, data, **kwargs)
    elif filename.endswith('.pkl'):
        return save_pkl(filename, data)
    elif filename.endswith('.csv'):
        return save_csv(filename, data)
    else:
        raise ValueError('not a recognized filetype: ' + f)


def load_csv(filename):
    df = pd.read_csv(filename)
    if len(df) == 0:
        return None
    return df


def load_pkl(filename):
    df = pd.read_pickle(filename)
    if len(df) == 0:
        return None


def load_tif(filename):
    return ops.io.read_stack(filename)


def save_csv(filename, df):
    df.to_csv(filename, index=None)


def save_pkl(filename, df):
    df.to_pickle(filename)


def save_tif(filename, data_, **kwargs):
    kwargs, _ = restrict_kwargs(kwargs, ops.io.save_stack)
    # `data` can be an argument name for both the Snake method and `save_stack`
    # overwrite with `data_` 
    kwargs['data'] = data_
    ops.io.save_stack(filename, **kwargs)


def restrict_kwargs(kwargs, f):
    """Partition `kwargs` into two dictionaries based on overlap with default 
    arguments of function `f`.
    """
    f_kwargs = set(get_kwarg_defaults(f).keys()) | set(get_arg_names(f))
    keep, discard = {}, {}
    for key in kwargs.keys():
        if key in f_kwargs:
            keep[key] = kwargs[key]
        else:
            discard[key] = kwargs[key]
    return keep, discard


def load_file(filename):
    """Attempt to load file, raising an error if the file is not found or 
    the file extension is not recognized.
    """
    if not isinstance(filename, str):
        raise TypeError
    if not os.path.isfile(filename):
        raise IOError(2, 'Not a file: {0}'.format(filename))
    if filename.endswith('.tif'):
        return load_tif(filename)
    elif filename.endswith('.pkl'):
        return load_pkl(filename)
    elif filename.endswith('.csv'):
        return load_csv(filename)
    else:
        raise IOError(filename)


def get_arg_names(f):
    """List of regular and keyword argument names from function definition.
    """
    argspec = inspect.getargspec(f)
    if argspec.defaults is None:
        return argspec.args
    n = len(argspec.defaults)
    return argspec.args[:-n]


def get_kwarg_defaults(f):
    """Get the kwarg defaults as a dictionary.
    """
    argspec = inspect.getargspec(f)
    if argspec.defaults is None:
        defaults = {}
    else:
        defaults = {k: v for k,v in zip(argspec.args[::-1], argspec.defaults[::-1])}
    return defaults


def load_well_tile_list(filename):
    if filename.endswith('pkl'):
        wells, tiles = pd.read_pickle(filename)[['well', 'tile']].values.T
    elif filename.endswith('csv'):
        wells, tiles = pd.read_csv(filename)[['well', 'tile']].values.T
    return wells, tiles


def processed_file(suffix, directory='process', magnification='10X', temp_tags=tuple()):
    """Format output file pattern, for example:
    processed_file('aligned.tif') => 'process/10X_{well}_Tile-{tile}.aligned.tif'
    """
    file_pattern = f'{directory}/{magnification}_{{well}}_Tile-{{tile}}.{suffix}'
    if suffix in temp_tags:
        from snakemake.io import temp
        file_pattern = temp(file_pattern)
    return file_pattern


def input_files(suffix, cycles, directory='input', magnification='10X'):
    from snakemake.io import expand
    pattern = (f'{directory}/{magnification}_{{cycle}}/'
               f'{magnification}_{{cycle}}_{{{{well}}}}_Tile-{{{{tile}}}}.{suffix}')
    return expand(pattern, cycle=cycles)
