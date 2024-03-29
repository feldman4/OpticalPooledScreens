import snakemake
import ops.annotate
import ops.firesnake
from ops.firesnake import Snake
import ops.io
import numpy as np
import pandas as pd

cycles = 1,2,3,4,5,6,7
CYCLES = ['c{cycle}-SBS-{cycle}'.format(cycle=c) for c in cycles]

THRESHOLD_READS = 50  # threshold for detecting reads
THRESHOLD_DAPI = 230  # threshold for segmenting nuclei
THRESHOLD_CELL = 150  # threshold for segmenting cells
NUCLEUS_AREA = 50, 800

WELLS, TILES = ops.firesnake.load_well_tile_list('input/well_tile_list.csv')

# .tif file metadata recognized by ImageJ
DISPLAY_RANGES = (
    (100, 800,), 
    (100, 800), 
    (100, 1600), 
    (100, 800), 
    (100, 800))
LUTS = ops.io.GRAY, ops.io.GREEN, ops.io.RED, ops.io.MAGENTA, ops.io.CYAN

BARCODES = pd.read_csv('barcodes.txt', header=None)[0]

CYCLE_PH = 'c0-DAPI-A594'

rule all:
    input:
        # request individual files or list of files
        # 'process/10X_A1_Tile-107.log.tif'
        expand('process/10X_{well}_Tile-{tile}.cells.csv', zip, well=WELLS, tile=TILES),
        expand('process/10X_{well}_Tile-{tile}.phenotype.csv', zip, well=WELLS, tile=TILES),
        expand('process/10X_{well}_Tile-{tile}.annotate_SBS.tif', zip, well=WELLS, tile=TILES),
        # expand('process/10X_{well}_Tile-{tile}.annotate_SBS_extra.tif', zip, well=WELLS, tile=TILES),
        

rule align:
    priority: -1
    input:
        expand('input/10X_{cycle}/10X_{cycle}_{{well}}_Tile-{{tile}}.sbs.tif', cycle=CYCLES)
    output:
        'process/10X_{well}_Tile-{tile}.aligned.tif'
    run:
        Snake.align_SBS(output=output, data=input, 
            display_ranges=DISPLAY_RANGES, luts=LUTS)

rule transform_LoG:
    priority: -1
    input:
        'process/10X_{well}_Tile-{tile}.aligned.tif'
    output:
        'process/10X_{well}_Tile-{tile}.log.tif'
    run:
        display_ranges = [DISPLAY_RANGES[0]] + list(np.array(DISPLAY_RANGES[1:])/4)
        Snake.transform_log(output=output, data=input, skip_index=0,
            display_ranges=display_ranges, luts=LUTS)

rule compute_std:
    input:
        'process/10X_{well}_Tile-{tile}.log.tif'
    output:
        'process/10X_{well}_Tile-{tile}.std.tif'
    run:
        Snake.compute_std(output=output, data=input[0], remove_index=0)

rule find_peaks:
    input:
        'process/10X_{well}_Tile-{tile}.std.tif'
    output:
        'process/10X_{well}_Tile-{tile}.peaks.tif'
    run:
        Snake.find_peaks(output=output, data=input[0]) 

rule max_filter:
    """Dilates sequencing channels to compensate for single-pixel alignment error.
    """
    input:
        'process/10X_{well}_Tile-{tile}.log.tif'
    output:
        'process/10X_{well}_Tile-{tile}.maxed.tif'
    run:
        Snake.max_filter(output=output, data=input[0], width=3,
            remove_index=0, display_ranges=DISPLAY_RANGES[1:], luts=LUTS[1:],
            compress=1) 

rule segment_nuclei:
    input:
        'input/10X_{cycle}/10X_{cycle}_{{well}}_Tile-{{tile}}.sbs.tif'.format(cycle=CYCLES[1]),
        # discarded input, to change run order
        'process/10X_{well}_Tile-{tile}.log.tif'
    output:
        'process/10X_{well}_Tile-{tile}.nuclei.tif'
    run:
        Snake.segment_nuclei(output=output, data=input[0], 
            threshold=THRESHOLD_DAPI, area_min=NUCLEUS_AREA[0], area_max=NUCLEUS_AREA[1])

rule segment_cells:
    input:
        expand('input/10X_{cycle}/10X_{cycle}_{{well}}_Tile-{{tile}}.sbs.tif', cycle=CYCLES[0]),
        'process/10X_{well}_Tile-{tile}.nuclei.tif'
    output:
        'process/10X_{well}_Tile-{tile}.cells.tif'
    run:
        Snake.segment_cells(output=output, 
            data=input[0], nuclei=input[1], threshold=THRESHOLD_CELL)

rule extract_bases:
    input:
        'process/10X_{well}_Tile-{tile}.peaks.tif',
        'process/10X_{well}_Tile-{tile}.maxed.tif',
        'process/10X_{well}_Tile-{tile}.cells.tif'
    output:
        'process/10X_{well}_Tile-{tile}.bases.csv'
    run:
        Snake.extract_bases(output=output, peaks=input[0], maxed=input[1], 
            cells=input[2], threshold_peaks=THRESHOLD_READS, wildcards=dict(wildcards)) 

rule call_reads:
    input:
    #TODO: ADD PEAK INTENSITY TO READS TABLE
        'process/10X_{well}_Tile-{tile}.bases.csv',
        'process/10X_{well}_Tile-{tile}.peaks.tif',
    output:
        'process/10X_{well}_Tile-{tile}.reads.csv'
    run:
        Snake.call_reads(output=output, df_bases=input[0], peaks=input[1])

rule call_cells:
    input:
        'process/10X_{well}_Tile-{tile}.reads.csv'
    output:
        'process/10X_{well}_Tile-{tile}.cells.csv'
    run:
        Snake.call_cells(output=output, df_reads=input[0])

rule annotate_SBS:
    input:
        'process/10X_{well}_Tile-{tile}.log.tif',
        'process/10X_{well}_Tile-{tile}.reads.csv',
    output:
        'process/10X_{well}_Tile-{tile}.annotate_SBS.tif',
    run:
        luts = LUTS + (ops.annotate.GRMC, ops.io.GRAY)
        display_ranges = [DISPLAY_RANGES[0]] + list(np.array(DISPLAY_RANGES[1:])/4) + [[0, 4]]
        Snake.annotate_SBS(output=output, log=input[0], df_reads=input[1], 
            display_ranges=display_ranges, luts=luts, compress=1)

rule annotate_SBS_extra:
    input:
        'process/10X_{well}_Tile-{tile}.log.tif',
        'process/10X_{well}_Tile-{tile}.peaks.tif',
        'process/10X_{well}_Tile-{tile}.reads.csv',
    output:
        'process/10X_{well}_Tile-{tile}.annotate_SBS_extra.tif',
    run:
        luts = LUTS + (ops.annotate.GRMC, ops.io.GRAY, ops.io.GRAY)
        display_ranges = list(np.array(DISPLAY_RANGES)/4)
        display_ranges += [[0, 4], [0, THRESHOLD_READS*4], [0, 30]]
        Snake.annotate_SBS_extra(output=output, log=input[0], peaks=input[1], 
            df_reads=input[2], barcodes=BARCODES, 
            display_ranges=display_ranges[1:], luts=luts[1:], compress=1)


rule rescale_phenotype:
    input:
        ('input/10X_{cycle}/10X_{cycle}_{{well}}_Tile-{{tile}}.phenotype.tif'
            .format(cycle='c0-2x2')),
    output:
        ('input/10X_{cycle}/10X_{cycle}_{{well}}_Tile-{{tile}}.phenotype.tif'
            .format(cycle=CYCLE_PH)),
    run:
        import skimage.transform
        c1 = ops.io.read_stack(input[0])
        c1_ = skimage.transform.rescale(
            c1.transpose([1, 2, 0]), 2,  multichannel=True, 
            preserve_range=True).transpose([2, 0, 1]).astype(c1.dtype)
        ops.io.save_stack(output[0], c1_)


rule align_phenotype:
    input:
        ('input/10X_{cycle}/10X_{cycle}_{{well}}_Tile-{{tile}}.sbs.tif'
            .format(cycle=CYCLES[0])),
        ('input/10X_{cycle}/10X_{cycle}_{{well}}_Tile-{{tile}}.phenotype.tif'
            .format(cycle=CYCLE_PH)),
    output:
        'process/10X_{well}_Tile-{tile}.phenotype_aligned.tif'
    run:
        Snake.align_by_DAPI(output=output, data_1=input[0], data_2=input[1])


rule extract_phenotype:
    input:
        'process/10X_{well}_Tile-{tile}.phenotype_aligned.tif',
        'process/10X_{well}_Tile-{tile}.nuclei.tif',
    output:
        'process/10X_{well}_Tile-{tile}.phenotype.csv',
    run:
        Snake.extract_phenotype_FR(output=output, 
            data_phenotype=input[0], nuclei=input[1],
            wildcards=wildcards)
