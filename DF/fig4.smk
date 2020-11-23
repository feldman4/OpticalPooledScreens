configfile: "config.yaml"

from functools import partial
import ops.annotate
import ops.firesnake
from ops.firesnake import Snake
import ops.io
import pandas as pd

# "well_tile_list.csv" generated automatically when parsing "input_files.xlsx",
# edit to restrict which well and tile positions are analyzed
WELLS, TILES = pd.read_csv(config['WELL_TILE_LIST'])[['well', 'tile']].values.T
BARCODES = pd.read_csv(config['BARCODE_LIST'], header=None)[0]

# display options for saved .tif files (view in ImageJ)
channels = ('DAPI', 'SBS_G', 'SBS_T', 'SBS_A', 'SBS_C')
LUTS = [getattr(ops.io, config['LUTS'][x]) for x in channels]
DISPLAY_RANGES = [config['DISPLAY_RANGES'][x] for x in channels]

# naming convention for input and processed files
input_files = partial(ops.firesnake.input_files,
                      magnification=config['MAGNIFICATION'],
                      directory=config['INPUT_DIRECTORY'])

processed_file = partial(ops.firesnake.processed_file, 
                         magnification=config['MAGNIFICATION'],
                         directory=config['PROCESS_DIRECTORY'])


rule all:
    input:
        # request individual files or list of files
        [expand(processed_file(x), zip, well=WELLS, tile=TILES)
         for x in config['REQUESTED_FILES']]

rule align:
    priority: -1
    input:
        input_files('sbs.tif', config['SBS_CYCLES'])
    output:
        processed_file('aligned.tif')
    run:
        Snake.align_SBS(output=output, data=input, 
            display_ranges=DISPLAY_RANGES, luts=LUTS)

rule transform_LoG:
    priority: -1
    input:
        processed_file('aligned.tif')
    output:
        processed_file('log.tif')
    run:
        Snake.transform_log(output=output, data=input, skip_index=0,
            display_ranges=DISPLAY_RANGES, luts=LUTS)

rule compute_std:
    input:
        processed_file('log.tif')
    output:
        processed_file('std.tif')
    run:
        Snake.compute_std(output=output, data=input[0], remove_index=0)

rule find_peaks:
    input:
        processed_file('std.tif')
    output:
        processed_file('peaks.tif')
    run:
        Snake.find_peaks(output=output, data=input[0]) 

rule max_filter:
    """Dilates sequencing channels to compensate for single-pixel alignment error.
    """
    input:
        processed_file('log.tif')
    output:
        processed_file('maxed.tif')
    run:
        Snake.max_filter(output=output, data=input[0], width=3,
            remove_index=0, display_ranges=DISPLAY_RANGES[1:], luts=LUTS[1:]) 

rule segment_nuclei:
    input:
        input_files('sbs.tif', [config['SBS_CYCLES'][0]]),
        # not used, just here to change the order of rule execution
        processed_file('log.tif'),
    output:
        processed_file('nuclei.tif')
    run:
        Snake.segment_nuclei(output=output, data=input[0], 
            threshold=config['THRESHOLD_DAPI'], 
            area_min=config['NUCLEUS_AREA'][0], 
            area_max=config['NUCLEUS_AREA'][1])

rule segment_cells:
    input:
        input_files('sbs.tif', config['SBS_CYCLES'][0]),
        processed_file('nuclei.tif'),
    output:
        processed_file('cells.tif')
    run:
        Snake.segment_cells(output=output, 
            data=input[0], nuclei=input[1], threshold=config['THRESHOLD_CELL'])

rule extract_bases:
    input:
        processed_file('peaks.tif'),
        processed_file('maxed.tif'),
        processed_file('cells.tif'),
    output:
        processed_file('bases.csv')
    run:
        Snake.extract_bases(output=output, peaks=input[0], maxed=input[1], 
            cells=input[2], threshold_peaks=config['THRESHOLD_READS'], wildcards=dict(wildcards)) 

rule call_reads:
    input:
    #TODO: ADD PEAK INTENSITY TO READS TABLE
        processed_file('bases.csv'),
        processed_file('peaks.tif'),
    output:
        processed_file('reads.csv')
    run:
        Snake.call_reads(output=output, df_bases=input[0], peaks=input[1])

rule call_cells:
    input:
        processed_file('reads.csv')
    output:
        processed_file('cells.csv')
    run:
        Snake.call_cells(output=output, df_reads=input[0])

rule annotate_SBS:
    input:
        processed_file('log.tif'),
        processed_file('reads.csv'),
    output:
        processed_file('annotate_SBS.tif'),
    run:
        luts = LUTS + [ops.annotate.GRMC, ops.io.GRAY]
        display_ranges = [(a/4, b/4) for a,b in DISPLAY_RANGES] + [[0, 4]]
        Snake.annotate_SBS(output=output, log=input[0], df_reads=input[1], 
            display_ranges=display_ranges, luts=luts, compress=1)

rule annotate_SBS_extra:
    input:
        processed_file('log.tif'),
        processed_file('peaks.tif'),
        processed_file('reads.csv'),
    output:
        processed_file('annotate_SBS_extra.tif'),
    run:
        luts = LUTS + [ops.annotate.GRMC, ops.io.GRAY, ops.io.GRAY]
        display_ranges = [(a/4, b/4) for a,b in DISPLAY_RANGES]
        display_ranges += [[0, 4], [0, config['THRESHOLD_READS']*4], [0, 30]]
        Snake.annotate_SBS_extra(output=output, log=input[0], peaks=input[1], 
            df_reads=input[2], barcodes=BARCODES, 
            display_ranges=display_ranges[1:], luts=luts[1:], compress=1)
