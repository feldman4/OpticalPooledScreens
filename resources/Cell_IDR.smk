from functools import partial
import ops.annotate
import ops.firesnake
from ops.firesnake import Snake
import ops.io
import pandas as pd

# "well_tile_list.csv" generated automatically when parsing "input_files.xlsx",
# edit to restrict which well and tile positions are analyzed
WELLS, TILES = pd.read_csv(config['WELL_TILE_LIST'])[['well', 'tile']].values.T
SBS_CYCLES = [config['SBS_CYCLE_FORMAT'].format(cycle=x) for x in config['SBS_CYCLES']]

# display options for saved .tif files (view in ImageJ)
channels = ('DAPI', 'SBS_G', 'SBS_T', 'SBS_A', 'SBS_C')
LUTS = [getattr(ops.io, config['LUTS'][x]) for x in channels]
DISPLAY_RANGES = [config['DISPLAY_RANGES'][x] for x in channels]

# naming convention for input and processed files
input_files = partial(ops.firesnake.input_files,
                      magnification=config['MAGNIFICATION'],
                      directory=config['INPUT_DIRECTORY'])

processed_input = partial(ops.firesnake.processed_file, 
                         magnification=config['MAGNIFICATION'],
                         directory=config['PROCESS_DIRECTORY'],
                         )

processed_output = partial(ops.firesnake.processed_file, 
                         magnification=config['MAGNIFICATION'],
                         directory=config['PROCESS_DIRECTORY'],
                         temp_tags=config['TEMP_TAGS'],
                         )

rule all:
    input:
        # request individual files or list of files
        [expand(processed_input(x), zip, well=WELLS, tile=TILES) 
            for x in config['REQUESTED_TAGS']],
        [config['PROCESS_DIRECTORY'] + '/' + x for x in config['REQUESTED_FILES']],

rule align_SBS:
    priority: -1
    input:
        input_files('sbs.tif', SBS_CYCLES)
    output:
        processed_output('aligned.tif')
    run:
        Snake.align_SBS(output=output, data=input, 
            display_ranges=DISPLAY_RANGES, luts=LUTS)

rule align_phenotype:
    input:
        input_files('sbs.tif', SBS_CYCLES[0]),
        input_files('phenotype.tif', config['PHENOTYPE_CYCLE']),
    output:
        processed_output('phenotype_aligned.tif')
    run:
        Snake.align_by_DAPI(output=output, data_1=input[0], data_2=input[1],
            display_ranges=DISPLAY_RANGES, luts=LUTS)


rule transform_LoG:
    priority: -1
    input:
        processed_input('aligned.tif')
    output:
        processed_output('log.tif')
    run:
        Snake.transform_log(output=output, data=input, skip_index=0,
            display_ranges=DISPLAY_RANGES, luts=LUTS)

rule compute_std:
    input:
        processed_input('log.tif')
    output:
        processed_output('std.tif')
    run:
        Snake.compute_std(output=output, data=input[0], remove_index=0)

rule find_peaks:
    input:
        processed_input('std.tif')
    output:
        processed_output('peaks.tif')
    run:
        Snake.find_peaks(output=output, data=input[0]) 

rule max_filter:
    """Dilates sequencing channels to compensate for single-pixel alignment error.
    """
    input:
        processed_input('log.tif')
    output:
        processed_output('maxed.tif')
    run:
        Snake.max_filter(output=output, data=input[0], width=3,
            remove_index=0, display_ranges=DISPLAY_RANGES[1:], luts=LUTS[1:]) 

rule segment_nuclei:
    input:
        input_files('sbs.tif', SBS_CYCLES[0]),
        # not used, just here to change the order of rule execution
        processed_input('log.tif'),
    output:
        processed_output('nuclei.tif')
    run:
        Snake.segment_nuclei(output=output, data=input[0], 
            threshold=config['THRESHOLD_DAPI'], 
            area_min=config['NUCLEUS_AREA'][0], 
            area_max=config['NUCLEUS_AREA'][1], compress=1)

rule segment_cells:
    input:
        input_files('sbs.tif', SBS_CYCLES[0]),
        processed_input('nuclei.tif'),
    output:
        processed_output('cells.tif')
    run:
        Snake.segment_cells(output=output, 
            data=input[0], nuclei=input[1], 
            threshold=config['THRESHOLD_CELL'], compress=1)

rule extract_bases:
    input:
        processed_input('peaks.tif'),
        processed_input('maxed.tif'),
        processed_input('cells.tif'),
    output:
        processed_output('bases.csv')
    run:
        Snake.extract_bases(output=output, peaks=input[0], maxed=input[1], 
            cells=input[2], threshold_peaks=config['THRESHOLD_READS'], wildcards=wildcards)

rule call_reads:
    input:
    #TODO: ADD PEAK INTENSITY TO READS TABLE
        processed_input('bases.csv'),
        processed_input('peaks.tif'),
    output:
        processed_output('reads.csv')
    run:
        Snake.call_reads(output=output, df_bases=input[0], peaks=input[1])

rule call_cells:
    input:
        processed_input('reads.csv')
    output:
        processed_output('cells.csv')
    run:
        Snake.call_cells(output=output, df_reads=input[0])

rule extract_phenotypes:
    input:
        input_files('phenotype.tif', config['PHENOTYPE_CYCLE']),
        processed_input('cells.tif'),
        processed_input('nuclei.tif'),
    output:
        processed_output('phenotype.csv')
    run:
        Snake.extract_named_cell_nucleus_features(output=output, data=input[0], 
            cells=input[1], nuclei=input[2],
            nucleus_features=config['NUCLEUS_PHENOTYPE_FEATURES'],
            cell_features=config['CELL_PHENOTYPE_FEATURES'],
            wildcards=wildcards, 
            )

rule merge_cell_tables:
    input:
        cells=expand(processed_input('cells.csv'), zip, well=WELLS, tile=TILES),
        phenotype=expand(processed_input('phenotype.csv'), zip, well=WELLS, tile=TILES),
        barcodes=config['BARCODE_TABLE'],
    output:
        (config['PROCESS_DIRECTORY'] + '/combined.csv')
    run:
        Snake.merge_sbs_phenotype(output=output, sbs_tables=input.cells, 
        phenotype_tables=input.phenotype, barcode_table=input.barcodes, 
        sbs_cycles=config['SBS_CYCLES'])

rule annotate_SBS:
    input:
        processed_input('log.tif'),
        processed_input('reads.csv'),
    output:
        processed_output('annotate_SBS.tif'),
    run:
        luts = LUTS + [ops.annotate.GRMC, ops.io.GRAY]
        display_ranges = [(a/4, b/4) for a,b in DISPLAY_RANGES] + [[0, 4]]
        Snake.annotate_SBS(output=output, log=input[0], df_reads=input[1], 
            display_ranges=display_ranges, luts=luts, compress=1)

rule annotate_SBS_extra:
    input:
        processed_input('log.tif'),
        processed_input('peaks.tif'),
        processed_input('reads.csv'),
    output:
        processed_output('annotate_SBS_extra.tif'),
    run:
        luts = LUTS + [ops.annotate.GRMC, ops.io.GRAY, ops.io.GRAY]
        display_ranges = [(a/4, b/4) for a,b in DISPLAY_RANGES]
        display_ranges += [[0, 4], [0, config['THRESHOLD_READS']*4], [0, 30]]
        barcodes = pd.read_csv(config['BARCODE_TABLE'])
        
        Snake.annotate_SBS_extra(output=output, log=input[0], peaks=input[1], 
            df_reads=input[2], barcode_table=config['BARCODE_TABLE'], 
            sbs_cycles=config['SBS_CYCLES'],
            display_ranges=display_ranges[1:], luts=luts[1:], compress=1)
