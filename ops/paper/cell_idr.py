import fire
import os
import sys
import subprocess
import shutil

import pandas as pd

package_dir = os.path.sep.join(
    os.path.normpath(__file__).split(os.path.sep)[:-3])
aspera_openssh = os.path.join(package_dir, 'resources/asperaweb_id_dsa.openssh')
ascp_guess = os.path.join(os.environ['HOME'], '.aspera/connect/bin/ascp')
cell_idr_files = f'{package_dir}/resources/Cell_IDR_files.csv.gz'


def write_pairlist(df_files, filename, remote_prefix='20191104-gcloud/'):
    """Write list of filenames for use with ascp --file-pair-list option.
    """
    txt = []
    for f in df_files['file']:
        txt += [remote_prefix + f, f]
    txt = '\n'.join(txt)

    with open(filename, 'w') as fh:
        fh.write(txt)


def format_ascp_command(ascp, pairlist, local='.'):
    """Generate Aspera download command. Requires paths to ascp executable and SSH key. 
    See https://idr.openmicroscopy.org/about/download.html
    """
    ascp_opts = f'-T -l200m -P 33001 -i {aspera_openssh}'
    pair_opts = f'--file-pair-list={pairlist} --mode=recv --user=idr0071 --host=fasp.ebi.ac.uk'
    return f'{ascp} {ascp_opts} {pair_opts} {local}'


def get_cell_idr(directory, experiment='C', well='all', tile='all', ascp=ascp_guess):
    """Download data from Cell IDR."""
    os.makedirs(f'{directory}/experiment{experiment}', exist_ok=True)

    if not shutil.which(ascp):
        ascp = shutil.which('ascp')
        if ascp is None:
            print(f'Error: Aspera ascp executable not found at {ascp}')
            raise QuitError

    # select our example
    select_tile = f'idr_name == "experiment{experiment}"'
    if well != 'all':
        select_tile += ' & well == @well'
    if tile != 'all': 
        select_tile += ' & tile == @tile'

    select_image_tags = 'tag == ["phenotype", "sbs"]'    
    df_idr = (pd.read_csv(cell_idr_files, low_memory=False)
     .query(select_tile)
     .query(select_image_tags)
    )

    if df_idr.pipe(len)==0:
        raise ValueError('No valid tiles specified for the chosen experiment.')

    pairlist = f'{directory}/ascp_download_list.txt'
    write_pairlist(df_idr, pairlist)
    command = format_ascp_command(ascp, pairlist, local=directory)

    print(f'Downloading {len(df_idr)} files from Cell-IDR with command: {command}')
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f'Error in downloading files using {ascp}. This can be a user network issue. '
            'Try a secure network with better connectivity.'
            )
        raise QuitError

    well_tile_list = f'{directory}/experiment{experiment}/well_tile_list_example.csv'
    df_idr[['well','tile']].drop_duplicates().to_csv(well_tile_list, index=None)


def setup_example(directory, ascp=ascp_guess, well='A1', tile=102):
    """Create a fresh analysis directory for Cell IDR experiment C (A549 cells, CROPseq library, 
    p65 antibody).
    
    :param ascp: path to ascp executable (download from 
        https://downloads.asperasoft.com/en/downloads/62, see idr_example.ipynb for details)
    :param well: one of A1,A2,A3,B1,B2,B3 or "all"
    :param tile: image tile to download or "all" (see idr_example.ipynb and Cell_IDR_files.csv.gz 
        for available image tiles)
    """
    try:
        os.makedirs(directory)
    except FileExistsError as e:
        print(f'Error: directory already exists at {os.path.abspath(directory)}')
        raise QuitError

    # link resources
    links = {
        'CROPseq_NFkB_library.csv': 'barcodes.csv',
        'experimentC_example.yaml': 'config.yaml',
        'Cell_IDR.smk': 'Snakefile',
    }
    for here, there in links.items():
        os.symlink(f'{package_dir}/resources/{here}', f'{directory}/{there}')
        print(f'Linked {there}')

    get_cell_idr(directory, well=well, tile=tile, ascp=ascp)

    print('Setup complete.\nTo run the example snakemake pipeline, execute the following:')
    print(f'cd {directory}')
    print('snakemake --cores --configfile=config.yaml')


def setup_nature_protocols(directory, ascp=ascp_guess):
    """Setup the analysis used to generate metrics in Fig 3 and Fig 4.
    """
    return setup_example(directory, ascp=ascp, well='all', tile='all')


class QuitError(Exception):
    """Don't generate a stack trace if encountered in command line app.
    """
    pass


if __name__ == '__main__':
    commands = {
        'get_cell_idr': get_cell_idr,
        'setup_example': setup_example,
        'setup_nature_protocols': setup_nature_protocols,
    }
    try:
        fire.Fire(commands)
    except QuitError:
        sys.exit(1)