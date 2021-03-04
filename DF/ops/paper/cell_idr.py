def write_pairlist(df_files, filename, remote_prefix='20191104-gcloud/'):
    """Write list of filenames for use with ascp --file-pair-list option.
    """
    txt = []
    for f in df_files['file']:
        txt += [remote_prefix + f, f]
    txt = '\n'.join(txt)

    with open(filename, 'w') as fh:
        fh.write(txt)
        
def format_ascp_command(ascp, aspera_openssh, pairlist, local='.'):
    """Generate Aspera download command. Requires paths to ascp executable and SSH key. 
    See https://idr.openmicroscopy.org/about/download.html
    """
    ascp_opts = f'-T -l200m -P 33001 -i {aspera_openssh}'
    pair_opts = f'--file-pair-list={pairlist} --mode=recv --user=idr0071 --host=fasp.ebi.ac.uk'
    return f'{ascp} {ascp_opts} {pair_opts} {local}'
