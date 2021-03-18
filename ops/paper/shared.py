import pandas as pd


def make_barcode_table(path_to_suptable='1-s2.0-S0092867419310670-mmc3.xlsx',
                       path_to_NCBI_HGNC='NCBI_HGNC.csv'):
    """Generate barcode table used in Figures 3,4 from supplemental table in Cell 2019 paper. 
    Requires a table converting Entrez gene IDs to HGNC gene symbols. This barcode table is also
    provided in the repository at `resources/CROPseq_NFkB_library.csv`.
    """
    df_ncbi = pd.read_csv(path_to_NCBI_HGNC)
    gene_id_to_symbol = (df_ncbi
    .dropna(subset=['gene_id'])
    .set_index('gene_id')['gene_symbol'].to_dict())
    gene_id_to_symbol[-1] = 'non-targeting'

    df_sm = (pd.read_excel()
    .query('library == "NF-kB" & vector == "CROPseq"')
    .assign(gene_symbol=lambda x: x['gene_id'].map(gene_id_to_symbol))
    )

    return df_sm[['sgRNA', 'gene_symbol']]
