
import pandas as pd
import numpy as np
import os
import re
from scipy import io, sparse
import pyranges as pr


snare_seq_files = {
    'dir': '../data/SNAREseq/data/',
    'AdBrainCortex_prefix': 'GSE126074_AdBrainCortex_SNAREseq',
    'P0_BrainCortex_prefix': 'GSE126074_P0_BrainCortex_SNAREseq',
    'rna': '_cDNA',
    'atac': '_chromatin',
    'atac_gene': '_chromgene',
    'counts': '.counts.mtx',
    'barcode': '.barcodes.tsv',
    'gene': '.genes.tsv',
    'peak': '.peaks.tsv',
}


def _get_file_name(dataset, omic, file):
    if dataset == 'Ad':
        prefix = snare_seq_files['AdBrainCortex_prefix']
    elif dataset == 'P0':
        prefix = snare_seq_files['P0_BrainCortex_prefix']
    else:
        raise ValueError("dataset not valid")
    if omic not in ['rna', 'atac', 'atac_gene']:
        raise ValueError("omic not valid")
    if file not in ['counts', 'barcode', 'gene', 'peak']:
        raise ValueError("file not valid")
    return prefix + snare_seq_files[omic] + snare_seq_files[file]


out_path = "/home/kinsley/projects/cobolt_pkg/test_data"

dataset = "Ad"
# -------------------------------- read mRNA data
path = snare_seq_files['dir']
counts_filename = _get_file_name(dataset, 'rna', 'counts')
barcode_filename = _get_file_name(dataset, 'rna', 'barcode')
feature_filename = _get_file_name(dataset, 'rna', 'gene')
expression_data_rna = io.mmread(os.path.join(path, counts_filename)).T.tocsr().astype(float)
feature_df = pd.read_csv(os.path.join(path, feature_filename), header=None)
gene_names = feature_df[0].values.astype('str')
barcode_df = pd.read_csv(os.path.join(path, barcode_filename), header=None)
barcode_mrna = barcode_df[0].values.astype('str')
# --------------------------------- read ATAC data
path = '../../multiomics_clustering/data/mini_atlas/ATAC_MOp_EckerRen/peakbed'
# counts_filename = "CEMBA171206_3C.mtx"
# barcode_filename = "CEMBA171206_3C_barcode.tsv"
# feature_filename = "CEMBA171206_3C.peaks.tsv"
counts_filename = "counts.mtx"
barcode_filename = "barcodes.tsv"
feature_filename = "peaks.tsv"
expression_data_atac = io.mmread(os.path.join(path, counts_filename)).T.tocsr().astype(float)
feature_df = pd.read_csv(os.path.join(path, feature_filename), header=None)
peak_names = feature_df[0].values.astype('str')
barcode_df = pd.read_csv(os.path.join(path, barcode_filename), header=None)
barcode = barcode_df[0].values.astype('str')
# --------------------------------- verify
if not (barcode_mrna == barcode).all():
    raise ValueError("barcode not aligned.")

n_sample = 100
n_gene = 100
n_peak = 500

example_barcode = barcode[:n_sample]

np.savetxt(os.path.join(out_path, "snare", "barcodes.tsv"), example_barcode, fmt="%s")
np.savetxt(os.path.join(out_path, "snare", "genes.tsv"), gene_names[:n_gene], fmt="%s")
np.savetxt(os.path.join(out_path, "snare", "peaks.tsv"), peak_names[:n_peak], fmt="%s")
io.mmwrite(os.path.join(out_path, "snare", "gene_counts.mtx"), expression_data_rna[:n_sample, :n_gene].T.astype(int))
io.mmwrite(os.path.join(out_path, "snare", "peak_counts.mtx"), expression_data_atac[:n_sample, :n_peak].T.astype(int))

# =============================== read mop mrna ===============================
path = '../../multiomics_clustering/data/mini_atlas/10x_nuclei_v3_MOp_Zeng'
X = io.mmread(os.path.join(path, 'matrix.mtx')).T.tocsr().astype(float)
genes = pd.read_csv(os.path.join(path, 'features.tsv'), header=None, sep="\t")[1].values.astype('str')
barcode = pd.read_csv(os.path.join(path, 'barcode.tsv'))['x'].values.astype('str')

bool_gene = np.isin(genes, gene_names[:n_gene])
bool_gene[:30] = True

np.savetxt(os.path.join(out_path, "mrna", "barcodes.tsv"), barcode[:n_sample], fmt="%s")
np.savetxt(os.path.join(out_path, "mrna", "genes.tsv"), genes[bool_gene], fmt="%s")
io.mmwrite(os.path.join(out_path, "mrna", "counts.mtx"), X[:n_sample, bool_gene].T.astype(int))


# =============================== read mop atac ===============================
path = '../../multiomics_clustering/data/mini_atlas/ATAC_MOp_EckerRen/data'
sample = 'CEMBA171206_3C'
X = io.mmread(os.path.join(path, sample + '.counts.mtx')).T.tocsr().astype(float)
peaks = pd.read_csv(os.path.join(path, sample + '.peaks.tsv'), header=None)[0].values.astype('str')
barcode = pd.read_csv(os.path.join(path, sample + '.barcode.tsv'), header=None)[0].values.astype('str')

bool_peak = np.isin(peaks, peak_names[:n_peak])
bool_peak[10000:10010] = True

np.savetxt(os.path.join(out_path, "atac", "barcodes.tsv"), barcode[:n_sample], fmt="%s")
np.savetxt(os.path.join(out_path, "atac", "peaks.tsv"), peaks[bool_peak], fmt="%s")
io.mmwrite(os.path.join(out_path, "atac", "counts.mtx"), X[:n_sample, bool_peak].T.astype(int))
