

import pandas as pd
import numpy as np
import os
import re
from scipy import io, sparse


snare_seq_files = {
    'dir': '../data/SNAREseq/data/',
    'AdBrainCortex_prefix': 'GSE126074_AdBrainCortex_SNAREseq',
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
    else:
        raise ValueError("dataset not valid")
    if omic not in ['rna', 'atac', 'atac_gene']:
        raise ValueError("omic not valid")
    if file not in ['counts', 'barcode', 'gene', 'peak']:
        raise ValueError("file not valid")
    return prefix + snare_seq_files[omic] + snare_seq_files[file]


def load_snare_seq(dataset):
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
    # filtering
    bool_peak_quality = np.array((np.sum(expression_data_rna != 0, axis=0) > 5) &
                                 (np.sum(expression_data_rna, axis=0) > 10)).flatten()
    gene_names = gene_names[bool_peak_quality]
    expression_data_rna = expression_data_rna[:, bool_peak_quality]
    # --------------------------------- read ATAC data
    path = snare_seq_files['dir']
    counts_filename = _get_file_name(dataset, 'atac', 'counts')
    barcode_filename = _get_file_name(dataset, 'atac', 'barcode')
    feature_filename = _get_file_name(dataset, 'atac', 'peak')
    expression_data_atac = io.mmread(os.path.join(path, counts_filename)).T.tocsr().astype(float)
    feature_df = pd.read_csv(os.path.join(path, feature_filename), header=None)
    peak_names = feature_df[0].values.astype('str')
    barcode_df = pd.read_csv(os.path.join(path, barcode_filename), header=None)
    barcode = barcode_df[0].values.astype('str')
    # filtering
    bool_peak_quality = np.array((np.mean(expression_data_atac != 0, axis=0) <= 0.1) &
                                 (np.sum(expression_data_atac, axis=0) > 5)).flatten()
    peak_names = peak_names[bool_peak_quality]
    expression_data_atac = expression_data_atac[:, bool_peak_quality]
    # reorder
    dict_barcode = dict(zip(barcode, np.arange(len(barcode))))
    barcode_mapping = [dict_barcode[x] for x in barcode_mrna]
    barcode = barcode[barcode_mapping]
    expression_data_atac = expression_data_atac[barcode_mapping, ]
    # --------------------------------- verify
    if not (barcode_mrna == barcode).all():
        raise ValueError("barcode not aligned.")

    counts = {'rna': expression_data_rna, 'atac': expression_data_atac}
    feature = {'rna': gene_names, 'atac': peak_names}
    return counts, feature, barcode


def load_snare_seq_genes(dataset):
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
    # filtering
    bool_peak_quality = np.array((np.sum(expression_data_rna != 0, axis=0) > 5) &
                                 (np.sum(expression_data_rna, axis=0) > 10)).flatten()
    gene_names = gene_names[bool_peak_quality]
    expression_data_rna = expression_data_rna[:, bool_peak_quality]
    # --------------------------------- read ATAC data
    path = snare_seq_files['dir']
    counts_filename = _get_file_name(dataset, 'atac_gene', 'counts')
    barcode_filename = _get_file_name(dataset, 'atac_gene', 'barcode')
    feature_filename = _get_file_name(dataset, 'atac_gene', 'gene')
    expression_data_atac = io.mmread(os.path.join(path, counts_filename)).T.tocsr().astype(float)
    feature_df = pd.read_csv(os.path.join(path, feature_filename), header=None)
    peak_names = feature_df[0].values.astype('str')
    barcode_df = pd.read_csv(os.path.join(path, barcode_filename), header=None)
    barcode = barcode_df[0].values.astype('str')
    # filtering
    bool_peak_quality = np.array((np.sum(expression_data_atac != 0, axis=0) > 5) &
                                 (np.sum(expression_data_atac, axis=0) > 10)).flatten()
    peak_names = peak_names[bool_peak_quality]
    expression_data_atac = expression_data_atac[:, bool_peak_quality]
    # reorder
    dict_barcode = dict(zip(barcode, np.arange(len(barcode))))
    barcode_mapping = [dict_barcode[x] for x in barcode_mrna]
    barcode = barcode[barcode_mapping]
    expression_data_atac = expression_data_atac[barcode_mapping, ]
    # --------------------------------- verify
    if not (barcode_mrna == barcode).all():
        raise ValueError("barcode not aligned.")

    counts = {'rna': expression_data_rna, 'atac': expression_data_atac}
    feature = {'rna': gene_names, 'atac': peak_names}
    return counts, feature, barcode


def load_snare_seq_remapped(dataset="Ad",
                            gene_filter_lower=None, gene_filter_upper=None,
                            peak_filter_lower=None, peak_filter_upper=None,
                            filter_cells=True, filter_features=True):
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
    # filtering
    feature_sum = np.sum(expression_data_rna, axis=0)
    if gene_filter_lower and gene_filter_upper and filter_features:
        bool_peak_quality = np.array((np.sum(expression_data_rna != 0, axis=0) > 5) &
                                     (feature_sum > np.quantile(feature_sum, gene_filter_lower)) &
                                     (feature_sum < np.quantile(feature_sum, gene_filter_upper))).flatten()
        gene_names = gene_names[bool_peak_quality]
        expression_data_rna = expression_data_rna[:, bool_peak_quality]
    elif filter_features:
        bool_peak_quality = np.array((np.sum(expression_data_rna != 0, axis=0) > 5) &
                                     (feature_sum > 10)).flatten()
        gene_names = gene_names[bool_peak_quality]
        expression_data_rna = expression_data_rna[:, bool_peak_quality]

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
    # filtering
    feature_sum = np.sum(expression_data_atac, axis=0)
    if peak_filter_lower and peak_filter_upper and filter_features:
        bool_peak_quality = np.array((np.sum(expression_data_atac != 0, axis=0) > 5) &
                                     (feature_sum > np.quantile(feature_sum, peak_filter_lower)) &
                                     (feature_sum < np.quantile(feature_sum, peak_filter_upper))).flatten()
        peak_names = peak_names[bool_peak_quality]
        expression_data_atac = expression_data_atac[:, bool_peak_quality]
    elif filter_features:
        bool_peak_quality = np.array((np.sum(expression_data_atac != 0, axis=0) > 5) &
                                     (feature_sum > 10)).flatten()
        peak_names = peak_names[bool_peak_quality]
        expression_data_atac = expression_data_atac[:, bool_peak_quality]
    # reorder
    # --------------------------------- verify
    if not (barcode_mrna == barcode).all():
        raise ValueError("barcode not aligned.")

    counts = {'rna': expression_data_rna, 'atac': expression_data_atac}
    feature = {'rna': gene_names, 'atac': peak_names}
    return counts, feature, barcode


def load_mop_atac(subsample=5000,
                  gene_filter_lower=None, gene_filter_upper=None,
                  peak_filter_lower=None, peak_filter_upper=None,
                  filter_cells=True, filter_features=True):
    path = '../../multiomics_clustering/data/mini_atlas/ATAC_MOp_EckerRen/data'
    sample = 'CEMBA171206_3C'
    X = io.mmread(os.path.join(path, sample + '.counts.mtx')).T.tocsr().astype(float)
    peaks = pd.read_csv(os.path.join(path, sample + '.peaks.tsv'), header=None)[0].values.astype('str')
    barcode = pd.read_csv(os.path.join(path, sample + '.barcode.tsv'), header=None)[0].values.astype('str')
    # filtering
    feature_sum = np.sum(X, axis=0)
    if peak_filter_lower and peak_filter_upper and filter_features:
        bool_quality = np.array((np.sum(X != 0, axis=0) > 5) &
                                (feature_sum > np.quantile(feature_sum, peak_filter_lower)) &
                                (feature_sum < np.quantile(feature_sum, peak_filter_upper))).flatten()
        peaks = peaks[bool_quality]
        X = X[:, bool_quality]
    elif filter_features:
        bool_quality = np.array((np.sum(X != 0, axis=0) > 5) & (feature_sum > 10)).flatten()
        peaks = peaks[bool_quality]
        X = X[:, bool_quality]

    # annotation
    if filter_cells:
        anno = pd.read_csv(os.path.join(path, '../metadata/meta_annotated.txt'),
                           sep = "\t")
        barcode_anno = anno['barcode'][anno['sample'] == sample]
        bool_anno = np.isin(barcode, barcode_anno)
        barcode = barcode[bool_anno]
        X = X[bool_anno, ]
    
    if subsample:
        subsample = min(subsample, len(barcode))
        bool_sub = np.random.choice(len(barcode), subsample, replace=False)
        X = X[bool_sub,]
        barcode = barcode[bool_sub]

    counts = {'atac': X}
    feature = {'atac': peaks}
    return counts, feature, barcode


def load_mop_mrna(subsample=5000,
                  gene_filter_lower=None, gene_filter_upper=None,
                  peak_filter_lower=None, peak_filter_upper=None,
                  filter_cells=True, filter_features=True):
    path = '../../multiomics_clustering/data/mini_atlas/10x_nuclei_v3_MOp_Zeng'
    X = io.mmread(os.path.join(path, 'matrix.mtx')).T.tocsr().astype(float)
    genes = pd.read_csv(os.path.join(path, 'features.tsv'), header=None, sep="\t")[1].values.astype('str')
    barcode = pd.read_csv(os.path.join(path, 'barcode.tsv'))['x'].values.astype('str')
    # filtering
    feature_sum = np.sum(X, axis=0)
    if gene_filter_lower and gene_filter_upper and filter_features:
        bool_quality = np.array((np.sum(X != 0, axis=0) > 5) &
                                (feature_sum > np.quantile(feature_sum, gene_filter_lower)) &
                                (feature_sum < np.quantile(feature_sum, gene_filter_upper))).flatten()
        genes = genes[bool_quality]
        X = X[:, bool_quality]
    elif filter_features:
        bool_quality = np.array((np.sum(X != 0, axis=0) > 5) & (feature_sum > 10)).flatten()
        genes = genes[bool_quality]
        X = X[:, bool_quality]
    # annotation
    if filter_cells:
        anno = pd.read_csv(os.path.join(path, "cluster.membership.csv"),
                           header=0, names=['barcode', 'cluster'])
        barcode_anno = anno['barcode']
        bool_anno = np.isin(barcode, barcode_anno)
        barcode = barcode[bool_anno]
        X = X[bool_anno, ]

    if subsample:
        subsample = min(subsample, len(barcode))
        bool_sub = np.random.choice(len(barcode), subsample, replace=False)
        X = X[bool_sub,]
        barcode = barcode[bool_sub]

    counts = {'rna': X}
    feature = {'rna': genes}
    return counts, feature, barcode


def load_snare_seq_ad(subsample=5000):
    return load_snare_seq('Ad')


def load_merged(subsample=5000):
    data = {}
    for f in [load_snare_seq_ad, load_mop_atac, load_mop_mrna]:
        c1, f1, b1 = f(subsample)
        for mod in c1.keys():
            if mod not in data.keys():
                data[mod] = {'feature': [f1[mod]],
                             'barcode': [b1],
                             'counts': [c1[mod]]}
            else:
                data[mod]['feature'] += [f1[mod]]
                data[mod]['barcode'] += [b1]
                data[mod]['counts'] += [c1[mod]]
    data['rna'] = merge_genes(data['rna'])
    data['atac'] = merge_ranges(data['atac'])
    code_dict = {2: 0, 4: 1}
    data['rna']['dataset'] = np.array(
        [code_dict[len(x.split('_'))] for x in data['rna']['barcode']])
    code_dict = {2: 0, 1: 1}
    data['atac']['dataset'] = np.array(
        [code_dict[len(x.split('_'))] for x in data['atac']['barcode']])
    return data


def load_snare_seq_remapped_ad(subsample=5000,
                               gene_filter_lower=None, gene_filter_upper=None,
                               peak_filter_lower=None, peak_filter_upper=None,
                               filter_cells=True, filter_features=True):
    return load_snare_seq_remapped('Ad', gene_filter_lower, gene_filter_upper,
                                   peak_filter_lower, peak_filter_upper,
                                   filter_cells, filter_features)


def load_merged_remapped(subsample=5000,
                         gene_filter_lower=None, gene_filter_upper=None,
                         peak_filter_lower=None, peak_filter_upper=None,
                         filter_cells=True, filter_features=True):
    data = {}
    for f in [load_snare_seq_remapped_ad, load_mop_atac, load_mop_mrna]:
        c1, f1, b1 = f(subsample, gene_filter_lower, gene_filter_upper,
                       peak_filter_lower, peak_filter_upper, filter_cells, filter_features)
        for mod in c1.keys():
            if mod not in data.keys():
                data[mod] = {'feature': [f1[mod]], 'barcode': [b1], 'counts': [c1[mod]]}
            else:
                data[mod]['feature'] += [f1[mod]]
                data[mod]['barcode'] += [b1]
                data[mod]['counts'] += [c1[mod]]
    data['rna'] = merge_genes(data['rna'])
    data['atac'] = merge_genes(data['atac'])
    code_dict = {2: 0, 4: 1}
    data['rna']['dataset'] = np.array(
        [code_dict[len(x.split('_'))] for x in data['rna']['barcode']])
    code_dict = {2: 0, 1: 1}
    data['atac']['dataset'] = np.array(
        [code_dict[len(x.split('_'))] for x in data['atac']['barcode']])
    return data


def merge_genes(dt):
    # TODO: gene names or transcript names?
    # TODO: why is there a gene called 'a'?
    barcode = np.concatenate(dt['barcode'])

    feature = dt['feature'][0]
    for f in dt['feature'][1:]:
        feature = np.intersect1d(feature, f)

    counts = []
    for i in range(len(dt['counts'])):
        common = np.intersect1d(feature, dt['feature'][i], return_indices=True)
        counts += [ dt['counts'][i][:, common[2]] ]
    counts = sparse.vstack(counts)

    return {'feature': feature, 'counts': counts, 'barcode': barcode}
