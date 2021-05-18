
library(Matrix)

dat_dir  = c(
        'dir' = '../data/SNAREseq/data/',
        'atac' = 'GSE126074_AdBrainCortex_SNAREseq_chromatin',
        'rna' = 'GSE126074_AdBrainCortex_SNAREseq_cDNA'
)

file_names = c(
        'count' = '.counts.mtx',
        'barcode' = '.barcodes.tsv',
        'rna' = '.genes.tsv',
        'atac' = '.peaks.tsv'
)

read_data_matrix <- function(datatype){
    dir = paste0(dat_dir['dir'], dat_dir[datatype])
    counts = readMM(paste0(dir, file_names['count']))
    features = read.table(paste0(dir, file_names[datatype]),
                          stringsAsFactors = FALSE)$V1
    barcodes = read.table(paste0(dir, file_names['barcode']),
                          stringsAsFactors = FALSE)$V1
    rownames(counts) = features
    colnames(counts) = barcodes
    return(counts)
}
