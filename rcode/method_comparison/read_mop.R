
library(Matrix)

rna_mapping = c(
        "L5 ET" = "L5 PT", "L5 IT" = "L4/5 IT", "L5/6 NP" = "NP",
        "Lamp5" = "CGE", "Vip" = "CGE", "Sncg" = "CGE")

atac_mapping = c(
    "ASC" = "Astro", "L23 a" = "L2/3 IT", "L23 b" = "L2/3 IT",
    "L23 c" = "L2/3 IT", "L4" = "L4/5 IT", "L5 IT a" = "L4/5 IT",
    "L5 IT b" = "L4/5 IT", "Pv" = "Pvalb", "OGC" = "Oligo") # OGC oligodendrocyte

read_rna_meta = function(data_dir = "../data"){
    meta = read.csv(file.path(
        data_dir, "mini_atlas/10x_nuclei_v3_MOp_Zeng/cluster.membership.csv"))
    colnames(meta) = c("Barcode", "cluster_id")
    anno = read.csv(file.path(
        data_dir, "mini_atlas/10x_nuclei_v3_MOp_Zeng/cluster.annotation.csv"))
    meta = merge(meta, anno)
    rownames(meta) = meta$Barcode
    # rename some clusters
    mapping = c(
        "L5 ET" = "L5 PT", "L5 IT" = "L4/5 IT", "L5/6 NP" = "NP",
        "Lamp5" = "CGE", "Vip" = "CGE", "Sncg" = "CGE")
    meta$subclass_label_renamed = meta$subclass_label
    bool_rename = meta$subclass_label %in% names(mapping)
    meta$subclass_label_renamed[bool_rename] = mapping[meta$subclass_label[bool_rename]]
    
    mapping2 = c("L5 ET" = "L5 PT", "L5/6 NP" = "NP")
    bool_rename = meta$subclass_label %in% names(mapping2)
    meta$subclass_label[bool_rename] = mapping2[meta$subclass_label[bool_rename]]
    
    meta = meta[, c("Barcode", "subclass_label", "subclass_label_renamed")]
    colnames(meta) = c("barcode", "celltype_premerge", "celltype")
    meta$Dataset = "mRNA_MOP"
    return(meta)
}


read_atac_meta = function(data_dir, sample = "CEMBA171206_3C"){
    meta = read.table(file.path(
        data_dir, "mini_atlas/ATAC_MOp_EckerRen/metadata/meta_annotated.txt"
        ), header = TRUE)
    meta = meta[meta$sample == sample, ]
    rownames(meta) = meta$barcode
    meta$cluster_annotated = sapply(meta$cluster_annotated, 
                                         function(x) gsub("\\.", " ", x))
    # rename some clusters
    mapping = c(
        "ASC" = "Astro", "L23 a" = "L2/3 IT", "L23 b" = "L2/3 IT",
        "L23 c" = "L2/3 IT", "L4" = "L4/5 IT", "L5 IT a" = "L4/5 IT",
        "L5 IT b" = "L4/5 IT", "Pv" = "Pvalb", "OGC" = "Oligo") # OGC oligodendrocyte
    meta$cluster_renamed = meta$cluster_annotated
    bool_rename = meta$cluster_annotated %in% names(mapping)
    meta$cluster_renamed[bool_rename] = mapping[meta$cluster_renamed[bool_rename]]
    
    mapping2 = c("ASC" = "Astro", "Pv" = "Pvalb", "OGC" = "Oligo")
    bool_rename = meta$cluster_annotated %in% names(mapping2)
    meta$cluster_annotated[bool_rename] = mapping2[meta$cluster_annotated[bool_rename]]
    
    meta = meta[, c("barcode", "cluster_annotated", "cluster_renamed")]
    colnames(meta) = c("barcode", "celltype_premerge", "celltype")
    meta$Dataset = "ATAC_MOP"
    return(meta)
}


get_annotation = function(barcode, rename = TRUE){
    if (!rename){
        use_col = "celltype_premerge"
    } else {
        use_col = "celltype"
    }
    annotation = data.frame(barcode = barcode, Dataset = "", celltype = "Unannotated")
    rownames(annotation) = annotation$barcode
    dataset = sapply(strsplit(annotation$barcode, "_"), length)
    annotation$Dataset[dataset == 4] = "mRNA_MOP"
    annotation$Dataset[dataset == 1] = "ATAC_MOP"
    
    rna_barcode = intersect(rownames(annotation), rownames(rna_meta))
    annotation[rna_barcode, "celltype"] = rna_meta[rna_barcode, use_col]
    
    atac_barcode = intersect(rownames(annotation), rownames(atac_meta))
    annotation[atac_barcode, "celltype"] = atac_meta[atac_barcode, use_col]
    
    return(annotation)
}


read_mop = function(atac_sample = "CEMBA171206_3C", subsample = 10000, 
                    read_peaks = FALSE, data_dir = "../data/", filter = TRUE){
    # ============ read MOP mRNA counts ============
    rna_dir = file.path(data_dir, "mini_atlas/10x_nuclei_v3_MOp_Zeng/")
    rna_counts = readMM(file.path(rna_dir, "matrix.mtx"))
    barcode = read.csv(
        file.path(rna_dir, "barcode.tsv"), stringsAsFactors = FALSE, 
        header = TRUE,  row.names = 1)[, 1]
    gene = read.table(
        file.path(rna_dir, "features.tsv"), stringsAsFactors = FALSE, 
        sep = "\t")$V2
    colnames(rna_counts) = barcode
    rownames(rna_counts) = gene
    if (!(length(barcode) == ncol(rna_counts) & length(gene) == nrow(rna_counts))){
        stop("Dimensions do not match.")
    }
    rna_counts = rna_counts[!duplicated(gene), ]
    rna_meta = read_rna_meta(data_dir)
    if (filter){
        rna_counts = rna_counts[, rna_meta$barcode]
    }
    # ============ read MOP ATAC gene counts ============
    atac_dir = file.path(data_dir, "mini_atlas/ATAC_MOp_EckerRen/data/")
    atac_counts = readMM(file.path(atac_dir, paste0(atac_sample, ".gene_counts.mtx")))
    barcode = read.csv(
        file.path(atac_dir, paste0(atac_sample, ".barcode.tsv")), 
        stringsAsFactors = FALSE, header = FALSE)$V1
    gene = read.table(
        file.path(atac_dir, paste0(atac_sample, ".genes.tsv")), 
        stringsAsFactors = FALSE)$V1
    colnames(atac_counts) = barcode
    rownames(atac_counts) = gene
    if (!(length(barcode) == ncol(atac_counts) & length(gene) == nrow(atac_counts))){
        stop("Dimensions do not match.")
    }
    atac_counts = atac_counts[!duplicated(gene), ]
    atac_meta = read_atac_meta(data_dir)
    if (filter){
        atac_counts = atac_counts[, atac_meta$barcode]
    }
    # ============ read MOP ATAC peak counts ============
    if (read_peaks){
        atac_pcounts = readMM(file.path(atac_dir, paste0(atac_sample, ".counts.mtx")))
        peak = read.table(
            file.path(atac_dir, paste0(atac_sample, ".peaks.tsv")), 
            stringsAsFactors = FALSE)$V1
        colnames(atac_pcounts) = barcode
        rownames(atac_pcounts) = peak
        if (!(length(barcode) == ncol(atac_pcounts) & 
              length(peak) == nrow(atac_pcounts))){
            stop("Dimensions do not match.")
        }
        atac_pcounts = atac_pcounts[!duplicated(peak), ]
        if (filter){
            atac_pcounts = atac_pcounts[, atac_meta$barcode]
        }
    }
    # ============ subsampling ============
    if (length(subsample) == 1 & subsample){
        if (subsample < ncol(atac_counts)){
            sub_idx = sample(ncol(atac_counts), subsample)
            atac_counts = atac_counts[, sub_idx]
            if (read_peaks){
                atac_pcounts = atac_pcounts[, sub_idx]
            }
        }
        if (subsample < ncol(rna_counts)){
            rna_counts = rna_counts[, sample(ncol(rna_counts), subsample)]
        }
    } else if (length(subsample) > 1) {
        atac_counts = atac_counts[, colnames(atac_counts) %in% subsample]
        if (read_peaks){
            atac_pcounts = atac_pcounts[, colnames(atac_pcounts) %in% subsample]
        }
        rna_counts = rna_counts[, colnames(rna_counts) %in% subsample]
    }
    if (read_peaks){
        return(list(atac = atac_counts, rna = rna_counts, peak = atac_pcounts, 
                    atac_meta = atac_meta, rna_meta = rna_meta))
    }
    return(list(atac = atac_counts, rna = rna_counts, 
                atac_meta = atac_meta, rna_meta = rna_meta))
}
