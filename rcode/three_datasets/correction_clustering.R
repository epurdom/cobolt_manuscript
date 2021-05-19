
# Rscript --vanilla correction.R -k 10 -r M -a 1

library(mclust)
library(clusterExperiment)
library(uwot)
library(Seurat)
library(reshape2)
library(ggplot2)
library(RANN)
library(xgboost)
library(optparse)

option_list = list(
  make_option(c("-k", "--k"), type="character", default=NULL, 
              help="number of dimensions", metavar="character"),
  make_option(c("-r", "--run"), type="character", default=NULL, 
              help="data filtering", metavar="character"),
  make_option(c("-a", "--algoclus"), type="character", default="3", 
              help="clustering algorithm Seurat", metavar="character")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

k = opt$k
ndim = as.numeric(k)
run = opt$run
algoclus = as.numeric(opt$algoclus)

dir = file.path("../../output/", paste0("three_datasets", run), k)
output_dir = file.path("../../output/", 
                       paste0("three_datasets", k), 
                       paste0(run, "_", algoclus))

if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

## ----readBarcode-----------------------------------------
get_barcode <- function(file_name){
    barcode = read.table(
      file.path(dir, file_name), stringsAsFactors = FALSE)$V1
    tmp = sapply(barcode, function(x) length(strsplit(x, "_")[[1]]))
    res = rep("snare-seq", length(barcode))
    res[tmp == 4] = "mop_mrna"
    res[tmp == 1] = "mop_atac"
    res = data.frame(barcode = barcode, omic = res)
    rownames(res) = res$barcode
    return(res)
}

train_barcode <- get_barcode("train_barcode.txt")
barcode <- get_barcode("barcode.txt")


## ----readMeta--------------------------------------------
rna_meta = read.csv("../../data/mini_atlas/10x_nuclei_v3_MOp_Zeng/cluster.membership.csv")
colnames(rna_meta) = c("Barcode", "cluster_id")
rna_anno = read.csv("../../data/mini_atlas/10x_nuclei_v3_MOp_Zeng/cluster.annotation.csv")
rna_meta = merge(rna_meta, rna_anno) # use subclass_label
rownames(rna_meta) = rna_meta$Barcode

atac_meta = read.table("../../data/mini_atlas/ATAC_MOp_EckerRen/metadata/meta_annotated.txt", header = TRUE) # use cluster_annotated
atac_meta = atac_meta[atac_meta$sample == "CEMBA171206_3C", ]
rownames(atac_meta) = atac_meta$barcode
atac_meta$cluster_annotated = sapply(atac_meta$cluster_annotated, 
                                     function(x) gsub("\\.", " ", x))


## ----readLatent, fig.width=12, fig.height=8--------------
f_df = data.frame(
    f = list.files(path = dir, pattern = "*latent.csv"),
    stringsAsFactors = FALSE)
f_df$eval = sapply(f_df$f, function(x) 
    strsplit(gsub("_(test|train)_latent.csv", "", x), "30_2_[0-9]+")[[1]][2])
f_df$data = sapply(f_df$f, function(x) strsplit(x, "_")[[1]][9])
f_df$lr = sapply(f_df$f, function(x) strsplit(x, "_")[[1]][4])
f_df = f_df[f_df$lr == 0, ]


## --------------------------------------------------------
latent = list()
for ( i in 1:nrow(f_df) ){
    f_name = f_df$f[i]
    df = read.csv(file.path(dir, f_name), header = FALSE)
    # Note that here we normalize the data to mean 0
    df = data.frame(t(scale(t(df), center = TRUE, scale = FALSE)))
    df$sample_idx = read.csv(
      file.path(dir, gsub("latent", "barcode", f_name)), header = FALSE)$V1
    df = cbind(df, barcode[df$sample_idx, ])
    df$data = f_df$data[i]
    df$eval = f_df$eval[i]
    
    latent[[i]] = df
}
df = do.call("rbind", latent)
set.seed(2020)
df = df[sample.int(nrow(df)), ]
df$sample_idx = NULL


## --------------------------------------------------------
snare_rna = df[(df$omic == "snare-seq") & (df$eval == "True_False"), ]
rownames(snare_rna) = snare_rna$barcode
snare_rna = snare_rna[, 1:ndim]
snare_barcode = rownames(snare_rna)

snare_atac = df[(df$omic == "snare-seq") & (df$eval == "False_True"), ]
rownames(snare_atac) = snare_atac$barcode
snare_atac = snare_atac[, 1:ndim]
snare_atac = snare_atac[snare_barcode, ]

snare_joint = df[(df$omic == "snare-seq") & (df$eval == "True_True"), ]
rownames(snare_joint) = snare_joint$barcode
snare_joint = snare_joint[, 1:ndim]
snare_joint = snare_joint[snare_barcode, ]

if (!all(rownames(snare_rna) == rownames(snare_atac)) | 
    !all(rownames(snare_rna) == rownames(snare_atac)))
    stop("Barcode matching error.")

mop_rna = df[(df$omic == "mop_mrna"), 1:ndim]
mop_atac = df[(df$omic == "mop_atac"), 1:ndim]


## modeling ------------------------------------------------
n_dim = ndim

pred_xgboost = function(pred_var, depend_var, new_data){
    pred = list()
    for (i in 1:(n_dim-1)){
      fit <- xgboost(
        data = as.matrix(pred_var), 
        label = as.numeric(depend_var[, i]), 
        max_depth = 3, 
        eta = 0.8,
        nrounds = 100, 
        objective = "reg:squarederror",
        verbose = 0
      )
      pred[[i]] = predict(fit, as.matrix(new_data))
    }
    pred = as.data.frame(do.call("cbind", pred))
    rownames(pred) = rownames(new_data)
    return(pred)
}

mop_rna_adj = pred_xgboost(
  pred_var=snare_rna, depend_var=snare_joint, new_data=mop_rna)
mop_atac_adj = pred_xgboost(
  pred_var=snare_atac, depend_var=snare_joint, new_data=mop_atac)


## --------------------------------------------------------
mop_rna_adj$barcode = rownames(mop_rna_adj)
mop_rna_adj$omic = "mop_mrna"
mop_rna_adj$data = "train"
mop_rna_adj$eval = "True_False"

mop_atac_adj$barcode = rownames(mop_atac_adj)
mop_atac_adj$omic = "mop_atac"
mop_atac_adj$data = "train"
mop_atac_adj$eval = "False_True"

df = rbind(df[df$omic == "snare-seq", -n_dim], mop_rna_adj, mop_atac_adj)

save(df, file = file.path(output_dir, "xgboost.RData"))

## --------------------------------------------------------
bool_exclude = (df$omic == "snare-seq") & (df$eval %in% c("True_False", "False_True"))
df = df[!bool_exclude, ]
# =============== umap and clustering with snare-seq
rna <- CreateSeuratObject(
  counts = round(exp(t(df[, 1:(n_dim-1)]))), # place holder, we will not use this
  assay = "RNA", project = 'RNA')
rna@reductions$ours = CreateDimReducObject(
  embeddings = as.matrix(df[, 1:(n_dim-1)]), key = "DIM_", assay = "RNA")
rna <- RunUMAP(rna, reduction = "ours", dims = 1:(n_dim-1))
rna <- FindNeighbors(rna, reduction = "ours", dims = 1:(n_dim-1))

clus_res <- list()
for (resltn in c(seq(0.1, 0.4, 0.05), seq(0.5, 2, 0.1)) ){
  rna <- FindClusters(rna, resolution = resltn, algorithm = 3)
  clus_res[[as.character(resltn)]] = rna$seurat_clusters
}

umap_res <- rna@reductions$umap@cell.embeddings

clus_res_full = clus_res
df_full = df
umap_res_full = umap_res

bool_mop = df$omic != "snare-seq"
for (i in 1:length(clus_res)){
  clus_res[[i]] = clus_res[[i]][bool_mop]
}
umap_res = umap_res[bool_mop, ]
df = df[bool_mop, ]


save(rna, bool_mop,
     file = file.path(output_dir, "obj_snare_full.RData"))

save(
  umap_res, clus_res, df, clus_res_full, df_full, umap_res_full,
  file = file.path(output_dir, "xgboost_cluster_res_snare_full.RData")
)

## --------------------------------------------------------
df = df[df$omic != "snare-seq", ]

## --------------------------------------------------------
# =============== umap and clustering without snare-seq
rna <- CreateSeuratObject(
  counts = round(exp(t(df[, 1:(n_dim-1)]))), # place holder, we will not use this
  assay = "RNA", project = 'RNA')
rna@reductions$ours = CreateDimReducObject(
  embeddings = as.matrix(df[, 1:(n_dim-1)]), key = "DIM_", assay = "RNA")
rna <- RunUMAP(rna, reduction = "ours", dims = 1:(n_dim-1))
rna <- FindNeighbors(rna, reduction = "ours", dims = 1:(n_dim-1))

clus_res <- list()
for (resltn in c(seq(0.1, 0.4, 0.05), seq(0.5, 2, 0.1)) ){
  rna <- FindClusters(rna, resolution = resltn, algorithm = algoclus)
  clus_res[[as.character(resltn)]] = rna$seurat_clusters
}

umap_res <- rna@reductions$umap@cell.embeddings

save(rna, 
     file = file.path(output_dir, "obj.RData"))

save(
    umap_res, clus_res, df, 
    file = file.path(output_dir, "xgboost_cluster_res.RData")
)
