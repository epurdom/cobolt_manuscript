
# Rscript --vanilla 20210118_liger_mop.R -k 30 -f F

library("optparse")

option_list = list(
  make_option(c("-k", "--k"), type="integer", default=NULL, 
              help="k"),
  make_option(c("-f", "--isFilter"), type="character", default=NULL, 
              help="isFilter [default= %default]")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);


library(rliger)

k = opt$k
if (opt$isFilter == "T"){
  isFilter = TRUE  
} else if (opt$isFilter %in% c("F", "P", "M")){
  isFilter = FALSE  
}
print(k)
print(isFilter)
print(opt$isFilter)

output_dir = "../../output/method_comparison/"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

source("../read_mop.R")
dt = read_mop(subsample = 10**8, data_dir = "../../data/", filter = isFilter)
cell_dir = "../../data/quality_cells/"
if (opt$isFilter %in% c("P", "M")){
  rna_cf = ifelse(opt$isFilter == "P", 1000, 200)
  atac_cf = ifelse(opt$isFilter == "P", 200, 100)
  rna_cells = read.table(
    file.path(cell_dir, paste0("rna_cells_", rna_cf, ".txt")), stringsAsFactors = FALSE)$V1
  atac_cells = read.table(
    file.path(cell_dir, "atac_cells_", atac_cf, ".txt"), stringsAsFactors = FALSE)$V1
  data_liger <- createLiger(list(
    atac = dt$atac[, colnames(dt$atac) %in% atac_cells], 
    rna = dt$rna[, colnames(dt$rna) %in% rna_cells]))
  isFilter = opt$isFilter
} else {
    data_liger <- createLiger(list(atac = dt$atac, rna = dt$rna))
}

data_liger <- normalize(data_liger)
data_liger <- selectGenes(data_liger, datasets.use = 2)
data_liger <- scaleNotCenter(data_liger)
# suggestK(data_liger, num.cores = 4)
# T:, F:, P:
data_liger <- optimizeALS(data_liger, k=k)
data_liger <- runTSNE(data_liger, use.raw = T)

# int.pbmc <- quantileAlignSNF(int.pbmc) # this is deprecated
data_liger <- quantile_norm(data_liger)

data_liger <- louvainCluster(data_liger, resolution = 0.8)
data_liger <- runTSNE(data_liger)
tnse_res <- data_liger@tsne.coords
data_liger <- runUMAP(data_liger)
umap_res <- data_liger@tsne.coords

cluster_res <- list()
for (resltn in c(seq(0.25, 0.575, 0.025), seq(0.6, 2.5, 0.1))){
  data_liger <- louvainCluster(data_liger, resolution = resltn) #default
  cluster_res[[as.character(resltn)]] <- data_liger@clusters
}

H = data_liger@H
H_norm = data_liger@H.norm

dt$rna_meta$Dataset = "mRNA_MOP"
dt$atac_meta$Dataset = "ATAC_MOP"
meta = rbind(dt$rna_meta, dt$atac_meta)
meta = meta[rownames(umap_res), ]

if (!dir.exists(output_dir)) dir.create(output_dir)
save(data_liger, 
     file = file.path(output_dir, paste0("liger_obj_", k, "_", isFilter, ".RData")))
save(cluster_res, tnse_res, umap_res, H, H_norm, meta,
     file = file.path(output_dir, paste0("liger_res_", k, "_", isFilter, ".RData")))
