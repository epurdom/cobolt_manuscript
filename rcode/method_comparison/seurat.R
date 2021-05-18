
# sbatch --wrap="R CMD BATCH --no-save run_seurat.R run_seurat.Rout"

library(Seurat)
library(Matrix)

source("datasets_info.R")

read_seurat <- function(datatype){
    counts = read_data_matrix(datatype)
    cat("Nonzero counts ", mean((counts!=0)), 
        ", one counts ", mean((counts==1)), ".\n", sep = "")
    seu = CreateSeuratObject(counts)
    return(seu)
}

run_seurat <- function(seurat_object){
    # ================ filter cells ================ 
    seurat_object
    seurat_object <- subset(
        seurat_object, 
        subset = nFeature_RNA > 20)
    seurat_object
    # ================ transform ================ 
    seurat_object <- SCTransform(seurat_object, verbose = FALSE)
    # Note that this single command replaces NormalizeData, ScaleData, and FindVariableFeatures.
    # Transformed data will be available in the SCT assay, which is set as the default after running sctransform
    seurat_object <- RunPCA(object = seurat_object, verbose = FALSE)
    seurat_object <- RunUMAP(object = seurat_object, dims = 1:20, verbose = FALSE)
    seurat_object <- FindNeighbors(seurat_object, reduction = "pca", dims = 1:50)
    seurat_object <- FindClusters(seurat_object, resolution = 0.65) 
    return(seurat_object)
}

out_dir <- "../../output/method_comparison/"

for (datatype in c('rna', 'atac')){
    obj <- read_data(datatype)
    obj <- run_seurat(obj)
    save(obj, 
         file = file.path(
             out_dir, 
             paste0("snare_seurat_", datatype, ".RData")
             )
         )
}

