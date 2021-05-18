
ncores <- as.numeric(Sys.getenv('SLURM_CPUS_PER_TASK'))
print(ncores)

library(cisTopic)
library(Seurat)
library(ggplot2)
library(clusterExperiment)

source("datasets_info.R")

dataset = "snare"
datatype = "atac"

out_dir <- "../../output/method_comparison/"
if (!dir.exists(out_dir)) dir.create(out_dir)

load(file.path("../../output/method_comparison/", 
               paste0("snare_seurat_rna.RData")))

count_mtx <- read_data_matrix(datatype)

bool_peak_quality <- (rowMeans(count_mtx) <= 0.1) & (rowSums(count_mtx) > 5)
count_mtx <- count_mtx[bool_peak_quality, colnames(count_mtx) %in% colnames(obj)]
cisTopicObject <- createcisTopicObject(
    count_mtx, project.name = 'P0', keepCountsMatrix = FALSE)
seurat_cluster = obj$seurat_clusters[rownames(cisTopicObject@cell.data)]
cell.data =  data.frame(seurat_cluster = seurat_cluster, row.names = names(seurat_cluster))
cisTopicObject <- addCellMetadata(
    cisTopicObject, cell.data = cell.data)

cisTopicObject <- runWarpLDAModels(
    cisTopicObject, topic=c(5, 16:25, 30), 
    seed=123, nCores=ncores, addModels=FALSE)

par(mfrow=c(3,3))
cisTopicObject <- selectModel(cisTopicObject, type='perplexity')
cisTopicObject <- selectModel(cisTopicObject, type='derivative')
cisTopicObject <- selectModel(cisTopicObject, type='maximum')

cisTopicObject <- runUmap(cisTopicObject, target='cell')

png(filename = file.path(out_dir, paste0(dataset, "umap.png")), 
    width = 1000, height = 800)
plot_df <- as.data.frame(cisTopicObject@dr[['cell']]$Umap)
plot_df$seurat_cluster <- cisTopicObject@cell.data$seurat_cluster
ggplot(plot_df) + geom_point(aes(x = UMAP1, y = UMAP2, color = seurat_cluster)) +
    theme_classic() + 
    scale_color_manual(values=clusterExperiment::bigPalette)
# plotFeatures(cisTopicObject, method='Umap', colorBy = 'seurat_cluster', target='cell')
dev.off()

save(cisTopicObject, 
     file = file.path(out_dir, paste0(dataset, "_cisTopic.RData")))
