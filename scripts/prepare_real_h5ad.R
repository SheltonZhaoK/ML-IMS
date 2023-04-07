library(SeuratData)
library(SeuratDisk)
library(scrnabench)
library(Seurat)

filter_data_new <- function(dataList)
{
  nCount_RNA = NULL
  nFeature_RNA = NULL
  percent.mt = NULL
  if(is.list(dataList))
  {
    for (i in (1:length(dataList)))
    {
      dataList[[i]][["percent.mt"]] <- Seurat::PercentageFeatureSet(dataList[[i]], pattern = "^MT-")
      Total_mRNAs <- dataList[[i]][["nCount_RNA"]]$nCount_RNA
      mupper_bound <- 10^(mean(log10(Total_mRNAs)) + 2*stats::sd(log10(Total_mRNAs)))
      mlower_bound <- 10^(mean(log10(Total_mRNAs)) - 2*stats::sd(log10(Total_mRNAs)))
      Total_Genes <- dataList[[i]][["nFeature_RNA"]]$nFeature_RNA
      gupper_bound <- 10^(mean(log10(Total_Genes)) + 2*stats::sd(log10(Total_Genes)))
      glower_bound <- 10^(mean(log10(Total_Genes)) - 2*stats::sd(log10(Total_Genes)))
      dataList[[i]] <- subset(x = dataList[[i]], subset = nFeature_RNA > glower_bound & nFeature_RNA < gupper_bound &
                                nCount_RNA > mlower_bound & nCount_RNA < mupper_bound & percent.mt < 10)
    }
  }
  else{
    stop("A data list of datasets is required to preprocess datasets")
  }
  
  return(dataList)
}

dataList <- c("panc8", "pbmcsca", "ifnb", "bone_marrow")
for (i in dataList)
{
  if(i != "bone_marrow")
  {
    data <- LoadData(i)
    data <- UpdateSeuratObject(object = data)
    data <- list(data)
    names(data) <- c('Integrated')
  }
  else
  {
    data <- Seurat::CreateSeuratObject(counts = readRDS("../data/bone_marrow.rds"), min.cells = 3, min.features = 200)
    data@meta.data <- cbind(data@meta.data, read.csv("../data/metadata_full.csv"))
    data <- list(data)
    names(data) <- c('Integrated')
  }
  
  if(i == "panc8")
  {
    data$Integrated <- subset(x = data$Integrated, subset = tech == "fluidigmc1", invert = TRUE)
  }
  else if(i == "pbmcsca")
  {
    data$Integrated <- subset(x = data$Integrated, subset = Method == "CEL-Seq2", invert = TRUE)
    data$Integrated <- subset(x = data$Integrated, subset = Method == "Smart-seq2", invert = TRUE)
  }

  data <- filter_data_new(data)
  data <- run_log(data)
  data <- select_hvg(data)
  
  directory <- "../data/h5ad_real/"
  if(!file.exists(directory))
  {
    dir.create(directory)
  }
  file = paste(directory, i, '.h5Seurat', sep = "")
  SaveH5Seurat(data$Integrated, filename = file)
  Convert(file, dest = "h5ad")
  unlink(file)
  print(paste(i, "done.", sep = ""))
}
