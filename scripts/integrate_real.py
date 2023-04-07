import os
import scanpy as sc
import pandas as pd

def extract_SVG(adata):
    adata.var = adata.var.replace(1, True)
    adata.var = adata.var.replace(0, False)
    adata = adata[:, adata.var["vst.variable"]]
    adata.var['highly_variable'] = adata.var['vst.variable'].to_list()
    return adata

def run_BBKNN(adata, batchkey = "SID"):
    adata = extract_SVG(adata)
    sc.pp.scale(adata, max_value = 10)
    sc.tl.pca(adata, n_comps = 10)
    sc.external.pp.bbknn(adata, batch_key=batchkey, n_pcs = 10)
    sc.tl.umap(adata)
    return adata

def run_ingest(adata, batchkey = "SID"):
    batchList = list(set(adata.obs[batchkey].to_list()))
    adata = extract_SVG(adata)
    sc.pp.scale(adata, max_value = 10)
    sc.tl.pca(adata, svd_solver='arpack', n_comps = 10)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    adata_ref = adata[adata.obs[batchkey] == batchList[0]]
    adatas = [adata[adata.obs[batchkey] == i].copy() for i in batchList[1:]]
    sc.tl.pca(adata_ref, n_comps = 10)
    sc.pp.neighbors(adata_ref)
    sc.tl.umap(adata_ref)
    for iadata, adata in enumerate(adatas):
        sc.tl.ingest(adata, adata_ref)
    adata_corrected = adata_ref.concatenate(adatas)
    return adata_corrected

def main(inputDir, outputDir):
    batchDict = {"ifnb" : 'stim', "panc8": "tech", "pbmcsca": "Method", "bone_marrow": "sample_id"}

    for reference in batchDict.keys():
        fileName = inputDir + reference + ".h5ad"
        
        #run BBKNN integration
        adata = sc.read(fileName)
        adata = run_BBKNN(adata, batchkey = batchDict[reference])
        umapData = pd.DataFrame(adata.obsm["X_umap"], index = adata.obs.index, \
        columns = ["UMAP_1", "UMAP_2"])
        labels = pd.DataFrame(adata.obs)
        umapData.to_csv(outputDir + reference + "/" + "bbknn_umap.csv")
        labels.to_csv(outputDir + reference + "/" + "bbknn_labels.csv")
        print(reference, "BBKNN", " Done")


        #run ingest integration
        adata = sc.read(fileName)
        adata = run_ingest(adata, batchkey = batchDict[reference])
        umapData = pd.DataFrame(adata.obsm["X_umap"], index = adata.obs.index, \
        columns = ["UMAP_1", "UMAP_2"])
        labels = pd.DataFrame(adata.obs)
        umapData.to_csv(outputDir + reference + "/" + "ingest_umap.csv")
        labels.to_csv(outputDir + reference + "/" + "ingest_labels.csv")
        print(reference, "Ingest", " Done")

if __name__ == "__main__":
    inputDir = "../data/h5ad_real/"
    outputDir = "../data/"
    main(inputDir, outputDir)