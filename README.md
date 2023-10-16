# Tutorial: Detection of Algorithmic and Data Biases in Integrated Single-Cell RNA-Sequencing Datasets
![Machine Learning Based Integration Method Selection](./workflow.png)
This package aims to detect datasets batch-effects from batch-corrected integrated scRNA-seq datasets and to automatically find the best integration methods using Machine Learning Auditors.

The tutorial is arranged as follows:
1. Dependency requirements
2. Data Preparation
3. Test demo

## Dependency requirements
Begin by clone the repository.
```
git clone https://github.com/SheltonZhaoK/ML-IMS.git
```
The scripts are written in Python. A virtual python environment can be created with following commands:
``` 
python3 -m venv <dir>
```
The python environment could be activated by:
```
source <dir>/bin/activate
```
The python environment could be deactivated by:
```
deactivate
```
After activating the python virtual environment, all dependencies could be installed by:
```
pip3 install -r requirements.txt
```

## Data Preparation
The package flexibly takes different integrated datasets as inputs to the system and allows user to integrate their own datasets with custom integration methods. The only limitation is that the integrated datasets should have the same naming conventions and be stored in an iterable manner (see the naming convention and arrangement of demo real-life datasets).

Here, we provide three demo scripts to preprocess, integrate, and reproduce the real-world datasets. The integrations are done in R and Python. Therefore, 4 related R packages should be installed SeuratData, SeuratDisk, scrnabench, and Seurat. If user wants to use their own dataset, one may directly jump to the next section. To be noticed, the preprocessing and integration will take a long time for large size real-world datasets. We also uploaded the datasets to be directly used. 

To reproduce the real-world datasets, following commands should be executed:
```
Rscript integrate_real.R
Rscript prepare_real_h5ad.R
Python3 integrated_real.py
```
integrate_real.R and prepare_real_h5ad.R could be run parallely, and integrated_real.py must be run after the execution of prepare_real_h5ad.R.

## Test Demo
To run the Integration Method Selection system, please execute the following:
```
python3 run_IMS.py <dataset name>
```
By default, the system runs serially on one processor. User could add a flag -mc to the end of the command to enable multi-processing, and the speedup will depend on the number of processors available in the machine. For example, ```python3 run_IMS.py ifnb -mc```
```
python3 run_IMS.py <dataset name> -mc
```
The script is flexible that the user could add custom data or change parameters all in IMS_parameters.yaml file without touching the python script:
```
Directories:
   inputDir: ../data/
   outputDir: ../output/

Auditors:
  - FFNN
  - XGB
  - KNN

Auditing:
  numRepeats: 100
  numTests: 20
  trainTestSplit: 0.7
  testSampling: 0.2

Batches:
  ifnb: stim
  panc8: tech
  pbmcsca: Method
  bone_marrow: sample_id

Integrations:
  - cca
  - sctransform
  - harmony
  - fastmnn
  - bbknn
  - ingest

batchExplore: True
```
If user wants to add custom ML classifiers, one could simply implement the training and testing function, add to the following lines in wrapper functions in model.py, and edit the "Auditors" section in IMS_parameters.yaml:
```
def train_model(train_x, train_y, num_labels, name):
    if name == "FFNN":
        return train_ffnn_model(train_x, train_y, num_labels)
    elif  name == "XGB":
        return train_xgb_model(train_x, train_y, num_labels)
    elif name == "KNN":
        return train_knn_model(train_x, train_y, num_labels) 

def test_model(model, test_data, test_labels, result, name):
    if name == "FFNN":
        test_ffnn_model(model, test_data, test_labels, result)
    elif  name == "XGB":
        test_xgb_model(model, test_data, test_labels, result)
    elif name == "KNN":
        test_knn_model(model, test_data, test_labels, result)
```
By default, the system generates two output results stored in ../output/. The first output result is named as {dataset name}_integration_selection_{auditors}.csv (ifnb_integration_selection_FFNNXGBKNN.csv for example) which reports the sorted IMS score to select the best integration methods. The second output result is named as {dataset name}_integrations_batch_explore.csv (ifnb_integrations_batch_explore.csv for example) which reports the averaged classification accuracy of each batch over auditors. If user is not interested in seeing batch-related information, one may set batchExplore in IMS_parameters.yaml to False.

Additionally, in Auditing section of IMS_parameters.yaml, the variable numRepeats could be reduced to smaller number such as 50, which will improve the speed and not significantly affect the results. However, such an adjustment might cause incomprehensive representation of the datasets.

## Citation
To cite this work:
```
@inproceedings{Zhao_2023,
address={New York, NY, USA},
series={BCB ’23},
title={An Ensemble Machine Learning Approach for Benchmarking and Selection of scRNA-seq Integration Methods},
ISBN={9798400701269},
url={https://dl.acm.org/doi/10.1145/3584371.3613072},
DOI={10.1145/3584371.3613072},
booktitle={Proceedings of the 14th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics},
publisher={Association for Computing Machinery},
author={Zhao, Konghao and Bhandari, Sapan and Whitener, Nathan P and Grayson, Jason M and Khuri, Natalia},
year={2023},
month=oct,
pages={1–10},
collection={BCB ’23} }

```

