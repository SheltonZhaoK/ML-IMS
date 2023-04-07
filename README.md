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
git clone https://github.com/SheltonZhaoK/Detection-of-Algorithmic-and-Data-Biases-in-Integrated-Single-Cell-RNA-Sequencing-Datasets.git
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
The package flexibly takes different integrated datasets as an input to the system and allows user to integrate their own datasets with custom integration methods. The only limitation is that the integrated datasets should have the same naming conventions and be stored in an iterable manner (see the naming convention and arrangement of demo real life datasets).

Here, we provide three demo scripts to preprocess, integrate, and reproduce the real world datasets. The integrations are done in R and Python. Therefore, 4 related R packages should be installed SeuratData, SeuratDisk, scrnabench, and Seurat. If user wants to use their own dataset, one may directly jump to the next section. To be noticed, the preprocessing and integration will take a long time for large size real-world datasets. We also uploaded the datasets to be directly used. 

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
By default, the system runs serially on one processor. User could add a flag -mc to the end of the command to enable multi-processing, and the speedup will depend on the number of processors available in the machine.
```
python3 run_IMS.py <dataset name> -mc
```
The script is flexible that the user could select ML classifiers by just changing the line "auditors = ["FFNN", "XGB", "KNN"]" in run_IMS.py. The training and testing of models are wrapped in the model.py. If user wants to add custom ML classifiers, one could simply implement the training and testing function, and add to the following lines in model.py:
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
To add custom data or classifiers, for convenience user will only need to change the custom input files directory and output results directory in run_IMS.py, model selection in run_IMS.py, and implementation of new models in model.py. If the datasets are large, 100 rounds of auditing can be reduced by changing the "numRepeats" variable in run_IMS.py.







