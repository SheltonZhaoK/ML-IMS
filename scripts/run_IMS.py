import sys, subprocess, random, os, time, yaml
import numpy as np
import pandas as pd

from os import path
from multiprocessing.pool import Pool
from multiprocessing import Manager

from data import Data
from model import *
'''
Read and prepare data
   :param countfile: a string of the name of gene counts/data csv file
   :param labelfile: a string of the name of the annotation/label csv file

   :return data: a panda dataFrame of data
   :return label: a panda dataFrame of data labels
'''
def read_data(countfile, labelfile):
    data = pd.read_csv(countfile, index_col=None, low_memory=False)
    data.rename(columns={'Unnamed: 0': 'cell_id'}, inplace=True)
    label = pd.read_csv(labelfile, index_col=None, low_memory=False)
    label.rename(columns={'X': 'cell_id'}, inplace=True)
    return data, label

'''
Filter out the labels that has less than "cutoff" data instances
   :param data: a panda dataFrame
   :param labels: a panda dataFrame
   :param col2balance: a string of the label to be balanced
'''
def filter_data(data, labels, col2balance, cutoff):
    rows2drop = []
    for did in pd.unique(labels[col2balance]):
        ix = labels[labels[col2balance] == did].index
        if(len(ix) <= cutoff):
            rows2drop.extend(ix)
    data.drop(index=rows2drop, inplace=True)
    labels.drop(index=rows2drop, inplace=True)

'''
Compute the average difference between batch classification accuracy and baseline per auditor
   :param result: a panda dataFrame
   :param columns: a list of 
   
   :return data: a panda dataFrame of data
'''
def compute_average_residues(result, columns):
    average_residues = 0
    mean = result.mean(axis=0)
    baseline = 1.0 / len(columns)
    for column in columns:
        average_residues += mean[column] - baseline
    return average_residues/len(columns)

'''
Audit dataset
    :param dataset: a panda dataFrame
    :param results: a list of panda dataFrames
    :param numRepeats: an integer value of rounds of auditing
    :param numTests: an integer value of number of test per round
    :param auditors: a list of string of the name of ML auditors
    :param col2balance: a string of label predicted
    :param trainTestSplit: a fraction of train test split
    :param testSampling: a fraction of sampling testing data

    :return results: a list of panda dataFrames for multiprocessing callback, not necessary for CPU
'''
def audit_dataset(dataset, results, numRepeats, numTests, auditors, col2balance, trainTestSplit, testSampling):
    for i in range(1, numRepeats+1):
        train_x, train_y, numlabels  = dataset.makeTrainingData(col2balance, trainTestSplit)
        models = [train_model(train_x, train_y, numlabels, auditor) for auditor in auditors]
        for j in range(1, numTests+1):
            test_x, test_y = dataset.makeTestData(col2balance, testSampling)
            for i, auditor in enumerate(auditors):
                test_model(models[i], test_x, test_y, results[i], auditor)
    return results

'''
Display the information of the averaged classification accuracy of each batch over auditors
    :param dataset: a panda dataFrame
    :param results: a list of panda dataFrames
    :param columns: a list of string of batch names
    :param integration: a string of integration method
'''
def explore_ind_batch_effect(batchReport, results, columns, integration):
    baseline = 1.0 / len(columns)
    if len(batchReport) == 0:
        cols = ["Integrations"]
        cols.extend(columns)
        for col in cols:
            batchReport[col] = []
    average_results = [i.mean(axis=0) for i in results]
    batch_ind_average = [0]*len(columns)
    for i in range(len(columns)):
        for j in range(len(average_results)):
            batch_ind_average[i] += average_results[j][columns[i]] - baseline
    results = [i/len(average_results) for i in batch_ind_average]
    results.insert(0, integration)
    batchReport.loc[len(batchReport)] = results

'''
Print and export the results
    :param report: a panda dataFrame
    :param batchReport: a panda dataFrame
    :param batchExplore: a boolean
    :param outputDirectory: a string
'''
def format_results(report, batchReport, batchExplore, outputDirectory):
    if batchExplore == True:
        print(batchReport)
    # report = report.sort_values(by=['IMSS'])
    print(report)

    '''
    Change output directory if necessary
    '''
    report.to_csv(os.path.join(outputDirectory, f"{benchmark}_integration_selection_{"".join(auditors)}.csv"))
    batchReport.to_csv(os.path.join(outputDirectory, f"{benchmark}_integrations_batch_explore.csv"))

def main(inTrainingCells, testSampling, numTests, numRepeats, inputDirectory, outputDirectory, benchmark, integrations, batchDict, multiprocessing, auditors, batchExplore):
    report = pd.DataFrame(columns = ["Integrations", "IMSS"])
    batchReport = pd.DataFrame()

    start = time.time()
    for integration in integrations:
        '''
        Change input data if necessray
        '''
        col2balance, label = batchDict[benchmark], batchDict[benchmark]
        dataDir = os.path.join(inputDirectory, benchmark)
        inputFile = os.path.join(dataDir, f"{integration}_umap.csv")
        labelFile = os.path.join(dataDir, f"{integration}_labels.csv")

        counts, annotation = read_data(inputFile, labelFile)
        dataset = Data(counts, annotation)
        columns = pd.unique(annotation[label])
        dataset.balanceData(col2balance)
        dataset.setLabel(label)
        
        results = [pd.DataFrame(columns = columns)] * len(auditors)

        if multiprocessing == "-c":
            audit_dataset(dataset, results, numRepeats, numTests, auditors, col2balance, trainTestSplit, testSampling)
        elif multiprocessing == "-mc":
            pool = Pool()
            processes_objects = [pool.apply_async(audit_dataset, args = (dataset, results, 1, numTests, auditors,col2balance, trainTestSplit, testSampling)) for i in range(numRepeats)]
            pool.close()
            pool.join()
            aggregated_audit_results = [objects.get() for objects in processes_objects]
            for i in range (len(aggregated_audit_results)):
                for j in range(len(results)):
                    results[j] = pd.concat([results[j], aggregated_audit_results[i][j]])

        if batchExplore == True:
            explore_ind_batch_effect(batchReport, results, columns, integration)

        average_residues = [compute_average_residues(i, columns) for i in results]
        report.loc[len(report)] = [integration, sum(average_residues)/len(average_residues)]

    format_results(report, batchReport, batchExplore, outputDirectory)
    print(f'Execution time: {time.time()-start} seconds')

if __name__ == "__main__":
    multiprocessing = '-mc'
    if len(sys.argv) > 2:
        multiprocessing = sys.argv[2]
    benchmark = sys.argv[1]

    paramFile = open("./IMS_parameters.yaml")
    param = yaml.load(paramFile, Loader=yaml.FullLoader)

    numRepeats = param['Auditing']['numRepeats']
    numTests = param['Auditing']['numTests']
    trainTestSplit = param['Auditing']['trainTestSplit']
    testSampling = param['Auditing']['testSampling']

    inputDirectory, outputDirectory = param['Directories']['inputDir'], param['Directories']['outputDir']
    
    integrations = param['Integrations']
    batchExplore = param['batchExplore']
    batchDict = param['Batches']

    auditors = param['Auditors']

    main(trainTestSplit, testSampling, numTests, numRepeats, inputDirectory, outputDirectory, benchmark, integrations, batchDict, multiprocessing, auditors, batchExplore)