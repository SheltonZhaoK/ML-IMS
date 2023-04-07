import sys, subprocess, random, os, time
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
    :param numRepeats: an int numTests, auditors, col2balance, minTrainingCells
'''
def audit_dataset(dataset, results, numRepeats, numTests, auditors, col2balance, minTrainingCells):
    for i in range(1, numRepeats+1):
        train_x, train_y, numlabels  = dataset.makeTrainingData(col2balance, minTrainingCells)
        models = [train_model(train_x, train_y, numlabels, auditor) for auditor in auditors]
        for j in range(1, numTests+1):
            test_x, test_y = dataset.makeTestData(col2balance, minTestCells)
            for i, auditor in enumerate(auditors):
                test_model(models[i], test_x, test_y, results[i], auditor)
    return results

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

def format_results(report, batchReport, batchExplore):
    if batchExplore == True:
        print(batchReport)
    report = report.sort_values(by=['IMSS'])
    print(report)

    report.to_csv(f'../output/{benchmark}_integration_selection_{"".join(auditors)}.csv')
    batchReport.to_csv(f'../output/{benchmark}_integrations_batch_explore.csv')

def main(inTrainingCells, minTestCells, numTests, numRepeats, directory, benchmark, integrations, batchDict, multiprocessing, auditors, batchExplore):
    report = pd.DataFrame(columns = ["Integrations", "IMSS"])
    batchReport = pd.DataFrame()

    start = time.time()
    for integration in integrations:
        col2balance, label = batchDict[benchmark], batchDict[benchmark]
        dataDir = f'{directory}{benchmark}/'
        inputFile = f'{dataDir}{integration}_umap.csv'
        labelFile = f'{dataDir}{integration}_labels.csv'

        counts, annotation = read_data(inputFile, labelFile)
        dataset = Data(counts, annotation)
        columns = pd.unique(annotation[label])
        dataset.balanceData(col2balance)
        dataset.setLabel(label)
        
        results = [pd.DataFrame(columns = columns)] * len(auditors)

        if multiprocessing == "-c":
            audit_dataset(dataset, results, numRepeats, numTests, auditors, col2balance, minTrainingCells)
        elif multiprocessing == "-mc":
            pool = Pool()
            processes_objects = [pool.apply_async(audit_dataset, args = (dataset, results, 1, numTests, auditors,col2balance, minTrainingCells)) for i in range(numRepeats)]
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

    format_results(report, batchReport, batchExplore)
    print(f'Execution time: {time.time()-start} seconds')

if __name__ == "__main__":
    multiprocessing = '-c'
    if len(sys.argv) > 2:
        multiprocessing = sys.argv[2]
    benchmark = sys.argv[1]

    numRepeats, numTests, minTrainingCells, minTestCells = 5, 20, 0.7, 0.2 #fixme
    batchExplore = True
    
    batchDict = {"ifnb" : 'stim', "panc8": "tech", "pbmcsca": "Method", "bone_marrow" : "sample_id"}
    directory = '../data/' 
    integrations = ['cca', 'sctransform','harmony', 'fastmnn', 'bbknn', 'ingest']
    auditors = ["FFNN", "XGB", "KNN"]

    main(minTrainingCells, minTestCells, numTests, numRepeats, directory, benchmark, integrations, batchDict, multiprocessing, auditors, batchExplore)

    

    
