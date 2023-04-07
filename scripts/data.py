'''
This class provides utilities to prepare data for training and testing models. 
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

class Data:
   def __init__(self, counts, annotation):
      self.counts = counts 
      self.annotation = annotation 
      self.label = None
      self.train_index = None

   def getData(self):
      return self.counts, self.annotation

   def getLabel(self):
      return self.label

   def getTrainingIndex(self):
      return self.train_index

   '''
   Balance the dataset
      :param col2balance: a string of the label to be balanced
   '''
   def balanceData(self, col2balance):
      mincells = self.annotation[col2balance].value_counts().min()
      ix = []
      for x in pd.unique(self.annotation[col2balance]):
          ix.extend(self.annotation[self.annotation[col2balance] == x].sample(n=mincells).index)
      self.annotation = self.annotation.loc[ix].copy()
      self.counts = self.counts.loc[ix].copy()
      return 

   '''
   Encode categorical features as an integer array starting from 0
      :param col2balance: a string of the label to be predict
   '''
   def setLabel(self, label2encode):
      ord_enc = OrdinalEncoder().fit(self.annotation[[label2encode]])
      self.annotation['encoding'] = ord_enc.transform(self.annotation[[label2encode]])
      self.label = self.annotation['encoding']
      return

   '''
   Make training data
      :param col2balance: a string of the label to be balanced
      :param fraction: a float of the fraction to factor the training data of each label
   
      :returns train_x: a numpy array of shape (dataSize * numFeatures) for training input
      :returns train_y: a numpy array of shape (dataSize * numLabels) for training output
      :returns num_lables: an int of number of the unique labels 
   '''
   def makeTrainingData(self, col2balance, fraction):
      numcells = self.annotation[col2balance].value_counts().min()
      mincells = int(fraction * numcells)
      num_labels = len(set(self.label)) 
      ix = []
      for x in pd.unique(self.annotation[col2balance]):
         ix.extend(self.annotation[self.annotation[col2balance] == x].sample(mincells).index)
      train_y = self.label.loc[ix].copy()
      train_y = to_categorical(train_y, num_classes=num_labels, dtype='float32')
      train_x = np.array(self.counts.loc[ix].copy().drop('cell_id', axis=1))
      self.train_index = ix
      return train_x, train_y, num_labels

   '''
   Make testing data
      :param col2balance: a string of the label to be balanced
      :param fraction: a float of the fraction to factor the training data of each label

      :returns train_x: a numpy array of shape (dataSize * numFeatures) for testing input
      :returns train_y: a numpy array of shape (dataSize * numLabels) for testing output
   '''
   def makeTestData(self, col2balance, fraction):
      numcells = self.annotation[col2balance].value_counts().min()
      mincells = int(fraction * numcells)

      num_labels = len(set(self.label))
      test_x = self.counts.copy()
      test_x = test_x.drop(index = self.train_index)
      temp_annotation = self.annotation.loc[test_x.index].copy() 
      ix = []
      for x in pd.unique(temp_annotation[col2balance]):
         ix.extend(temp_annotation[temp_annotation[col2balance] == x].sample(mincells).index)
      test_x = test_x.loc[ix]
      test_y = self.label.loc[test_x.index].copy()
      test_y = to_categorical(test_y, num_classes=num_labels, dtype='float32')
      test_x = np.array(test_x.drop('cell_id', axis=1))
      return test_x, test_y
