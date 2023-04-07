import xgboost as xgb
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

'''
Train the Feed Forward Feural Network model
   :param train_x: a numpy array of shape (dataSize * numFeatures) for the input layer
   :param train_y: a numpy array of shape (dataSize * numLabels) for output layer
   :param num_lables: an int of number of the unique labels

   :return model: an Sequential object with optimized FFNN parameters
'''
def train_ffnn_model(train_x, train_y, num_labels):
    num_layers = 1
    num_nodes = 512
    model = Sequential()
    model.add(Dense(num_nodes, input_dim=train_x.shape[1], \
          kernel_regularizer=l2(0.01), activation="relu"))
    for l in range(1, num_layers):
        num_nodes = int(num_nodes / 2)
        model.add(Dense(num_nodes, activation="relu"))
    model.add(Dense(num_labels, activation="softmax"))
    model.compile(loss='categorical_crossentropy', \
                 optimizer=optimizers.RMSprop(learning_rate=1e-4), \
                 metrics=['categorical_accuracy'])
    model.fit(train_x, train_y, epochs=50, batch_size=128,verbose=0)
    return model

'''
Train the Extreme Gradient Boosting model
   :param train_x: a numpy array of shape (dataSize * numFeatures) for training input 
   :param train_y: a numpy array of shape (dataSize * numLabels) for training output
   :param num_lables: an int of number of the unique labels

   :return model: an xgboost.core.Booster object with optimized XGB parameters
'''
def train_xgb_model(train_x, train_y, num_labels):
    d_train = xgb.DMatrix(data=train_x, label=np.argmax(train_y, axis= -1))
    param = {'max_depth': 4, 'eta': 1, 'objective': 'multi:softprob', 'num_class': num_labels}
    param['nthread'] = 4
    param['eval_metric'] = 'mlogloss'
    num_round = 50
    model = xgb.train(param, d_train, num_round)
    return model

'''
Train the k-Nearest Neighbors model
   :param train_x: a numpy array of shape (dataSize * numFeatures) for training input 
   :param train_y: a numpy array of shape (dataSize * numLabels) for training output
   :param num_lables: an int of number of the unique labels

   :return model: an KNeighborsClassifier object with optimized knn parameters
'''
def train_knn_model(train_x, train_y, num_labels):
    model = KNeighborsClassifier(n_neighbors=15)
    model.fit(train_x, train_y)
    return model

'''
Evaluate the accuracy of a classification
   :param observed: a 1D numpy array of true labels
   :param predicted: a 1D numpy array of the labels predicted

   :return: a numpy array with the probability of each label correctly predicted
'''
def evaluate_performance(observed, predicted):
    cf = confusion_matrix(observed, predicted).astype(float)
    cf = cf / cf.sum(axis=1)[:, np.newaxis]
    return cf.diagonal()

'''
Test the Feed Forward Feural Network model and record the results
 :param model: an Sequential object with optimized FFNN parameters
 :param test_data: a numpy array of shape (dataSize * numFeatures) for testing input
 :param test_labels: a numpy array of shape (dataSize * numLabels) for testing output
 :param ffnn_result: a panda dataFrame that stores the performance of the model on each label
 :param accuracy_ffnn, kappa_ffnn: lists that stores that average performance of the model
'''
def test_ffnn_model(model, test_data, test_labels, result):
    predicted = np.argmax(model.predict(test_data), axis=-1)
    observed = np.argmax(test_labels, axis=-1)
    result.loc[len(result)] = evaluate_performance(observed, predicted)

''''
Test the Extreme Gradient Boosting model and record the results
 :param model: an xgboost.core.Booster object with optimized XGB parameters
 :param test_data: a numpy array of shape (dataSize * numFeatures) for testing input
 :param test_labels: a numpy array of shape (dataSize * numLabels) for testing output
 :param xgb_result: a panda dataFrame that stores the performance of the model on each label
 :param accuracy_xgb, kappa_xgb: lists that stores that average performance of the model
'''
def test_xgb_model(model, test_data, test_labels, result):
    test_data = xgb.DMatrix(test_data)
    predicted = np.argmax(model.predict(test_data), axis=-1)
    observed = np.argmax(test_labels, axis=-1)
    result.loc[len(result)] = evaluate_performance(observed, predicted)

'''
Test the k-Nearest Neighbors model and record the results
 :param model: an KNeighborsClassifier object with optimized knn parameters
 :param test_data: a numpy array of shape (dataSize * numFeatures) for testing input
 :param test_labels: a numpy array of shape (dataSize * numLabels) for testing output
 :param knn_result: a panda dataFrame that stores the performance of the model on each label
 :param accuracy_knn, kappa_knn: lists that stores that average performance of the model
'''
def test_knn_model(model, test_data, test_labels, result):
    predicted = np.argmax(model.predict(test_data), axis=-1)
    observed = np.argmax(test_labels, axis=-1)
    result.loc[len(result)] = evaluate_performance(observed, predicted)


'''
A wrapper to train the model
'''
def train_model(train_x, train_y, num_labels, name):
    if name == "FFNN":
        return train_ffnn_model(train_x, train_y, num_labels)
    elif  name == "XGB":
        return train_xgb_model(train_x, train_y, num_labels)
    elif name == "KNN":
        return train_knn_model(train_x, train_y, num_labels)

'''
A wrapper to test the model
'''
def test_model(model, test_data, test_labels, result, name):
    if name == "FFNN":
        test_ffnn_model(model, test_data, test_labels, result)
    elif  name == "XGB":
        test_xgb_model(model, test_data, test_labels, result)
    elif name == "KNN":
        test_knn_model(model, test_data, test_labels, result)

