#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 22:19:37 2019

@author: KushDani
"""

import tensorflow as tf
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk #this is scikit learn
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import os

#deals with kernel death
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#csv file is 150mb which is too large to upload, contact me for data
dataFrame = pd.read_csv('/Users/KushDani/Downloads/creditcard.csv')
print(dataFrame.head())

classCount = dataFrame.Class.value_counts()
classCount.plot(kind = 'bar')
plt.title("Fraud Tally")
plt.xlabel("Class")
plt.ylabel("Count")

#print(dataFrame.isnull().sum()) #checks for missing values
#print(dataFrame.describe())

#standardize dataset
dataFrame['normalizedAmount'] = sk.preprocessing.StandardScaler().fit_transform(dataFrame['Amount'].values.
         reshape(-1,1))
dataFrame = dataFrame.drop(['Time', 'Amount'], axis=1)
print(dataFrame.head())

x = dataFrame.iloc[:, dataFrame.columns != 'Class'] #all data except for labels
y = dataFrame.iloc[:, dataFrame.columns == 'Class'] #respective labels

#UNDERSAMPLING PREPARATION
numFraud = len(dataFrame[dataFrame['Class'] == 1])
fraudIndices = np.array(dataFrame[dataFrame['Class'] == 1].index)
validIndices = dataFrame[dataFrame['Class'] == 0].index
                                        #range       #shape
randomValidIndices = np.random.choice(validIndices, numFraud, replace = False)

#replace being False gives unique samples            #in this case makes number of valid instances
                                                     #same as number of fraud instances

#place all indices for undersampled set into one array together
undersampledIndices = np.concatenate([fraudIndices, randomValidIndices])

#gather all respective data according to index (hence, iloc) for undersampled set
undersampledData = dataFrame.iloc[undersampledIndices, :]
#print(undersampledData.columns)

#undersampling deals with heavy imbalance of dataset
x_undersampledData = undersampledData.iloc[:, undersampledData.columns != 'Class']
y_undersampledData = undersampledData.iloc[:, undersampledData.columns == 'Class']

#print('Percentage of normal transactions :', len(undersampledData[undersampledData['Class'] == 0])/
#     len(undersampledData))
#print(len(undersampledData))

x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x_undersampledData,
                                                                       y_undersampledData,
                                                                       test_size = 0.25,
                                                                       random_state=88,
                                                                       shuffle=True)
#print(len(x_train), len(x_test))
#print(len(y_train[y_train['Class'] == 1]), len(y_train[y_train['Class'] == 0]))

#COMMENTED CODE BELOW IS FOR HYPERPARAMETER OPTIMIZATION
"""
def create_model(unitsL1=16, unitsL3=8, unitsL5=1,
                 kernel_initL1='uniform', kernel_initL3=tf.initializers.truncated_normal, kernel_initL5='uniform',
                 activationL1='relu', activationL3='relu', activationL5='sigmoid',
                 drop=0.1,
                 #optimizer=tf.train.AdamOptimizer(0.0005),
                 lr=0.0005):
    
    K.clear_session()
    
    model = keras.Sequential()

    model.add(keras.layers.Dense(unitsL1,
                             kernel_initializer=kernel_initL1,
                             activation=activationL1))

    model.add(keras.layers.Dropout(drop))

    model.add(keras.layers.Dense(unitsL3,
                             kernel_initializer=kernel_initL3,
                             activation=activationL3))

    model.add(keras.layers.Dropout(drop))

    model.add(keras.layers.Dense(unitsL5,
                             kernel_initializer=kernel_initL5,
                             activation=activationL5))

    model.compile(optimizer=tf.train.AdamOptimizer(lr),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])
    
    return model

model = KerasClassifier(build_fn=create_model, batch_size=50, epochs=50)

#GRID-SEARCH PREPARATION
activation =  ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear', 'softmax', 'softplus', 'softsign']
#need for checking momentum depends on optimizer 
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
learn_rate = [0.0005, 0.001, 0.01, 0.1, 0.2, 0.3]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
neurons = [1, 2, 4, 8, 16, 32, 64]
init = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#manual change needed to check differences in optimizer performance
optimizer = [ 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
epochs= [10, 50, 100, 150]
batch_size = [10, 20, 40, 50]

param_grid = dict(epochs=epochs, batch_size=batch_size,
                  unitsL1=neurons, unitsL3=neurons, unitsL5=neurons,
                  kernel_initL1=init, kernel_initL3=init, kernel_initL5=init,
                  activationL1=activation, activationL3=activation, activationL5=activation,
                  drop=dropout_rate,
                  lr=learn_rate)

#Search/Fit parameters
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
###################################################################
#Current memory on my computer is not sufficient to run grid search

grid_result = grid.fit(x_train.values, y_train.values)

# Summary
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
"""


model = keras.Sequential()
model.add(keras.layers.Dense(16,
                             kernel_initializer='uniform',
                             activation='relu'))
model.add(keras.layers.Dropout(0.9))
model.add(keras.layers.Dense(8,
                             kernel_initializer=tf.initializers.truncated_normal,
                             activation='relu'))
model.add(keras.layers.Dropout(0.9))
model.add(keras.layers.Dense(1,
                             kernel_initializer='uniform',
                             activation='sigmoid'))
model.compile(optimizer=tf.train.AdamOptimizer(0.3),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train.values, y_train.values, epochs=150, verbose=1, validation_split=0.2,
          callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=20)])
                                                    
#print(model.summary())
                                            
accuracy = model.evaluate(x_test.values, y_test.values, batch_size=50, verbose=1)[1]
print('Test accuracy:', accuracy*100)


