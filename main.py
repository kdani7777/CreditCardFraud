#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 22:19:37 2019

@author: KushDani
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk #this is scikit learn
from sklearn import preprocessing
from sklearn import model_selection
import os

#deals with kernel death
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#csv file is 150mb which is too large to upload, contact me for the data
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

model = keras.Sequential()

model.add(keras.layers.Dense(16,
                             kernel_initializer='uniform',
                             activation='relu'))

model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.Dense(8,
                             kernel_initializer=tf.initializers.truncated_normal,
                             activation='relu'))

model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.Dense(1,
                             kernel_initializer='uniform',
                             activation='sigmoid'))

model.compile(optimizer=tf.train.AdamOptimizer(0.0005),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train.values, y_train.values, epochs=150, verbose=1, validation_split=0.2)
          #callbacks=[keras.callbacks.EarlyStopping(monitor='acc',
                                                    #patience=10)])
                                                    
#print(model.summary())
                                            
accuracy = model.evaluate(x_test.values, y_test.values, batch_size=50, verbose=1)[1]
print('Test accuracy:', accuracy*100)


