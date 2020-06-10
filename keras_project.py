#! /usr/bin/env python
# -*- coding: utf-8 -*-


## Import packages
import os
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
#from sklearn.feature_selection import SelectKBest, f_classif
#from sklearn.feature_selection import chi2
#from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import time

def make_submission(y_predict, name=None, date=True):
    n_elements = len(y_predict)

    if name is None:
      name = 'submission'
    if date:
      name = name + '_{}'.format(time.strftime('%d-%m-%Y_%Hh%M'))

    with open(name + ".txt", 'w') as f:
        f.write('"ID","PREDICTION"\n')
        for i in range(n_elements):
            if np.isnan(y_predict[i,1]):
                raise ValueError('NaN detected!')
            line = '{:0.0f},{:0.0f}\n'.format(y_predict[i,0],y_predict[i,1])
            f.write(line)
    print("Submission file successfully written!")

def keras_model (X,y):    
    model = Sequential()
    model.add(Dense(416, activation='relu'))
    model.add(Dense(208, activation='relu'))
    model.add(Dense(104, activation='relu'))
    model.add(Dense(11, activation='softmax')) 
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y[:,1].reshape((-1, 1)), epochs=20, batch_size=64, validation_split = 0.2, verbose = True)
    return model

def predict_class(model, X_test):
    
    predict_vector = np.zeros([len(X_test)])
    for i in range(0,len(X_test)):
        proba_nb = model.predict(X_test[i].reshape(1,-1), batch_size = 64)
        
        max_nb = np.argmax(proba_nb)
        if max_nb == 10:
            predict_vector[i] = -1
        else: 
            predict_vector[i] = max_nb
    
    return predict_vector

if __name__ == "__main__":
    
    # Load data_train
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    
    nb_samples, nb_features = X_train.shape
    
    # Feature Engineering
    X_train2 = X_train[:,1:nb_features]
    X_test2 = X_test[:,1:nb_features]
    
    for i in range(0,len(X_train2)):
        for j in range(0,416):
            if (X_train2[i][j] == -9999999):
                X_train2[i][j] = 0.0
                
    for i in range(0,len(X_test2)):
        for j in range(0,416):
            if (X_test2[i][j] == -9999999):
                X_test2[i][j] = 0.0
    
    X_train2 = scale(X_train2)
    X_test2 = scale(X_test2)
    
    for i in range (0, len(y_train)):
        if (y_train[i,1] == -1):
            y_train[i,1] = 10
            
            
    #Noise = np.random.RandomState()
    #X_train_double = np.concatenate((X_train2, X_train2 + Noise.normal(0,1,416)))
    #y_train_double = np.concatenate((y_train, y_train))
    
    MLP = keras_model(X_train2, y_train)
    print("Model learnt")
        
    
    y_predict = np.zeros((len(X_test),2))
    y_predict[:,0] = X_test[:,0] # Ids for test sample
    
    y_predict[:,1] = predict_class(MLP, X_test2)
    
    # Make a submission file
    make_submission(y_predict, name="toy_submission")
    
    
