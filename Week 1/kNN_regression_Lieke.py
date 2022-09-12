# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:28:01 2022

@author: 20191819
"""

import scipy
import numpy as np

def kNN_regression(k, X_train, y_train, X_test, y_test):
    
    y_pred = np.zeros(y_test.shape)
    
    #for loop over all the test cases
    for i in range(X_test.shape[0]):
        list_dis = [] 
        indices = []
        #for loop over all the training cases
        for j in range(X_train.shape[0]):
             
            #calculating the Euclidean distance
            Dis = np.sqrt((X_train[j]-X_test[i])**2).sum()
            list_dis.append(Dis)   
            
        #sorting the distances in the list
        list_dis_sorted = list_dis.copy()
        list_dis_sorted.sort()
        
        #getting the k lowest distances and their index in the original list
        k_values = list_dis_sorted[:k]
        
        #getting the indices of the lowest distances
        for h in range(len(k_values)):
            indices.append(list_dis.index(k_values[h]))
        
        #getting the output of these indices
        list_y = y_train[indices]
        
        #calculating the mean output
        yi = sum(list_y)/k
        
        #adding the prediction to y_pred
        y_pred[i,:] = yi
        
    return y_pred

def evaluation_kNN(y_test, y_pred):
    
    diff = abs(y_test-y_pred)
    N = y_test.shape[0]
    
    mean_diff = diff.sum()/N
    return mean_diff
    
    