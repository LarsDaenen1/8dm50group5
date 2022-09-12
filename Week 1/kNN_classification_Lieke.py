# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import scipy
import numpy as np

def kNN_classifier(k, X_train, y_train, X_test, y_test):
    
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
        mean_y = sum(list_y)/k
        
        #classifying the prediction
        if mean_y < 0.5:
            yi = 0
        else:
            yi = 1
        
        #adding the prediction to y_pred
        y_pred[i,:] = yi
        
    return y_pred

def evaluation_kNN(y_test, y_pred):
    
    #initial values for cm
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    
    #for loop over all the cases
    for i in range(y_test.shape[0]):
        
        #calculating the TNs
        if (y_test[i,:][0] == 0) & (y_pred[i,:][0] ==0):
            TN = TN+1
        
        #calculating the TPs
        if (y_test[i,:][0] == 1) & (y_pred[i,:][0] ==1):
            TP = TP+1        
        
        #calculating the FNs
        if (y_test[i,:][0] == 1) & (y_pred[i,:][0] ==0):
            FN = FN+1
        
        #calculating the FPs
        if (y_test[i,:][0] == 0) & (y_pred[i,:][0] ==1):
            FP = FP+1
    
    #creating the confusion matrix
    cm = np.zeros((2,2))
    cm[0,0] = TP
    cm[1,0] = FP
    cm[0,1] = FN
    cm[1,1] = TN
    
    #calculating the accuracy
    acc = (TP+TN)/(TP+TN+FP+FN)
    
    return acc, cm