# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 10:15:14 2022

@author: 20191819
"""

import numpy as np
import matplotlib.pyplot as plt

def class_conditional_prob(X,Y):
    
    #calculating the changes of classification
    py_1 = Y[:,0].sum()/Y.shape[0]
    py_0 = 1-py_1
    
    #defining the range over which x is plotted
    x_range = np.linspace(0,1,100)
    
    #creating a figure for the plots
    # fig, ax = plt.subplots((30))
    
    #for loop over all the features in X
    for i in range(X.shape[1]):
        

        #calculating the standard deviation and mean of the features when y=1
        boolean1 = Y[:,0] == 1
        x1 = X[:,i]
        XY1 = x1[boolean1]
        
        sigmaXY1 = np.std(XY1)**2
        muXY1 = np.mean(XY1)
        
        #calculating the standard deviation and mean of the features when y=0
        boolean0 = Y[:,0] == 0
        x0 = X[:,i]
        XY0 = x0[boolean0]
        
        sigmaXY0 = np.std(XY0)**2
        muXY0 = np.mean(XY0)
        
        #creating empty list for memory
        list_p1 = []
        list_p0 = []
        
        #for loop over the x-range
        for j in np.unique(XY1):
            
            #calculating P(X=x|Y=1)
            pxy1 = np.sqrt(1/(2*np.pi*sigmaXY1))*np.exp(-1/(2*sigmaXY1)*(j-muXY1)**2)
            p_cond1 = pxy1/py_1
            
            #adding the Ps to a list
            list_p1.append(p_cond1)
        
        for k in np.unique(XY0):
            #calculating P(X=x|Y=0)
            pxy0 = np.sqrt(1/(2*np.pi*sigmaXY0))*np.exp(-1/(2*sigmaXY0)*(k-muXY0)**2)
            p_cond0 = pxy0/py_0
            
            #adding p to the list
            list_p0.append(p_cond0)
        
        #plotting the data
        fig,ax = plt.subplots()
        ax.set_title('The conditional probability density of feature ' + str(i+1) + ' for y = 1')
        ax.set_xlabel('Feature value')
        ax.set_ylabel('Probability density')
        ax.plot(np.unique(XY1),list_p1)
        ax.plot(np.unique(XY0),list_p0)
        ax.legend(['P(X=x|Y=1)', 'P(X=x|Y=0)'])

