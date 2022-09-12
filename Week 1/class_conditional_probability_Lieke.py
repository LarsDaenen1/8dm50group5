# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 10:15:14 2022

@author: 20191819
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt

def class_conditional_prob(X,Y):
    
    #calculating the changes of classification
    py_1 = Y[:,0].sum()/Y.shape[0]
    # py_0 = 1-py_1
    
    #defining the range over which x is plotted
    x_range = np.linspace(0,1,100)
    
    #creating a figure for the plots
    # fig, ax = plt.subplots((30))
    
    #for loop over all the features in X
    for i in range(X.shape[1]):
        

        #calculating the standard deviation and mean of the features when y=1
        boolean = Y[:,0] == 1
        x = X[:,i]
        XY = x[boolean]
        
        sigmaXY = np.std(XY)**2
        muXY = np.mean(XY)
        
        #creating empty list for memory
        list_p = []
        
        #for loop over the x-range
        for j in range(x_range.shape[0]):
            
            #calculating P(X=x|Y=y)
            pxy = np.sqrt(1/(2*np.pi*sigmaXY))*np.exp(-1/(2*sigmaXY)*(x_range[j]-muXY)**2)
            p_cond = pxy/py_1
            
            #adding the Ps to a list
            list_p.append(p_cond)
        
        #plotting the data
        fig,ax = plt.subplots()
        ax.set_title('')
        ax.plot(x_range,list_p)
