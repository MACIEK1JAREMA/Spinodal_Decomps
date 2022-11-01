# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:12:56 2022

@author: Wstev
"""

import numpy as np
import matplotlib.pyplot as plt

def gradient(x,y):
    """
    Inputs : x,y, two 1D arrays of the same length
    
    Outputs : m the gradient from a least squares fit of x against y
    :merr ,erro in gradient found from standard deviations in x and y
    """
    x = np.log(x)
    y = np.log(y)
    
    xerror = np.zeros(len(y)) #plots for error bars not immediately usefuly will keep for future use
    yerror = np.zeros(len(x))

    # Determine mean x, mean y, and N to use later in code
    xbar = np.mean(x)
    ybar = np.mean(y)
    N = len(x) # N is the number of data points

    # Find gradient and intercept for line of best fit using least squares criterion
    numerator = np.sum((x - xbar)*y)
    denominator = np.sum((x - xbar)**2)
    m = numerator / denominator
    c = ybar - m * xbar

    # Find uncertainty in slope and intercept

    sigmam = np.sqrt(np.sum((y-m*x-c)**2)/((N-2)*np.sum((x-xbar)**2)))
    sigmac = np.sqrt((np.sum((y-m*x-c)**2)/(N-2))*((1/N)+(xbar**2/np.sum((x-xbar)**2))))

    print("gradient is: {0:.6f} +/- {1:.6f}". format(m,sigmam,))
    #print("intercept is: {0:.6f} +/- {1:.6f}". format(c,sigmac))

    # Plot points and error bars. Plot points as a '.', colours set to black
    plt.errorbar(x,y, yerr=yerror, fmt='.', color='k', capsize=2, ecolor='k')

    # Plot line of best fit
    #plt.plot(x,m*x+c, color='k')

    return m # return gradient 


# Sample data to test function
x = np.linspace(1,100) 
y = 3*x**2 + np.random.rand(len(x))/100
lnx = np.log(x)
lny = np.log(y)
z = gradient(x,y)

