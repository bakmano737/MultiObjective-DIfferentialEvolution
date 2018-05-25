######################################################################
# ZDT1                                                               #
######################################################################
# Expects x is an array of 30 values in [0,1]                        #
######################################################################
import numpy as np
import matplotlib.pyplot as plt

def g(x):
    return 1 + 9.0/29.0 * np.sum(x[:,1:],axis=1)

def h(x):
    return 1 - np.sqrt(x[:,0]/g(x))

def f1(x):
    return x[:,0]

def f2(x):
    return g(x)*h(x)

def tp():
    x0 = np.arange(0.0,1.0,1.0E-4)
    x  = np.zeros((x0.size,30))
    x[:,0] = x0.T
    return [f1(x),1-np.sqrt(f1(x))]

def plotTP():
    f1,f2 = tp()
    plt.plot(f1,f2,color='black',linewidth=3)
    plt.show()

#plotTP()