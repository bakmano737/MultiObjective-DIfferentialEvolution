######################################################################
# Schaffer N1 Optimization Test Problem                              #
######################################################################
# Single Paramter x - limited to [-10,10]                            #
######################################################################

import numpy as np
import matplotlib.pyplot as plt

# My DE requires normalized paramters
#   norm takes x  from [-10,10] to [0,1]    (x0)
# denorm takes x0 from [0,1]    to [-10,10] (x)

def denorm(x0):
    return 20*x0-10

def norm(x):
    return (x+10)/20

def f1(x):
    return x**2

def f2(x):
    return (x-2)**2

def tp():
    xs = np.arange(0.0,2.0,1e-4)
    f1_xs = f1(xs)
    f2_xs = f2(xs)
    return [f1_xs,f2_xs]

def plotTP():
    f1,f2 = tp()
    plt.plot(f1,f2,color='black',linewidth=3)
    plt.show()

plotTP()