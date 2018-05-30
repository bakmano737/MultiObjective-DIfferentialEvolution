######################################################################
# Schaffer N1 Optimization Test Problem                              #
######################################################################
# Single Paramter x - limited to [-10,10]                            #
######################################################################

import numpy as np
import matplotlib.pyplot as plt
import diffevol as de

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

def sn1(x):
    return [f1(x),f2(x)]

def tp():
    xs = np.arange(0.0,2.0,1e-4)
    f1_xs = f1(xs)
    f2_xs = f2(xs)
    return [f1_xs,f2_xs]

def plotTP():
    f1,f2 = tp()
    plt.plot(f1,f2,color='black',linewidth=3)
    plt.show()

def desim():
    # Define Evolution Constants
    G = 50
    N = 50
    pcr = 0.7
    fde = 0.3
    lam = 0.7
    pmut = 0.25
    # Create the history array
    Hist = np.zeros(G)
    # Create an initial population
    Pop = np.random.rand(N,1)
    # Evaluate cost function for initial pop
    cf = sn1
    Cost = cf(Pop)
    # Run DE
    ParetoFront = de.demo(Pop,Cost,pcr,fde,lam,pmut,0,G,Hist,cf)
    sim = ParetoFront[0]
    print("Ideal Parameter Vals:")
    print(sim)
    f1s = ParetoFront[1][0]
    f2s = ParetoFront[1][1]
    [f1t, f2t] = tp()
    plt.plot(f1s,f2s,'rc',f1t,f2t,c='b',linewidth=3)
    plt.show()
    return

desim()