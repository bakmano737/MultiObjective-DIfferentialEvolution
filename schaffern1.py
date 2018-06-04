######################################################################
# Schaffer N1 Optimization Test Problem                              #
######################################################################
# Single Paramter x - limited to [-10,10]                            #
######################################################################

import numpy as np
import matplotlib.pyplot as plt
import diffevol as de

# My DE requires normalized paramters
#   norm takes x  from [-R,R] to [0,1]    (x0)
# denorm takes x0 from [0,1]    to [-R,R] (x)

# R
__sn1R__ = 10

def denorm(x0):
    return 2*__sn1R__*x0-__sn1R__

def norm(x):
    return (x+__sn1R__)/2*__sn1R__

def f1(x):
    return x**2

def f2(x):
    return (x-2)**2

def sn1(x):
    return np.hstack((f1(denorm(x)),f2(denorm(x))))

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
    G = [5,10,50]
    S = ['b+','r^','ko']
    N = 50
    pcr = 0.7
    fde = 0.7
    pmut = 0.5
    # Create an initial population
    Pop = np.random.rand(N,1)
    # Evaluate cost function for initial pop
    cf = sn1
    Cost = cf(Pop)
    # Run DE
    [f1t, f2t] = tp()
    plot = plt.subplot(111)
    plot.plot(f1t,f2t,'b',linewidth=3,label="True Front")
    for i,g in enumerate(G):
        gen = "Gen: {0}".format(g)
        PF = de.demo(Pop,Cost,pcr,fde,pmut,0,0,g,cf)
        R1 = PF[PF[:,0]==1]
        print("Number of Rank 1 Solutions: {0}".format(R1.shape[0]))
        print("Ideal Parameter Vals:")
        print(denorm(R1[:,1]))
        f1s = R1[:,2]
        f2s = R1[:,3]
        plot.plot(f1s,f2s,S[i],label=gen)
    plt.legend()
    plt.show()
    return

desim()