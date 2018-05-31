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
    return np.hstack((f1(x),f2(x)))

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
    G = 10
    N = 10
    pcr = 0.7
    fde = 0.3
    pmut = 0.25
    # Create an initial population
    Pop = np.random.rand(N,1)
    # Evaluate cost function for initial pop
    cf = sn1
    Cost = cf(Pop)
    # Run DE
    PF = de.demo(Pop,Cost,pcr,fde,pmut,0,G,cf)
    # Rank 1
    R1 = PF[PF[:,0]==1]
    print("Ideal Parameter Vals:")
    print(R1[:,1])
    f1s = R1[:,2]
    f2s = R1[:,3]
    [f1t, f2t] = tp()
    plt.plot(f1s,f2s,'ro',f1t,f2t,c='b',linewidth=3)
    plt.show()
    return

#desim()