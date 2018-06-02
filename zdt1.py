######################################################################
# ZDT1                                                               #
######################################################################
# Expects x is an array of 30 values in [0,1]                        #
######################################################################
import numpy as np
import matplotlib.pyplot as plt
import diffevol as de

def g(x):
    return 1 + 9.0/29.0 * np.sum(x[:,1:],axis=1,keepdims=True)

def h(x):
    return 1 - np.sqrt(x[:,[0]]/g(x))

def f1(x):
    return x[:,[0]]

def f2(x):
    return g(x)*h(x)

def zdt1(x):
    return np.hstack((f1(x),f2(x)))

def tp():
    x0 = np.arange(0.0,1.0,1.0E-4)
    x  = np.zeros((x0.size,30))
    x[:,0] = x0.T
    return [f1(x),1-np.sqrt(f1(x))]

def plotTP():
    f1,f2 = tp()
    plt.plot(f1,f2,color='black',linewidth=3)
    plt.show()

def desim():
    # Define Evolution Constants
    G = [5,10,20]
    S = ['b+','r^','ko']
    N = 50
    p = 30
    pcr = 0.7
    fde = 0.3
    pmut = 0.5
    # Create an initial population
    Pop = np.random.rand(N,p)
    # Evaluate cost function for initial pop
    cf = zdt1
    Cost = cf(Pop)
    # Run DE
    [f1t, f2t] = tp()
    plot = plt.subplot(111)
    plot.plot(f1t,f2t,'b',linewidth=3,label="True Front")
    for i,g in enumerate(G):
        gen = "Gen: {0}".format(g)
        PF = de.demo(Pop,Cost,pcr,fde,pmut,0,g,cf)
        R1 = PF[PF[:,0]==1]
        print("Number of Rank 1 Solutions: {0}".format(R1.shape[0]))
        print("Ideal Parameter Vals:")
        print(R1[:,1])
        f1s = R1[:,2]
        f2s = R1[:,3]
        plot.plot(f1s,f2s,S[i],label=gen)
    plt.legend()
    plt.show()
    return

#plotTP()
desim()