import schaffern1 as sn1
import zdt1
import diffevol as de
import numpy as np
import matplotlib.pyplot as plt

def rankTest():
    N  = 50
    p  = 30
    X  = np.random.rand(N,p)
    C  = zdt1.zdt1(X)
    plt.scatter(C[:,0],C[:,1])
    plt.show()
    P  = np.hstack((X,C))
    R  = de.compRank(P,p,2)
    r  = R[:,0]
    f1 = R[:,1+p]
    f2 = R[:,2+p]
    plt.scatter(f1,f2,c=r)
    plt.show()

rankTest()