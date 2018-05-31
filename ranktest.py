import schaffern1 as sn1
import numpy as np
import matplotlib.pyplot as plt

def bestRank(Cost):
    Pareto = np.ones(Cost.shape[0],dtype=bool)
    for i,cst in enumerate(Cost):
        if Pareto[i]:
            Pareto[Pareto] = np.any(Cost[Pareto]<=cst, axis=1)
    return Pareto

def compRank(Pop):
    N = Pop.shape[0]
    rank = 1
    Rank = np.zeros((N,1))
    Ranked = np.array([[],[],[],[]]).T
    while Ranked.shape[0] < N:
        p = bestRank(Pop[:,1:])
        r = rank*np.ones((p.shape[0],1))
        ranked = np.hstack((r[p],Pop[p,:]))
        Ranked = np.vstack((Ranked,ranked))
        #print(Ranked.shape[0])
        Pop = Pop[~p]
        Rank = Rank[~p]
        rank += 1
    return Ranked

def rankTest():
    print("Begin")
    N  = 25
    X0 = np.random.rand(N,1)
    X  = sn1.denorm(X0)
    C  = sn1.sn1(X)
    P  = np.hstack((X,C))
    R  = compRank(P)
    r  = R[:,0]
    f1 = R[:,2]
    f2 = R[:,3]
    plt.scatter(f1,f2,c=r)
    plt.show()

rankTest()