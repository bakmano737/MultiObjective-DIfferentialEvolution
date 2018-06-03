##########################
# diffevol.py            #
# Differential Evolution #
#   by James V. Soukup   #
#   for CEE 290 HW #4    #
##########################
##################################################
# The models and cost functions are in models.py #
##################################################
import numpy as np
from numpy import random as rnd

######################################################################
# Differential Evolution                                             #
######################################################################
#  Recombination:                                                    #
#    child = parent + fde*(mate1-mate2)                              #
#    mate1 - First randomly selected member of the population        #
#    mate2 - Second randomly selected member of the population       #
#  Parameters:                                                       #
#    Pop    - Initial population of parameters                       #
#    cost   - Costs of initial population                            #
#    cr     - Crossover probability                                  #
#    fde    - Child variability factor                               #
#    pmut   - Mutation Probability                                   #
#    i      - Generation counter                                     #
#    im     - Max Generation Count                                   #
#    etol   - Exit Tolerance (Convergance)                           #
#    hist   - Lowest SSR of all previous generations (Analysis)      #
#    cf     - Cost Function                                          #
#    carg   - Cost Function Arguments                                #
######################################################################
def diffevol(Pop,cost,cr,fde,pmut,i,im,hist,etol,cf,carg):
    #########################
    # Step One: Selection   #
    #########################
    # Generate two unique random integers #
    # for each member of the population   #
    r = rnd.choice(Pop[:,0].size, (Pop[:,0].size,2))
    # Replace pairs of duplicates with a unique pair
    dup    = r[:,0]==r[:,1]
    r[dup] = rnd.choice(Pop[:,0].size,r[dup].shape,False)
    # Define the mating partners
    FirstMates = Pop[r[:,0],:]
    SecndMates = Pop[r[:,1],:]
    ####################
    # Step Two: Mating #
    ####################
    # Partial Crossover
    Pcr = rnd.choice([0,1],Pop.shape,p=[1-cr,cr])
    # Recombination
    mateDiff = np.subtract(FirstMates,SecndMates)
    crssover = np.multiply(fde*Pcr,mateDiff)
    Child    = np.mod(np.add(Pop,crssover),1)
    # Mutation
    Mut = rnd.rand(*Child.shape)
    Mut = Mut<pmut
    Child[Mut] = rnd.rand(*Child[Mut].shape)
    #########################
    # Step Three: Rejection #
    #########################
    # Evaluate Cost for Child Population
    chCst = cf(Child,carg)
    costc = chCst[1][1]
    costp = cost[1][1]
    # Replace dominated offspring with parent
    dom = np.array(np.greater(costc,costp)).reshape((-1,))
    Child[dom] = Pop[dom]
    np.minimum(costc,costp,out=costc)
    chCst[1][1] = costc

    # Best in show
    best = np.min(costc)
    hist[i] = best

    # Check convergance
    #if best <= etol:
    #   return [Child,chCst]

    # Check Generation Counter 
    if (im <= i+1):
        # Maximum Number of generations reached
        # Return the current population
        return [Child,chCst]

    ##############################
    # Create the next generation #
    ##############################
    return diffevol(Child,chCst,cr,fde,pmut,i+1,im,hist,etol,cf,carg)

######################################################################
# Differential Evolution Alternate Recombination                     #
######################################################################
#  Recombination:                                                    #
#    child = parent + fde*(mate1-mate2) + lam*(best-parent)          #
#    best  - Individual with lowest SSR in current generation        #
#    mate1 - First randomly selected member of the population        #
#    mate2 - Second randomly selected member of the population       #
#  Parameters:                                                       #
#    Pop   - Initial population of parameters                        #
#    cost  - Costs of initial population                             #
#    cr    - Crossover probability                                   #
#    fde   - Child variability factor                                #
#    lam   - Best parent scaling factor                              #
#    pmut  - Mutation Probability                                    #
#    i     - Generation counter                                      #
#    im    - Max Generation Count                                    #
#    etol  - Exit Tolerance (Convergance)                            #
#    hist  - Lowest SSR of all previous generations (Analysis)       #
#    cf    - Cost Function                                           #
#    carg  - Cost Function Arguments                                 #
######################################################################
def dealt(Pop,cost,cr,fde,lam,pmut,i,im,hist,etol,cf,carg):
    #########################
    # Step One: Selection   #
    #########################
    # Generate two unique random integers #
    # for each member of the population   #
    r = rnd.choice(Pop[:,0].size, (Pop[:,0].size,2))
    # Replace pairs of duplicates with a unique pair
    dup    = r[:,0]==r[:,1]
    r[dup] = rnd.choice(Pop[:,0].size,r[dup].shape,False)
    # Define the mating partners
    FirstMates = Pop[r[:,0],:]
    SecndMates = Pop[r[:,1],:]
    # Best in show
    besti = np.argmin(cost[1][1])
    bestp = Pop[besti,:]
    hist[i] = cost[1][1][besti]

    ####################
    # Step Two: Mating #
    ####################
    # Partial Crossover
    Pcr = rnd.choice([0,1],Pop.shape,p=[1-cr,cr])
    # Recombination
    mateDiff = np.subtract(FirstMates,SecndMates)
    bestDiff = np.subtract(bestp,Pop)
    crssover = np.multiply(fde*Pcr,mateDiff)
    bestover = np.multiply(lam*Pcr,bestDiff)
    fullover = np.add(crssover,bestover)
    Child    = np.mod(np.add(Pop,fullover),1)
    # Mutation
    Mut = rnd.rand(*Child.shape)
    Mut = Mut<pmut
    Child[Mut] = rnd.rand(*Child[Mut].shape)
    #########################
    # Step Three: Rejection #
    #########################
    # Evaluate Cost for Child Population
    chCst = cf(Child,carg)
    costc = chCst[1][1]
    costp = cost[1][1]
    # Replace dominated offspring with parent
    dom = np.array(np.greater(costc,costp)).reshape((-1,))
    Child[dom] = Pop[dom]
    np.minimum(costc,costp,out=costc)
    chCst[1][1] = costc

    # Check convergance
    #if best <= etol:
    #   return [Child,chCst]

    # Check Generation Counter 
    if (im <= i+1):
        # Maximum Number of generations reached
        # Return the current population
        return [Child,chCst]

    ##############################
    # Create the next generation #
    ##############################
    return dealt(Child,chCst,cr,fde,lam,pmut,i+1,im,hist,etol,cf,carg)

######################################################################
# Differential Evolution Multi Objective Pareto Ranking              #
######################################################################
#  Recombination:                                                    #
#  Parameters:                                                       #
#    Pop   - Initial population of parameters                        #
#    cost  - Costs of initial population                             #
#    cr    - Crossover probability                                   #
#    fde   - Child variability factor                                #
#    lam   - Best parent scaling factor                              #
#    pmut  - Mutation Probability                                    #
#    bhs   - Boundary Handling Strategy (0-wrap,1-reflect,2-snap)    #
#    i     - Generation counter                                      #
#    im    - Max Generation Count                                    #
#    cf    - Cost Functions                                          #
######################################################################
def demo(Pop,Cost,cr,fde,pmut,bhs,i,im,cf):
    #########################
    # Step One: Selection   #
    #########################
    # Generate two unique random integers #
    # for each member of the population   #
    N = Pop.shape[0]
    p = Pop.shape[1]
    c = Cost.shape[1]
    r = rnd.choice(N, (N,2))
    # Replace pairs of duplicates with a unique pair
    dup    = r[:,0]==r[:,1]
    r[dup] = rnd.choice(N,r[dup].shape,False)
    # Neither element of r can be its own index
    a = np.arange(N).reshape((N,1))
    r[np.equal(r,a)] = np.mod(r[np.equal(r,a)]+1,N)
    # Define the mating partners
    FirstMates = Pop[r[:,0],:]
    SecndMates = Pop[r[:,1],:]

    ####################
    # Step Two: Mating #
    ####################
    # Partial Crossover
    Pcr = rnd.choice([0,1],Pop.shape,p=[1-cr,cr])
    # Recombination
    mateDiff = np.subtract(FirstMates,SecndMates)
    crssover = np.multiply(fde*Pcr,mateDiff)
    newchild = np.add(Pop,crssover) 
    # Maintain Parameter Space
    if bhs == 0:
        # Wrap-around
        Child = np.mod(newchild,1)
    elif bhs ==1:
        # Reflection
        Child = np.mod(-newchild,1)
    elif bhs ==2:
        # Set to Bound
        Child = newchild
        Child[Child>1] = 1.0
        Child[Child<0] = 0.0
    else:
        print("Invalid BHS")
        exit()
    # Mutation
    Mut = rnd.rand(*Child.shape)
    Mut = Mut<pmut
    Child[Mut] = rnd.rand(*Child[Mut].shape)

    #########################
    # Step Three: Rejection #
    #########################
    # Evaluate Cost for Child Population
    ChCst = cf(Child)
    # Pareto Ranking
    # Gather the info for the population
    TP = np.vstack((Pop, Child))
    TC = np.vstack((Cost,ChCst))
    # Everything is unranked to begin
    Unrank = np.hstack((TP,TC))
    # Nothing is ranked to begin
    Ranked = np.empty((0,p+c))
    # Only need enough rank 1 solutions to seed the next gen
    while Ranked.shape[0] < N:
        # Get the rank 1 solutions
        b = bestRank(Unrank[:,p:])
        # Add the rank 1 solutions to the Ranked population
        Ranked = np.vstack((Ranked,Unrank[b]))
        # Remove the rank 1 solutions from the unranked population
        Unrank = Unrank[~b]
        # Rank the new population
    # Disaggregate the first N best solutions
    Child = Ranked[:N,:p]
    ChCst = Ranked[:N,p:]
    # Check Generation Counter 
    if (im <= i+1):
        # Maximum Number of generations reached
        # Return the rank1 population and cost
        return compRank(np.hstack((Child,ChCst)))

    ##############################
    # Create the next generation #
    ##############################
    return demo(Child,ChCst,cr,fde,pmut,bhs,i+1,im,cf)

def bestRank(Cost):
    Pareto = np.ones(Cost.shape[0],dtype=bool)
    for i,cst in enumerate(Cost):
        if Pareto[i]:
            Pareto[Pareto] = np.any(Cost[Pareto]<=cst, axis=1)
    return Pareto

def compRank(Pop):
    N = Pop.shape[0]
    p = Pop.shape[1]
    rank = 1
    Rank = np.zeros((N,1))
    Ranked = np.empty((0,p+1))
    while Ranked.shape[0] < N:
        b = bestRank(Pop[:,1:])
        r = rank*np.ones((b.shape[0],1))
        ranked = np.hstack((r[b],Pop[b,:]))
        Ranked = np.vstack((Ranked,ranked))
        Pop = Pop[~b]
        Rank = Rank[~b]
        rank += 1
    return Ranked