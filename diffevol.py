##########################
# diffevol.py            #
# Differential Evolution #
#   by James V. Soukup   #
#   for CEE 290 HW #3    #
##########################
##################################################
# The models and cost functions are in models.py #
##################################################
import models 
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rnd

##################################################
# Differential Evolution                         #
#  Parameters:                                   #
#    Pop    - Initial population of parameters   #
#    Cost   - Costs of initial population        #
#    pcr    - Crossover probability              #
#    gam    - Child variability factor           #
#    pmut   - Mutation Probability               #
#    i      - Generation counter                 #
#    imax   - Max Generation Count               #
#    cf     - Cost Function                      #
#    cfargs - Cost Function Arguments            #
#    mf     - Model Function                     #
#    mfargs - Model Function Arguments           #
##################################################
######################################################################
def diffevol(Pop,cost,cr,gam,pmut,i,im,h,etol,cf,carg):
    # Check Generation Counter #
    if (im <= i):
        # Maximum Number of generations reached
        # Return the current population
        return [Pop,cost]
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
    crssover = np.multiply(gam*Pcr,mateDiff)
    Child    = np.mod(np.add(Pop,crssover),1)
    # Mutation
    Mut = rnd.rand(*Child.shape)
    Mut = Mut<pmut
    Child[Mut] = rnd.rand(*Child[Mut].shape)
    #########################
    # Step Three: Rejection #
    #########################
    # Evaluate Cost for Child Population
    childCost = cf(Child,carg)
    costc = childCost[1][1]
    costp = cost[1][1]
    # Replace dominated offspring with parent
    if np.isnan(np.sum(costc)):
        # COST FUNCTION FAILURE
        print("Cost Function Failure")
        costc[~np.isnan(costc)] = 1e9
    dom = np.array(np.greater(costc,costp)).reshape((-1,))
    Child[dom] = Pop[dom]
    np.minimum(costc,costp,out=costc)
    childCost[1][1] = costc

    # Best in show
    best = np.min(costc)
    h[i] = best
    if best <= etol:
        return [Child,childCost]

    ##############################
    # Create the next generation #
    ##############################
    return diffevol(Child,childCost,cr,gam,pmut,i+1,im,h,etol,cf,carg)

######################################################################
# Simulator Function - Use this to run DE and process the reuslts    #
######################################################################
def deSimulate(G,N,P,pcr,fde,pmut,etol,cf,carg):
    # Create the history array
    Hist = np.zeros(G)
    # Create an initial population
    Pop = rnd.rand(N,P)
    # Evaluate cost function for initial pop
    Cost = cf(Pop,carg)
    # Run DE
    FinalGen = diffevol(Pop,Cost,pcr,fde,pmut,0,G,Hist,etol,cf,carg)
    # Parse the output [Population,[[simtim,simslug],[res,ssr]]]
    FinalPop = FinalGen[0]
    FinalCst = FinalGen[1]
    FinalSSR = FinalCst[1][1]
    # Determine the individual with the lowest SSR
    optimum  = np.argmin(FinalSSR)
    # Get the parameters, cost, and simulation of the champion
    BestPars = FinalPop[optimum]
    BestCost = FinalSSR[optimum]
    BestVals = FinalCst[0][optimum]
    # Save the current output for later
    return [BestPars,BestCost,BestVals,Hist]

# Problem #8 - Plots
# Use this function to plot the reuslts for Slug and Storage models
def dePlots():
    # Number of Generations
    G = 250
    # Population Size
    N = 50
    # Other DE Parameters
    # Crossover Probability
    pcr = 0.9
    # Recombination Variability
    fde = 0.7
    # Exit Error Tolerance
    etol = 1e-6

    ##################################################################
    # Slug Model
    ##################################################################
    obsSlug  = np.array([0.55,0.47,0.30,0.22,0.17,0.14])
    obsTime  = np.array([5.00,10.0,20.0,30.0,40.0,50.0])
    sP = 2
    d = 10
    Q = 50
    spm = float(1)/float(sP)
    scf = models.slugCost
    sca = [obsTime,Q,d,obsSlug]
    Slugs = []
    Obs = np.genfromtxt('measurement.csv',delimiter=',')
    dt = Obs[1,0] - Obs[0,0]
    ica = [Obs,dt]
    iP = 4
    ipm = float(1)/float(iP)
    icf = models.interceptionModel_CF
    Stors = []
    sims = 3
    while sims > 0:
        Slugs.append(deSimulate(G,N,sP,pcr,fde,spm,etol,scf,sca))
        Stors.append(deSimulate(G,N,iP,pcr,fde,ipm,etol,icf,ica))
        sims -= 1

    # Plot Slug Model Results
    ###
    slugPlt = plt.subplot(121)
    storPlt = plt.subplot(122)
    for i,(slug,stor) in enumerate(zip(Slugs,Stors)):
        simn = "Sim {0}".format(i)
        print(simn)
        print("Slug Model")
        print("\tParameter Values:")
        print("\t\tS={0:6.4f}".format(slug[0][0]))
        print("\t\tT={0:6.4f}".format(slug[0][1]))
        print("\tCost: {0:10.6f}".format(np.min(slug[1])))
        slugPlt.semilogy(slug[3], label=simn)
        print("Interception Model")
        print("\tParameter Values:")
        print("\t\ta={0:6.4f}".format(stor[0][0]))
        print("\t\tb={0:6.3f}".format(999*stor[0][1]+1))
        print("\t\tc={0:6.4f}".format(5.0*stor[0][2]))
        print("\t\td={0:6.4f}".format(3.0*stor[0][3]))
        print("\tCost: {0:10.6f}".format(np.min(stor[1])))
        storPlt.semilogy(stor[3], label=simn)
    ###
    slugPlt.set_xlabel('Generation')
    slugPlt.set_ylabel('Minimum SSR')
    slugPlt.set_title('Slug Model')
    slugPlt.legend()
    storPlt.set_xlabel('Generation')
    storPlt.set_ylabel('Minimum SSR')
    storPlt.set_title('Interception Model')
    storPlt.legend()
    plt.show()

dePlots()