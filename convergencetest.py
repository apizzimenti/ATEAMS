from ateam.structures import Lattice
from ateam.models import relativecocycles
from ateam import Chain



import numpy as np

import cmath

import matplotlib.pyplot as plt

size=15
field=5
L = Lattice().fromCorners([size,size,size],dimension=2, field=field)


q1=5


q2=5

polyakovAvgs=[]

biglooplist=[]

for x in range(size):
    for y in range(size):

        loopCoords=[(n,x,y) for n in range(size)]
        loopInds=[L.vertexMap[coord] for coord in loopCoords]
        loopPairs=[[loopInds[i], loopInds[(i+1)%size]] for i in range(size)]
        listBoundary=[list(x) for x in L.boundary[1]]
        edgeInds=[listBoundary.index(pair) for pair in loopPairs ]
        biglooplist.append(edgeInds)

#energyLists=[]
avgEdgeEnergies=[]
avgFaceEnergies=[]

HP = relativecocycles(L, q1, q2)

edgeEnergies=[]
faceEnergies=[]



numSteps=50
for state in Chain(HP, steps=numSteps):
        #print(state)
    
    energies=state[1]
    edgeEnergies.append(energies[0])
    faceEnergies.append(energies[1])
    polyakovs=[]
    for loop in biglooplist:
        loopvals=[int(state[0][i]) for i in loop]
        polyakov=cmath.exp(sum(loopvals)*2j*np.pi/field)
        
        polyakovs.append(polyakov)
    avgPolyakov=sum(polyakovs)/len(polyakovs)
    polyakovAvgs.append(abs(avgPolyakov))



#plt.scatter(range(numSteps),edgeEnergies,label="edge energy")
#plt.scatter(range(numSteps),faceEnergies,label="face energy")

avgPolyakovAvgs=[]

startstep=15
for i in range(5, numSteps-startstep):
    avgPolyakovAvgs.append(sum(polyakovAvgs[startstep:startstep+i])/len(polyakovAvgs[startstep:startstep+i]) )

plt.scatter(range(len(avgPolyakovAvgs)),avgPolyakovAvgs)

plt.show()