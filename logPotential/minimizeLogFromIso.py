'''
Created by Francesco
29 April 2019
'''

import numpy as np
import npquad
import pyCudaPacking as pcp
from matplotlib import pyplot as plt
import sys

nDim = 3
numParticles = 4096

dirName = sys.argv[1]
start = int(sys.argv[2])
numSamples = int(sys.argv[3])
cutDistance = np.quad("1")
gapTh = np.quad("1e-7")
forceTh = np.quad("1e-5")
maxdt = np.quad("0.01")

dirName = "/home/farceri/Documents/Data/" + dirName + str(numParticles) + "/" + str(nDim) + "D/"
optionalData = ["neighbors", "stableHessian", "energy", "stableList", "overlaps", "logZero"]
notWellMinimized = []
p = pcp.Packing(nDim = nDim, numParticles = numParticles)
for i in range(start, start+numSamples):
#for i in notWellMinimized:
    print("\nsample:", i)
    p.load(dirName + "harmMinimized/" + str(i))
    phi = p.getPhi()
    print("Initial packing fraction:", phi)
    gaps = p.getNeighborGaps().astype(float).data
    minGap = np.min(gaps) #for an overjammed configuration as iso+1s the minGap is the maximum overlap
    print("Initial minimum gap:", minGap)
    overlaps = p.getSmallOverlaps().data
    print("number of overlaps:", len(overlaps))
    factor = 1
    while (minGap < gapTh): #shrink the system so that the minimum gap is bigger than gapTh
        radDecrease = np.quad("1") + factor*minGap
        phiDecrease = radDecrease**nDim
        p.setPhi(phi*phiDecrease)
        factor += 1
        gaps = p.getNeighborGaps().astype(float).data
        minGap = np.min(gaps)
    print("Final packing fraction:", p.getPhi(), "deltaPhi:", phi - p.getPhi())
    print("Final minimum gap:", minGap, "factor:", factor)
    overlaps = p.getSmallOverlaps().data
    print("number of overlaps:", len(overlaps))
    #compute logZero
    gaps = p.getNeighborGaps().astype(float).data
    gaps = gaps[gaps>gapTh]
    hist, edges = np.histogram(gaps, np.geomspace(np.min(gaps), np.max(gaps)), density=True)
    edges = (edges[:-1] + edges[1:])/2
    p.setPotentialType(pcp.potentialEnum.log)
    p.setLogZero(2*edges[np.argmax(hist)])
    print("LogZero", p.getLogZero())
    p.calcNeighborsCut(cutDistance)
    gaps = p.getNeighborGaps().astype(float).toarray()
    contacts = p.getContacts().todense()
    print("Estimated number of interacting particles per particle:", len(gaps[contacts==True])/numParticles)
    # minimize with FIRE
    iterationFIRE = p.minimizeFIRE(criticalForce = forceTh, maxIterations = 1e6, dtMax = maxdt)
    print("Energy after logarithmic minimization:", p.getEnergy(), ", number of iterations:", iterationFIRE)
    overlaps = p.getSmallOverlaps().data
    print("number of overlaps:", len(overlaps))
    gaps = p.getNeighborGaps().astype(float).data
    if(len(gaps[gaps<0])>0):
        print("the final packing of sample", str(i), "won't be saved")
        notWellMinized.append(i)
    else:
        p.save(dirName + "logMinimized/" + str(i), optionalData, overwrite=True)
        fz = open(dirName + "logMinimized/" + str(i) + "/logMin.dat","w")
        fz.write("{} {}\n".format(p.getEnergy(), iterationFIRE))
        fz.close()

if(len(notWellMinized)>0):
    print("unsaved final packings:", notWellMinized)
