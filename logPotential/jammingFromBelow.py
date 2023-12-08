'''
Created by Francesco
April 17 2019
'''

import numpy as np
import npquad
import pyCudaPacking as pcp
import sys

deviceNumber = int(sys.argv[1])
nDim = int(sys.argv[2])
numParticles = int(sys.argv[3])
dirPacking = sys.argv[4]
cutDistance = np.quad("1")
logZero = np.quad("1")

dirName = "/home/farceri/Documents/Data/jammingFromBelow/" + str(numParticles) + "/"
optionalData = ["neighbors", "stableHessian", "energy", "stableList", "overlaps"]

p = pcp.Packing(deviceNumber = deviceNumber, nDim = nDim, numParticles = numParticles)
p.load(dirName + dirPacking)
p.setPotentialType(pcp.potentialEnum.contactPower)
p.setNeighborType(pcp.neighborEnum.nList)
p.calcNeighborsCut(cutDistance)
_, excess = p.getStableListAndExcess()
if np.isnan(excess) == True:
    excess = 0
print("Number of initial excess contacts:", excess)
phiInit = p.getPhi()
minGap = p.getMinGap()
print("minGap:", minGap)
radIncrease = (np.quad("1") + minGap)/(np.quad("1") + np.quad("0.1")*minGap)
phiIncrease = radIncrease**nDim
phi = p.getPhi()*phiIncrease
p.setPhi(phi)
print("phiCurrent - phiInit:", p.getPhi() - phiInit)
gaps = p.getNeighborGaps().data
print("negative gaps:", len(gaps[gaps<0]))
minGap = p.getMinGap()
print("minGap:", minGap)
p.minimizeFIRE(np.quad("1e-20"))
print("Starting energy:", p.getEnergy())
print("Starting packing fraction: ", p.getPhi())
p.calcNeighborsCut(cutDistance)

for excess, phiJ in p.isostaticFromBelow(phiC=p.getPhi()*(1.0005*phiIncrease), maxDeltaZ=0, phiInitial=p.getPhi(), phiResolution=np.quad("1e-7")):
    print(excess, phiJ)
    p.save(dirName + "/harmMinimized" + str(np.float32(p.getPhi().astype(float))), optionalData, overwrite=True)
_, excess = p.getStableListAndExcess()
print("Final packing fraction: ", p.getPhi(), "Number of excess contacts: ", excess)
gaps = p.getNeighborGaps().data
print("negative gaps:", len(gaps[gaps<0]))
