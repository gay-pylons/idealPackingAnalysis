'''
Created by Francesco
29 April 2019
'''

import numpy as np
import npquad
import pyCudaPacking as pcp
import sys

deviceNumber = int(sys.argv[1])
nDim = int(sys.argv[2])
numParticles = int(sys.argv[3])
start = int(sys.argv[4])
numSamples = int(sys.argv[5])
cutDistance = np.quad("1")
gapTh = np.quad("1e-8")

dirName = "/home/farceri/Documents/Data/jammingFromAbove/" + str(numParticles) + "/3D/"
optionalData = ["neighbors", "stableHessian", "energy", "stableList", "overlaps", "logZero"]

p = pcp.Packing(deviceNumber = deviceNumber, nDim = nDim, numParticles = numParticles)
for i in range(start, start + numSamples):
    p.load(dirName + str(i))
    print("sample:", i)
    phiInit = p.getPhi()
    print("current phi:", phiInit)
    #minimize with harmonic potential to check configuration
    p.setPotentialType(pcp.potentialEnum.contactPower)
    p.setNeighborType(pcp.neighborEnum.nList)
    p.calcNeighborsCut(cutDistance)
    p.setPotentialPower(2)
    p.minimizeFIRE(np.quad("1e-20"))
    p.save(dirName + str(i), optionalData = ["energy", "excessContacts"], overwrite=True)
    #decrease packing fraction by deflating twice the biggest overlap
    gaps = p.getNeighborGaps().data
    minGap = np.min(gaps)
    print("minGap:", minGap)
    if(minGap > -gapTh):
        minGap = -gapTh
    radDecrease = (np.quad("1") + minGap)/(np.quad("1") - np.quad("1.1")*minGap)
    phiDecrease = radDecrease**nDim
    p.setPhi(phiInit * phiDecrease)
    print("phi after shrinking:", p.getPhi())
    print("delta phi:", p.getPhi() - phiInit)
    gaps = p.getNeighborGaps().data
    print("negative gaps:", len(gaps[gaps<0]))
    phiSmall = p.getPhi()
    minGap = p.getMinGap()
    print("minGap:", minGap)
    radIncrease = (np.quad("1") + minGap)/(np.quad("1") + np.quad("0.1")*minGap)
    phiIncrease = radIncrease**nDim
    phi = p.getPhi()*phiIncrease
    p.setPhi(phi)
    print("phi after compression:", p.getPhi())
    print("deltaPhi", p.getPhi() - phiSmall)
    print("phiInit - phiCurrent:", phiInit - p.getPhi())
    gaps = p.getNeighborGaps().data
    print("negative gaps:", len(gaps[gaps<0]))
    p.minimizeFIRE(np.quad("1e-20"))
    print("Starting energy:", p.getEnergy())
    p.calcNeighborsCut(cutDistance)

    for excess, phiJ in p.isostaticFromBelow(phiC=p.getPhi()*(1.000001*phiIncrease), maxDeltaZ=0, phiInitial=p.getPhi(), phiResolution=np.quad("1e-7")):
        print(excess, phiJ)
    _, excess = p.getStableListAndExcess()
    print("Final deltaPhi: ", p.getPhi() - phi, "Number of excess contacts: ", excess)
    gaps = p.getNeighborGaps().data
    print("negative gaps:", len(gaps[gaps<0]))
    p.save(dirName + "harmMinimized/" + str(i), optionalData, overwrite=True)
