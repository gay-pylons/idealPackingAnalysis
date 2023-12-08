'''
Created by Francesco
29 May 2019
'''

import pyCudaPacking as pcp
import numpy as np
import npquad
import os
import sys
import time

nDim = 2
numParticles = int(sys.argv[1])
dirName = "/home/farceri/Documents/Data/1SS/" + str(numParticles) + os.sep + str(nDim) + "D"
optionalData = ["neighbors", "stableHessian", "energy", "stableList", "overlaps", "logZero"]
deviceNumber = int(sys.argv[2])
start = int(sys.argv[3])
numSamples = int(sys.argv[4])
nStepsPerDecadePhi = 1
phiJEstimate = np.quad("0.85")
p = pcp.Packing(deviceNumber=deviceNumber, nDim=nDim, numParticles=numParticles)

for i in range(start, start+numSamples):
    print("sample:", i)
    path = dirName + os.sep + str(i)
    lastFinishedPath = path + os.sep + "lastFinishedSystem"
    checkpointPath = path + os.sep + "checkpoint"
    p.setLogNormalRadii(polyDispersity = 0.23)
    p.setRandomPositions()
    phiCEstimate = phiJEstimate
    phiStart = phiCEstimate * 1.2

    excess = np.nan
    while np.isnan(excess):
        for stableList, excess, phiC in p.isostaticitySearch(phiCEstimate, phiStart, nStepsPerDecadePhi=nStepsPerDecadePhi, checkpointDir=checkpointPath, removeCheckpointDir=True):
            p.save(lastFinishedPath, overwrite=True)
            pcp.fileIO.saveScalar(lastFinishedPath, "phiCEstimate", phiC)
            print("energy:", p.getEnergy(), "excess:", excess, "phiC:", phiC)
        phiCEstimate *= 1.1
        phiStart *= 1.2

    print("final excess:", excess)
    if excess == 0:
        p.save(path, optionalData, overwrite=True)
