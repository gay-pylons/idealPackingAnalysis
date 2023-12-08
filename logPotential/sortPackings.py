'''
Created by Francesco
March 20 2019
'''

import numpy as np
import sys
import pyCudaPacking as pcp
import npquad
import os

def sortCompressions(dirName, deltaPhi, deltaPhiString, rangeRatio):
    phiJ = np.loadtxt(dirName + "phiJ.dat")
    for indexSample in os.listdir(dirName + "samples"):
        for indexPacking in os.listdir(dirName + "samples" + os.sep + str(indexSample)):
            if(os.path.isdir(dirName + "samples" + os.sep + str(indexSample) + os.sep + str(indexPacking))):
                p.load(dirName + "samples" + os.sep + str(indexSample) + os.sep + str(indexPacking))
                if((phiJ[int(indexSample)] - p.getPhi() < deltaPhi + deltaPhi/rangeRatio) and (phiJ[int(indexSample)] - p.getPhi() > deltaPhi - deltaPhi/rangeRatio)):
                    if(os.path.getsize(dirName + "samples" + os.sep + str(indexSample) + os.sep + str(indexPacking) + os.sep + "stableHessian.mtx") != 56):
                        print(indexSample, phiJ[int(indexSample)] - p.getPhi())
                        p.save(dirName + "DeltaPhi" + deltaPhiString + os.sep + str(indexSample), optionalData, overwrite=True)
                        if(os.path.exists(dirName + "deltaPhi" + deltaPhiString + os.sep + str(indexPacking) + os.sep + "omegas.dat")):
                            omegas = np.loadtxt(dirName + "deltaPhi" + deltaPhiString + os.sep + str(indexPacking) + os.sep + "omegas.dat")
                            ipr = np.loadtxt(dirName + "deltaPhi" + deltaPhiString + os.sep + str(indexPacking) + os.sep + "ipr.dat")
                            np.savetxt(dirName + "DeltaPhi" + deltaPhiString + os.sep + str(indexSample) + os.sep + "omegas.dat", omegas)
                            np.savetxt(dirName + "DeltaPhi" + deltaPhiString + os.sep + str(indexSample) + os.sep + "ipr.dat", ipr)

def sortJammedPackings(dirName, deltaPhi, deltaPhiString):
    phiList = np.loadtxt(dirName + "samples" + os.sep + str(indexPacking) + ".dat", delimiter = " ")
    phiJ = (phiList[-1,0]*phiList[-100,1] - phiList[-100,0]*phiList[-1,1])/(phiList[-100,1] - phiList[-1,1])
    if(phiJ - phiList[-1,0] < deltaPhiMax and phiJ - phiList[-1,0] > deltaPhiMin):
        print(indexPacking, phiJ - phiList[-1,0])
        p.load(dirName + "packings" + os.sep + str(indexPacking))
        p.save(dirName + "deltaPhi" + deltaPhiString + os.sep + str(indexPacking), overwrite=True)
        omegas = np.loadtxt(dirName + "packings" + os.sep + str(indexPacking) + os.sep + "omegas.dat")
        ipr = np.loadtxt(dirName + "packings" + os.sep + str(indexPacking) + os.sep + "ipr.dat")
        np.savetxt(dirName + "deltaPhi" + deltaPhiString + os.sep + str(indexPacking) + os.sep + "omegas.dat", omegas)
        np.savetxt(dirName + "deltaPhi" + deltaPhiString + os.sep + str(indexPacking) + os.sep + "ipr.dat", ipr)

if __name__ == '__main__':
    nDim = int(sys.argv[1])
    numParticles = int(sys.argv[2])
    dirName = sys.argv[3]
    deltaPhiString = sys.argv[4]
    deltaPhi = float(sys.argv[4])
    rangeRatio = float(sys.argv[5])
    numSamples = int(sys.argv[6])

    optionalData = ["stableHessian", "energy", "stableList", "overlaps", "logZero"]
    p = pcp.Packing(nDim=nDim, numParticles=numParticles)
    sortCompressions(dirName, deltaPhi, deltaPhiString, rangeRatio)
