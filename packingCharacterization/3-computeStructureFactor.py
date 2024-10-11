'''
Created by Francesco
1 February 2019
refinagled for idealPackingLibrary slightly but otherwise identical.
'''

#import pyCudaPacking as pcp
import imp
pcp = imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import numpy as np
import npquad
import sys

def computeBinnedStructureFactor(p, kbin):
    numBins = len(kbin)
    k, sf, bins = p.getStructureFactorBinned(kBins = kbin)

    sfbin = np.zeros(numBins, dtype = np.quad)
    sfcount = np.zeros(numBins, dtype = np.quad)
    for i in range(len(k)):
        for j in range(1, numBins):
            if(k[i] > kbin[j-1] and k[i] < kbin[j]):
                sfbin[j] += sf[i]
                sfcount[j] += 1

    for i in range(numBins):
        if(sfcount[i] > 0):
            sfbin[i] = sfbin[i]/sfcount[i]
            sfbin[i] = sfbin[i]/kbin[i]

    saveStructureFactor = np.zeros((numBins, 2), dtype = np.quad)
    for i in range(numBins):
        saveStructureFactor[i] = [kbin[i], sfbin[i]]

    return saveStructureFactor


if __name__ == '__main__':
    nDim = int(sys.argv[1])
    numParticles = int(sys.argv[2])
    dirName = sys.argv[3]
    dirPacking = sys.argv[4]
    minimize = sys.argv[5]
    kbin = np.logspace(-1, 3, 100, dtype = np.quad)
    p = pcp.Packing(nDim = nDim, numParticles = numParticles)
    p.load(dirName + dirPacking)
    print("Packing fraction:", p.getPhi())

    if(minimize == "minimizeLog"):
        cutDistance = np.quad("1")
        logZero = float(sys.argv[6])
        maxdt = np.quad("0.01")
        p.setPotentialType(pcp.potentialEnum.log)
        p.setNeighborType(pcp.potentialEnum.nList)
        p.setLogZero(logZero)
        p.calcNeighborsCut(cutDistance)
        contacts = p.getContacts().data
        z = len(contacts[contacts==True])/numParticles
        print("Estimated number of interacting particles per particle:", z)
        #minimize logarithmic potential
        for iterationFIRE in p.FIREMinimizer(np.quad("1e-5"), maxIterations = 1e8, initialCut = cutDistance, recalcNeighbors = False, dtMax = maxdt):
            if(iterationFIRE % 1e3 == 0):
                contacts = p.getContacts().data
                z = len(contacts[contacts==True])/numParticles
                print("iteration:", iterationFIRE, "Estimated number of interacting particles per particle:", z)
                print("\nEnergy after logarithmic minimization:", p.getEnergy(), ", number of iterations:", iterationFIRE)
                gaps = p.getNeighborGaps().astype(float).data
                if(len(gaps[gaps<0])>0):
                    print("There are", len(gaps[gaps<0]), "negative gaps")
                    break
                else:
                    p.save(dirName + dirPacking + "logZero" + sys.argv[6], optionalData, overwrite=True)

        print("\nEnergy after logarithmic minimization:", p.getEnergy(), ", number of iterations:", iterationFIRE)
        p.save(dirName + dirPacking + "logZero" + sys.argv[6], optionalData, overwrite=True)

    saveStructureFactor = computeBinnedStructureFactor(kbin)
    pcp.save2DArray(dirName + dirPacking + "structureFactor.dat", saveStructureFactor)
