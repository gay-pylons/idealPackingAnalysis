'''
created by Francesco
22 June 2018
'''

import pyCudaPacking as pcp
import numpy as np
import npquad
import sys

deviceNumber = 0
nDim = 4
numParticles = 1024
gapTh = np.quad("1e-08")
forceTh = np.quad("1e-05")
logZero = np.quad("0.1")
maxIter = 1e7
cutDistance = np.quad("1")
maxdt = np.quad("0.01")
sample = sys.argv[1]

if(nDim == 2):
    phi1 = 0.8
    phi2 = 0.84
    phi3 = 0.86
    phiMax = np.quad("0.86")
    phiTh = np.quad("0.83")
    phiMin = np.quad("0.75")
elif(nDim == 3):
    phi1 = 0.6
    phi2 = 0.65
    phi3 = 0.67
    phiMax = np.quad("0.67")
    phiTh = np.quad("0.64")
    phiMin = np.quad("0.55")
elif(nDim == 4):
    phi1 = 0.4
    phi2 = 0.46
    phi3 = 0.49
    phiMax = np.quad("0.49")
    phiTh = np.quad("0.45")
    phiMin = np.quad("0.2")
elif(nDim == 5):
    phi1 = 0.2
    phi2 = 0.31
    phi3 = 0.32
    phiMax = np.quad("0.32")
    phiTh = np.quad("0.3")
    phiMin = np.quad("0.1")

def computeLogZero():
    p.calcNeighborsCut(cutDistance)
    gaps = p.getNeighborGaps().astype(float).data
    hist, edges = np.histogram(gaps, np.geomspace(np.min(gaps), np.max(gaps)), density=True)
    if(p.getPhi() > phiTh):
        edges = 1.2*(edges[:-1] + edges[1:])/2
        logZero = edges[np.argmax(hist)]
    else:
        width = edges[1:] - edges[:-1]
        edges = (edges[:-1] + edges[1:])/2
        peakGapCount = 2*(nDim+1)*numParticles/gaps.shape[0]
        sums = np.zeros(width.shape[0])
        sums[0] = hist[0]*width[0]
        for s in range(1, sums.shape[0]):
            sums[s] = sums[s-1] + hist[s]*width[s]
        logZero = edges[np.argwhere(sums>peakGapCount)[0,0]]
    return logZero

def checkOverlaps():
    overlapCheck = False
    overlaps = p.getSmallOverlaps().astype(float).toarray()
    if(len(overlaps[overlaps>0])>0):
        print("There are", len(overlaps[overlaps>0]), "overlaps")
        if(phi<phiTh):
            p.setPotentialType(pcp.potentialEnum.contactPower)
            p.minimizeFIRE(np.quad("1e-20"))
            p.setPotentialType(pcp.potentialEnum.log)
        else:
            overlapCheck = True
    return overlapCheck

dirName = "/home/farceri/Documents/Data/logCompression/" + str(numParticles) +"/" + str(nDim) + "D/samples/" + sample + "/"
compressionFile = "/home/farceri/Documents/Data/logCompression/" + str(numParticles) + "/" + str(nDim) + "D/compressions/" + sample + ".dat"
optionalData = ["neighbors", "stableHessian", "energy", "stableList", "overlaps", "logZero"]
fz = open(compressionFile,"w")
fz.close()

phiList = np.concatenate((np.geomspace(phi1, phi2, 20), np.geomspace(phi2, phi3, 20)))
p=pcp.Packing(deviceNumber=deviceNumber, nDim=nDim, numParticles=numParticles, boxSize=np.ones(nDim, dtype=np.quad))
#p.load("/home/farceri/Documents/Data/logCompression/8192/3D/singles/0.55")
p.setLogNormalRadii(polyDispersity = 0.23)
p.setRandomPositions()
p.setPhi(phiMin)
phi = p.getPhi()
#minimize with harmonic potential
p.setPotentialType(pcp.potentialEnum.contactPower)
p.setPotentialPower(2)
phiIncrease = np.quad("1") + np.quad("1e-03")
p.setPhi(phi * phiIncrease)
p.minimizeFIRE(np.quad("1e-20"))
p.setPhi(phi)
#switch to log potential
p.setPotentialType(pcp.potentialEnum.log)
p.setNeighborType(pcp.neighborEnum.nList)
p.setLogZero(logZero)
p.calcNeighborsCut(cutDistance)
iterationFIRE = p.minimizeFIRE(criticalForce = forceTh, dtMax = maxdt)
checkOverlaps()
minGap = 1
while (minGap > gapTh or phi < phiTh):
    #compute logZero
    p.setLogZero(computeLogZero())
    print("\nlogZero", p.getLogZero())
    p.calcNeighborsCut(cutDistance)
    #minimize with FIRE
    iterationFIRE = p.minimizeFIRE(criticalForce = forceTh, maxIterations = maxIter, dtMax = maxdt)
    if(checkOverlaps() == True):
        break
    gaps = p.getNeighborGaps().astype(float).toarray()
    contacts = p.getContacts().todense()
    minGap = np.min(p.getNeighborGaps().astype(float).data)
    print("minGap", minGap, "interacting particles per particle:", len(gaps[contacts==True])/numParticles)
    p.save(dirName + "sample", optionalData, overwrite=True)
    #save minimized packings
    if(p.getHessian(stable=True).data.size != 0):
        for f in range(phiList.shape[0]):
            if((phi < phiList[f] + 0.0005 and phi > phiList[f])):
                p.save(dirName + str(np.float32(p.getPhi().astype(float))), optionalData, overwrite=True)
    #increase the packing fraction by filling a fraction of the minimum gap
    fz = open(compressionFile,"a")
    fz.write("{} {} {} {} {}\n".format(phi, p.getLogZero(), minGap, iterationFIRE, p.getEnergy()))
    fz.close()
    radIncrease = (np.quad("1") + minGap)/(np.quad("1") + np.quad("0.9")*minGap)
    phiIncrease = radIncrease**nDim
    phi = phi*phiIncrease
    p.setPhi(phi)
    print("packing fraction:", p.getPhi())
