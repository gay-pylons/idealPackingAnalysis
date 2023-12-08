'''
created by Francesco
22 June 2018
'''

import pyCudaPacking as pcp
import numpy as np
import npquad
import sys

deviceNumber = int(sys.argv[1])
nDim = int(sys.argv[2])
numParticles = int(sys.argv[3])
gapTh = np.quad("1e-07")
forceTh = np.quad(sys.argv[4])
cutDistance = np.quad("1")
logZero = np.quad(numParticles**(-1/nDim))
#initial logZero (1/N)**(1/d)
maxdt = np.quad("0.01")
phiIndex = 0
dirName = sys.argv[5]
readPacking = sys.argv[6]
dirPacking = sys.argv[7]
saveInitLogMin = sys.argv[8]
savePackings = sys.argv[9]

if(nDim == 2):
    phi1 = 0.8
    phi2 = 0.83
    phi3 = 0.85
    phiMax = np.quad("0.86")
    phiTh = np.quad("0.83")
    phiMin = np.quad("0.75")
elif(nDim == 3):
    phi1 = 0.6
    phi2 = 0.64
    phi3 = 0.67
    phiMax = np.quad("0.67")
    phiTh = np.quad("0.64")
    phiMin = np.quad("0.55")
elif(nDim == 4):
    phi1 = 0.4
    phi2 = 0.465
    phi3 = 0.485
    phiMax = np.quad("0.49")
    phiTh = np.quad("0.45")
    phiMin = np.quad("0.2")
elif(nDim == 5):
    phi1 = 0.3
    phi2 = 0.31
    phi3 = 0.32
    phiMax = np.quad("0.32")
    phiTh = np.quad("0.3")
    phiMin = np.quad("0.1")

def savePressure(saveString):
    if saveString == "savePressure":
        stressTensor = p.getStressTensor()
        trace = 0
        for i in range(nDim):
            trace = trace + stressTensor[i][i]
        Pressure = -trace/(nDim*numParticles)
        if(Pressure > 0):
            fp = open(dirName + "pressure_" + run + ".dat", "a")
            fp.write("{}, {}\n".format(phi, phi/Pressure))
            fp.close()

#compute the BMCSL equation of state Z(phi)
def saveZ(saveString):
    if saveString == "saveZ":
        rad = p.getRadii()
        O1 = np.sum(rad)*np.sum(rad**2)/(np.sum(rad**3)*numParticles)
        O2 = (np.sum(rad**2))**3/((np.sum(rad**3)**2)*numParticles)
        Z = 1/(1 - phi) + 3*O1*phi/((1 - phi)**2) + O2*phi**2*(3 - phi)/((1-phi)**3)
        fz = open(dirName + "z_" + n + ".dat", "a")
        fz.write("{}, {}\n".format(phi, 1/Z))
        fz.close()

def computeLogZero():
    p.calcNeighborsCut(cutDistance)
    gaps = p.getNeighborGaps().astype(float).data
    hist, edges = np.histogram(gaps, np.geomspace(np.min(gaps), np.max(gaps)), density=True)
    if(p.getPhi() > phiTh):
        edges = (edges[:-1] + edges[1:])/2
        logZero = 2*edges[np.argmax(hist)]
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

optionalData = ["neighbors", "stableHessian", "energy", "stableList", "overlaps", "logZero", "maxUnbalancedForce"]
fz = open(dirName + "sample.dat","w")
fz.close()

p=pcp.Packing(deviceNumber=deviceNumber, nDim=nDim, numParticles=numParticles, boxSize=np.ones(nDim, dtype=np.quad))
phiList = np.concatenate((np.linspace(phi1, phi2, 20), np.geomspace(phi2, phi3, 20)))

if readPacking == "readPacking":
    p.load(dirName + dirPacking)
    if(saveInitLogMin == "saveInitLogMin"):
        #switch to log potential
        p.setPotentialType(pcp.potentialEnum.log)
        p.setNeighborType(pcp.neighborEnum.nList)
        p.setLogZero(logZero)
        print("\nLogZero", p.getLogZero())
        p.calcNeighborsCut(cutDistance)
        iterationFIRE = p.minimizeFIRE(criticalForce = forceTh, dtMax = maxdt)
        print("\nFirst minimization: energy after logarithmic minimization:", p.getEnergy(), ", number of iterations:", iterationFIRE)
        checkOverlaps()
        p.save(dirName + str(np.float32(p.getPhi().astype(float))), optionalData, overwrite=True)
else:
    p.setLogNormalRadii(polyDispersity = 0.23)
    p.setRandomPositions()
    p.setPhi(phiMin)
    phi = phiMin
    #minimize with harmonic potential
    p.setPotentialType(pcp.potentialEnum.contactPower)
    p.setPotentialPower(2)
    phiIncrease = np.quad("1") + np.quad("1e-03")
    p.setPhi(phi * phiIncrease)
    p.minimizeFIRE(np.quad("1e-20"))
    p.setPhi(phi)
    print("\nEnergy after harmonic minimization:", p.getEnergy())
    #switch to log potential
    p.setPotentialType(pcp.potentialEnum.log)
    p.setNeighborType(pcp.neighborEnum.nList)
    p.setLogZero(logZero)
    print("\nLogZero", p.getLogZero())
    p.calcNeighborsCut(cutDistance)
    iterationFIRE = p.minimizeFIRE(criticalForce = forceTh, dtMax = maxdt)
    print("\nFirst minimization: energy after logarithmic minimization:", p.getEnergy(), ", number of iterations:", iterationFIRE)
    checkOverlaps()
    if(saveInitLogMin == "saveInitLogMin"):
        p.save(dirName + str(np.float32(p.getPhi().astype(float))), optionalData, overwrite=True)

phi = p.getPhi()
print("\nInitial packing fraction:", p.getPhi())
minGap = 1
while (minGap > gapTh or phi < phiTh):
    #compute logZero
    p.setLogZero(computeLogZero())
    print("\nLogZero", p.getLogZero())
    gaps = p.getNeighborGaps().astype(float).toarray()
    contacts = p.getContacts().todense()
    print("Estimated number of interacting particles per particle:", len(gaps[contacts==True])/numParticles)
    #minimize with FIRE
    iterationFIRE = p.minimizeFIRE(criticalForce = forceTh, maxIterations = 1e6, dtMax = maxdt)
    print("\nnumber of iterations:", iterationFIRE, "\nmaxUnbalancedForce:", p.getMaxUnbalancedForce(), "force ratio:", p.getMaxUnbalancedForce()/np.mean(p.getForceMagnitudes().astype(float).data))
    if(checkOverlaps() == True):
        break
    gaps = p.getNeighborGaps().astype(float).toarray()
    contacts = p.getContacts().todense()
    print("Estimated number of interacting particles per particle:", len(gaps[contacts==True])/numParticles)
    #save scalars, hessian and overlaps for checking
    p.save(dirName + "sample", optionalData, overwrite=True)
    #save minimized packings
    if (savePackings == "savePackings" and p.getHessian(stable=True).data.size != 0):
     for f in range(len(phiList)):
        if((phi < phiList[f] + 0.0005 and phi > phiList[f])):
            p.save(dirName + str(np.float32(p.getPhi().astype(float))), optionalData, overwrite=True)
    #increase the packing fraction by filling a fraction of the minimum gap
    gaps = p.getNeighborGaps().astype(float).data
    minGap = np.min(gaps[gaps>0])
    print("minGap = {}".format(minGap))
    fz = open(dirName + "sample.dat","a")
    fz.write("{} {} {} {} {}\n".format(phi, p.getLogZero(), minGap, iterationFIRE, p.getEnergy()))
    fz.close()
    radIncrease = (np.quad("1") + minGap)/(np.quad("1") + np.quad("0.9")*minGap)
    phiIncrease = radIncrease**nDim
    phi = phi*phiIncrease
    p.setPhi(phi)
    print("packing fraction:", p.getPhi())
    phiIndex = phiIndex + 1

print("number of compression steps:", phiIndex)
