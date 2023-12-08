'''
Created by Francesco
April 9 2019
'''

import pyCudaPacking as pcp
import computeStructureFactor
import numpy as np
import npquad
import sys

nDim = int(sys.argv[1])
numParticles = int(sys.argv[2])
dirName = sys.argv[3]
fileName = sys.argv[4]

p = pcp.Packing(nDim = nDim, numParticles = numParticles, boxSize = np.ones(3, dtype=np.quad))
optionalData = ["neighbors", "stableHessian", "energy", "stableList", "overlaps", "logZero"]
kbin = np.logspace(-1, 3, 100, dtype = np.quad)

L = np.genfromtxt(dirName + fileName, max_rows=1)
print("Box length: ", L)
posSigma = np.loadtxt(dirName + fileName, skiprows=1, ndmin=2)/L
phi = np.sum((np.pi/6)*posSigma[:,3]**3)
print("Read packing fraction: ", phi)
pos = np.array(posSigma[:,:3], dtype=np.quad)
rad = np.array(posSigma[:,3]/2, dtype=np.quad)
p.setPositions(pos)
p.setRadii(rad)
p.setPotentialType(pcp.potentialEnum.contactPower)
p.setNeighborType(pcp.neighborEnum.nList)
p.calcNeighborsCut(np.quad("1"))
p.minimizeFIRE(np.quad("1e-20"))
p.save(dirName + str(np.float32(p.getPhi().astype(float))), optionalData, overwrite=True)
print("Packing fraction:", p.getPhi())
#compute and save Structure Factor
saveStructureFactor = computeStructureFactor.computeBinnedStructureFactor(kbin)
pcp.save2DArray(dirName + str(np.float32(p.getPhi().astype(float))) + "/structureFactor.dat", saveStructureFactor)
