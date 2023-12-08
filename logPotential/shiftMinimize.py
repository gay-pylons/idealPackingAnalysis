'''
Created by Francesco
4 February 2020
'''

import pyCudaPacking as pcp
import numpy as np
import npquad
import sys
import os

dirName = sys.argv[1]
p = pcp.Packing()
p.load(dirName)
iterations = p.minimizeFIRE(criticalForce = np.quad("1e-25"), maxIteratins = 1e05, dtMax = np.quad("0.01"))
print("\nnumber of iterations:", iterations, "\nmaxUnbalancedForce:", p.getMaxUnbalancedForce(), "force ratio:", p.getMaxUnbalancedForce()/np.mean(p.getForceMagnitudes().astype(float).data))
shiftList = np.array(("1e-25", "1e-20", "1e-15", "1e-10", "1e-05"))
epsilonDelta = np.zeros((shiftList.shape[0], 2))

q = pcp.Packing()
for i in range(shiftList.shape[0]):
    q.load(dirName)
    shift = q.randomPerturbation(np.quad(shiftList[i]))
    iterations = q.minimizeFIRE(criticalForce = np.quad("1e-25"), maxIterations = 1e05, dtMax = np.quad("0.01"))
    print("\nnumber of iterations:", iterations, "\nmaxUnbalancedForce:", q.getMaxUnbalancedForce(), "force ratio:", q.getMaxUnbalancedForce()/np.mean(q.getForceMagnitudes().astype(float).data))
    epsilonDelta[i,0] = np.sum(np.sqrt(np.sum(shift**2, axis=1)))
    epsilonDelta[i,1] = np.sum(q.comparePositions(p))
    print("epsilon:", epsilonDelta[i,0], "delta:", epsilonDelta[i,1])

np.savetxt(dirName + os.sep + "epsilonDelta.dat", epsilonDelta)
