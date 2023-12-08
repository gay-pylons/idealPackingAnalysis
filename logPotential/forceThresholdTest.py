'''
Created by Francesco
1 February 2020
'''

import numpy as np
import pyCudaPacking as pcp
import npquad
import sys
import os

dirName = sys.argv[1]
initialForceTh = np.quad(sys.argv[2])
forceThList = np.array(("1e-12", "1e-15", "1e-18", "1e-20", "1e-25"))
data = np.zeros((forceThList.shape[0]+1,7))
data[:,0] = np.arange(1,data.shape[0]+1,1)
optionalData = ["neighbors", "stableHessian", "energy", "stableList", "overlaps", "logZero", "maxUnbalancedForce"]

q = pcp.Packing()
p = pcp.Packing()
q.load(dirName), p.load(dirName)
iteration = q.minimizeFIRE(criticalForce = initialForceTh, maxIterations = 1, dtMax = np.quad("0.01"))
print("threshold:", sys.argv[2], "maxUnbalancedForce:", q.getMaxUnbalancedForce(), "iteration:", iteration)
if(sys.argv[3] == "compare25"):
    s = pcp.Packing()
    s.load(dirName + os.sep + "minimized1e-25")
    iteration = s.minimizeFIRE(criticalForce = np.quad("1e-25"), maxIterations = 1, dtMax = np.quad("0.01"))
    print("threshold:", "1e-25", "maxUnbalancedForce:", s.getMaxUnbalancedForce(), "iteration:", iteration)
    diff = s.comparePositions(q)
    data[0,2] = np.mean(diff)
data[0,1] = initialForceTh
data[0,3] = q.getMaxUnbalancedForce()
data[0,4] = np.max(q.getForceMagnitudes().astype(float).data)
data[0,5] = np.mean(q.getForceMagnitudes().astype(float).data)
data[0,6] = 0


for i in range(forceThList.shape[0]):
    iteration = p.minimizeFIRE(criticalForce = np.quad(forceThList[i]), maxIterations = 1e05, dtMax = np.quad("0.01"))
    data[i+1,1] = forceThList[i]
    if(sys.argv[3] == "compare25"):
        diff = s.comparePositions(p)
    else:
        diff = q.comparePositions(p)
    data[i+1,2] = np.mean(diff)
    data[i+1,3] = p.getMaxUnbalancedForce()
    data[i+1,4] = np.max(p.getForceMagnitudes().astype(float).data)
    data[i+1,5] = np.mean(p.getForceMagnitudes().astype(float).data)
    data[i+1,6] = iteration
    print("\nthreshold:", forceThList[i], "maxUnbalancedForce:", p.getMaxUnbalancedForce(), "diff:", np.mean(diff), "iteration:", iteration)
    p.save(dirName + os.sep + "minimized" + forceThList[i], optionalData, overwrite=True)

np.savetxt(dirName + os.sep + "forceThCheck.dat", data)
