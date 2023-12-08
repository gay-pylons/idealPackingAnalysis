'''
Created by Francesco
19 July 2018
'''

import numpy as np
import sys
import pyCudaPacking as pcp
import npquad

def pbcDistance(x, y):
    delta = x - y
    delta -= np.round(delta)
    return delta


def calcNorm(x):
    norm = 0
    for d in range(nDim):
        norm += x[d]**2
    return np.sqrt(norm)


import numpy as np

nDim = int(sys.argv[1])
numParticles = int(sys.argv[2])
dirName = sys.argv[3]
stringPhi = sys.argv[4]

pos = pcp.load2DArray(dirName + "pos_poly23_" + stringPhi + ".dat", dtype = np.quad)
rad = pcp.loadArray(dirName + "rad_poly23_" + stringPhi + ".dat", dtype = np.quad)
posSigma = np.zeros((numParticles, nDim+1), dtype = np.quad)

for i in range(numParticles):
    for j in range(i):
        if(i != j):
            delta = calcNorm(pbcDistance(pos[i].astype(float), pos[j].astype(float)))
            gap = delta - (rad[i] + rad[j])
            if(gap < 0):
                print("particles {} and {}, gap: {}".format(i, j, gap))

L = 16.9179047820185
for i in range(numParticles):
    for d in range(nDim):
        posSigma[i,d] = L*pos[i,d]
    posSigma[i,3] = L*rad[i]*2

print("max sigma:", -np.min(-posSigma[:,3]), "min sigma:", np.min(posSigma[:,3]))

pcp.save2DArray(dirName + "packing_poly23_" + stringPhi + ".dat", posSigma)
