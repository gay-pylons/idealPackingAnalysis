'''
Created by Francesco
19 July 2018
'''

import numpy as np
import sys

def pbcDistance(x, y, length=1):
    delta = x - y
    delta -= np.round(delta/length)*length
    return delta


def calcNorm(x):
    return np.sqrt(np.sum(x**2))


numParticles = 2**10
posDia = np.zeros((numParticles, 4))
stringPhi = sys.argv[1]
filePos = "pos_poly23_" + stringPhi + ".dat"
fileRad = "rad_poly23_" + stringPhi + ".dat"
filePosDia = "init_config_N1024_poly23_phi" + stringPhi + ".xyz"

pos = np.loadtxt(filePos, ndmin=2)
rad = np.loadtxt(fileRad)

for i in range(numParticles):
    posDia[i,0] = pos[i,0]
    posDia[i,1] = pos[i,1]
    posDia[i,2] = pos[i,2]
    posDia[i,3] = 2*rad[i]

phi = 0
for i in range(numParticles):
    phi += 4*np.pi*rad[i]**3/3
print(phi)
    
for i in range(numParticles):
    for j in range(i):
        if(i != j):
            delta = calcNorm(pbcDistance(pos[i], pos[j]))
            gap = delta - (posDia[i,3] + posDia[j,3])/2
            if(gap < 0):
                print("particles {} and {}, overlap: {}".format(i, j, gap))

np.savetxt(filePosDia, posDia, header=str(phi)+" "+str(1)+"\n\n\n", comments="")
    

