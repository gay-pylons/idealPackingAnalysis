'''
Created by Francesco
2 March 2019
'''

import numpy as np
import matplotlib.pyplot as plt
import pyCudaPacking as pcp
import npquad
import sys
import os

nDim = int(sys.argv[1])
numParticles = int(sys.argv[2])
dirName = sys.argv[3]

p = pcp.Packing(nDim = nDim, numParticles = numParticles)
gaps = []
for dirPacking in os.listdir(dirName):
    p.load(dirName + "/" + dirPacking)
    gapList = p.getNeighborGaps().astype(float).data
    #gapHist, edges = np.histogram(gapList, np.geomspace(np.min(gapList), np.max(gapList)), density=True)
    #edges = (edges[:-1] + edges[1:])/2
    #gapList /= edges[np.argmin(gapHist)]
    gapList /= np.max(gapList)
    for i in range(len(gapList)):
        gaps.append(gapList[i])

gapPDF, edges = np.histogram(gaps, np.geomspace(np.min(gaps), np.max(gaps)), density=True)
edges = (edges[:-1] + edges[1:])/2

savePDF = np.zeros((len(gapPDF), 2))
for i in range(len(gapPDF)):
    savePDF[i] = [edges[i], gapPDF[i]]

np.savetxt(dirName + "output/gapPDF.dat", savePDF)
plt.loglog(edges, gapPDF)
plt.show()
