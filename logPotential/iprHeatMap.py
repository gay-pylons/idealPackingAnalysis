'''
Created by Francesco
17 April 2019
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

nDim = int(sys.argv[1])
numParticles = int(sys.argv[2])
dirName = sys.argv[3]
fileName = sys.argv[4]
numBins = int(sys.argv[5])

omegas = np.loadtxt(dirName + str(numParticles) + "/" + str(nDim) + "D/eigmod_" + fileName + ".dat")
ipr = np.loadtxt(dirName + str(numParticles) + "/" + str(nDim) + "D/ipr_" + fileName + ".dat")

iprBins = np.geomspace(1/numParticles, 1, numBins)
modBins = np.geomspace(np.min(omegas), np.max(omegas), numBins)
print("min omega:", np.min(omegas), "max omega:", np.max(omegas))
print("min ipr:", np.min(ipr), "max ipr:", np.max(ipr))

heatMap, xtics, ytics, _ = plt.hist2d(omegas, ipr, bins=100, range=[iprBins, modBins], normed=True)

np.savetxt(dirName + str(numParticles) + "/" + str(nDim) + "D/iprHeatMap_" + fileName + ".dat", heatMap)
im = plt.matshow(heatMap)
plt.colorbar(im)

for i in range(numBins):
    xtics[i] = np.format_float_scientific(xtics[i], precision = 1, exp_digits = 1)
    ytics[i] = np.format_float_scientific(ytics[i], precision = 1, exp_digits = 1)
plt.xticks(range(numBins), xtics, rotation = 45)
plt.yticks(range(numBins), ytics)

plt.show()
