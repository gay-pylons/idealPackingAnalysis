'''
Created by Francesco
24 April 2018
'''

import numpy as np
from scipy.signal import argrelmax
from matplotlib import pyplot as plt
import sys

nDim = int(sys.argv[1])
numParticles = int(sys.argv[2])
dirName = sys.argv[3]
fileName = sys.argv[4]
read = sys.argv[5]
numBins = int(sys.argv[6])
numCDF = int(sys.argv[7])

numEigs = numParticles*nDim-nDim #number of eigenvalues of the hessian given N and nDim minus the pressure modes

if read == "readEigs":
    eigenvalues = np.loadtxt(dirName + fileName + ".dat")
    Ns = int(len(eigenvalues)/numEigs)
    print("number of samples: {}".format(Ns))

    omega = np.sqrt(eigenvalues)
    for i in range(len(omega)):
        if(np.isnan(omega[i])):
            omega[i] = 0.
            #print("There are NaN in position", i)

    omega = omega.reshape(Ns, numEigs)
    for i in range(Ns):
        #omegaHisto, omegaBin = np.histogram(omega[i], bins=np.geomspace(np.min(omega[i]),np.max(omega[i]), 50))
        #firstMax = argrelmax(omegaHisto, order=5)[0][0]
        #print(firstMax)
        #omega[i] /= omegaHisto[firstMax]
        omega[i] /= np.mean(omega[i])

    omega = omega.reshape(Ns*numEigs)

    np.savetxt(dirName + "eigmod_" + fileName + ".dat", omega)

elif read == "readModes":
    omega = np.loadtxt(dirName + "eigmod_" + fileName + ".dat")
    print("number of modes: {}".format(len(omega)))

elif read == "readLowFrequency":
    iprTh = 0.8
    omega = np.loadtxt(dirName + "eigmod_" + fileName + ".dat")
    ipr = np.loadtxt(dirName + "ipr_" + fileName + ".dat")
    omega = omega[ipr>iprTh]


omega = np.sort(omega)
maxomega = np.max(omega)
minomega = np.min(omega)
print("minOmega: ", minomega, "maxOmega", maxomega, "meanOmega", np.mean(omega))

delta = maxomega - minomega

omegaBin = np.geomspace(minomega, maxomega, numBins)
omegaPDF, _ = np.histogram(omega, bins = omegaBin, density=True)

savePDF = np.zeros((numBins, 2))
for i in range(len(omegaPDF)):
    savePDF[i] = np.array([omegaBin[i], omegaPDF[i]])
np.savetxt(dirName + "pdf_" + fileName + ".dat", savePDF)

omegaCDF = np.arange(len(omega))/len(omega)
indices = (np.unique(np.geomspace(1, len(omega), num=numCDF, dtype=int)) - 1)
omega = omega[indices]
omegaCDF = omegaCDF[indices]

saveCDF = np.zeros((len(indices), 2))
for i in range(len(indices)):
    saveCDF[i] = np.array([omega[i], omegaCDF[i]])
np.savetxt(dirName + "cdf_" + fileName + ".dat", saveCDF)

plt.figure(0)
plt.loglog(omegaBin[:-1], omegaPDF, marker='v', linewidth = 0.5, markerfacecolor = 'r')
plt.ylabel("$PDF(\omega)$")
plt.xlabel("$\omega$")

plt.figure(1)
plt.loglog(omega, omegaCDF, marker='v', linewidth = 0.5, markerfacecolor = 'r')
plt.ylabel("$CDF(\omega)$")
plt.xlabel("$\omega$")

plt.show()
