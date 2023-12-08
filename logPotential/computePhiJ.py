
'''
Created by Francesco
24 April 2018
'''

import numpy as np
from scipy import optimize
import sys
import os

def f(x, a=1, b=1e-04, c=1):
    return a*(np.max(x)-x + b)**c

dirName = sys.argv[1]
numSamples = int(sys.argv[2])
phiJ = np.zeros(numSamples)

for i in range(numSamples):
    energy = np.loadtxt(dirName + os.sep + "compressions" + os.sep + str(i) + ".dat", usecols = (4,))
    phi = np.loadtxt(dirName + os.sep + "compressions" + os.sep + str(i) + ".dat", usecols = (0,))
    #logzero = np.loadtxt(dirName + os.sep + str(i) + ".dat", usecols = (1,))
    #phiC = (phi[-1]*logzero[-100] - phi[-100]*logzero[-1])/(logzero[-100] - logzero[-1])
    par = optimize.curve_fit(f, phi[energy!=0], 1/energy[energy!=0], p0 = (1,1e-04,1), sigma = 1/energy[energy!=0]**2)
    par = par[0]
    phiJ[i] = np.max(phi)+par[1]
    #print(phiJ, phiC, phiJ - phiC)
    #plt.loglog(phiC - phi[-150:], 1/energy[-150:], '^')
    #plt.loglog(phiJ - phi[-150:], 1/energy[-150:], color= 'k')
    #plt.show()
    #plt.pause(2)

np.savetxt(dirName + os.sep + "phiJ.dat", phiJ)
