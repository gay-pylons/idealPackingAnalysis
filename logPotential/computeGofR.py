'''
Created by Francesco
2 March 2020
'''

import numpy as np
import matplotlib.pyplot as plt
import pyCudaPacking as pcp
import npquad
import sys
import os

def computeDistances(pos):
    boxSize = np.quad("1")
    #compute the pairwise vector displacements
    pDist = np.repeat(pos[:, np.newaxis, :], numParticles, axis=1) - np.repeat(pos[np.newaxis, :, :], numParticles, axis=0)
    pDist += boxSize / 2
    pDist %= boxSize
    pDist -= boxSize / 2
    pDist = np.sqrt(np.sum(pDist ** 2, axis=2))
    return pDist

def computeGofR(dist):
    histograms = np.zeros((numParticles, numBins))
    bins = np.geomspace(np.min(dist), np.max(dist), numBins+1)
    for i in range(numParticles):
        histograms[i], edges = np.histogram(dist[i], bins, density=True)
    gOfR = np.mean(histograms, axis=0)
    edges = (edges[:-1] + edges[1:])/2
    return gOfR/(4*np.pi*edges**2), edges

dirName = sys.argv[1]
phiString = sys.argv[2]
numParticles = int(sys.argv[3])#limit to 8000 for memory issue
name = sys.argv[4]
if sys.argv[5]=="read":
    data1 = np.loadtxt("/mnt/Data/Data/structureFactor/Ludo/8000/" + phiString + os.sep + "gOfR.dat")
    edges1 = data1[0,:]
    g1 = data1[1,:]
    data2 = np.loadtxt("/mnt/Data/Data/structureFactor/Ludo/8000/" + phiString + "logMinimized" + os.sep + "gOfR.dat")
    edges2 = data2[0,:]
    g2 = data2[1,:]
    data3 = np.loadtxt(dirName + os.sep + "gOfR.dat")
    edges3 = data3[0,:]
    g3 = data3[1,:]
else:
    numBins = 500
    p = pcp.Packing()
    p.load("/mnt/Data/Data/structureFactor/Ludo/8000/" + phiString)
    q = pcp.Packing()
    q.load("/mnt/Data/Data/structureFactor/Ludo/8000/" + phiString + "logMinimized")
    s = pcp.Packing()
    s.load(dirName)
    dist1 = computeDistances(p.getPositions()[:numParticles])
    dist1 = dist1[dist1!=0].astype(float)
    dist2 = computeDistances(q.getPositions()[:numParticles])
    dist2 = dist2[dist2!=0].astype(float)
    dist3 = computeDistances(s.getPositions()[:numParticles])
    dist3 = dist3[dist3!=0].astype(float)
    g1, edges1 = np.histogram(dist1.flatten(), np.linspace(np.min(dist1), np.max(dist1), numBins+1), density=True)
    g2, edges2 = np.histogram(dist2.flatten(), np.linspace(np.min(dist2), np.max(dist2), numBins+1), density=True)
    g3, edges3 = np.histogram(dist3.flatten(), np.linspace(np.min(dist3), np.max(dist3), numBins+1), density=True)
    edges1 = (edges1[:-1] + edges1[1:])/2
    edges2 = (edges2[:-1] + edges2[1:])/2
    edges3 = (edges3[:-1] + edges3[1:])/2
    g1 = g1/(4*np.pi*edges1**2)
    g2 = g2/(4*np.pi*edges2**2)
    g3 = g3/(4*np.pi*edges3**2)
    edges1 = edges1/np.mean(p.getRadii()).astype(float)
    edges2 = edges2/np.mean(q.getRadii()).astype(float)
    edges3 = edges3/np.mean(s.getRadii()).astype(float)
    #g1, edges1 = computeGofR(dist1)
    #g2, edges2 = computeGofR(dist2)
    #g3, edges3 = computeGofR(dist3)
    np.savetxt("/mnt/Data/Data/structureFactor/Ludo/8000/" + phiString + os.sep + "gOfR.dat", np.vstack((edges1, g1)))
    np.savetxt("/mnt/Data/Data/structureFactor/Ludo/8000/" + phiString + "logMinimized" + os.sep + "gOfR.dat", np.vstack((edges2, g2)))
    np.savetxt(dirName + os.sep + "gOfR.dat", np.vstack((edges3, g3)))
#make figure
fig, ax = plt.subplots(figsize = (5, 4), dpi = 150)
ax.plot(edges1, g1, color='b', linewidth=1.5)
ax.plot(edges2, g2, color='g', linewidth=1.5)
ax.plot(edges3, g3, color='r', linewidth=1.5)
ax.set_xlim(1.1,7.6)
ax.tick_params(axis='both', labelsize=13)
ax.set_xlabel("$r/\\langle \\sigma \\rangle$", fontsize=18)
ax.set_ylabel("$g(r)$", fontsize=18)
#ax.set_yticks((0, 1e-05, 2e-05, 3e-05))
#ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax.legend(("equilibrated", "equilibrated log minimized", "random compressed"), loc = "lower right", fontsize = 12)
plt.tight_layout()
plt.savefig("/home/farceri/Pictures/paper/gOfR_N" + sys.argv[3] + ".pdf", transparent=True, format = "pdf")
plt.show()
'''
#make figure of differences
figd, axd = plt.subplots(1, 2, figsize = (8.5, 4), dpi = 150)
axd[0].plot(edges1, g1-g2, color='g', linewidth=1.5)
axd[1].semilogx(edges1, g1-g2, color='g', linewidth=1.5)
axd[0].plot(edges1, g1-g3, color='r', linewidth=1.5)
axd[1].semilogx(edges1, g1-g3, color='r', linewidth=1.5)
axd[0].tick_params(axis='both', labelsize=13)
axd[1].tick_params(axis='both', labelsize=13)
axd[0].set_xlabel("$r/\\langle \\sigma \\rangle$", fontsize=18)
axd[1].set_xlabel("$r/\\langle \\sigma \\rangle$", fontsize=18)
axd[0].set_ylabel("$g_{eq}(r/\\langle \\sigma \\rangle) - g_{log}(r/\\langle \\sigma \\rangle)$", fontsize=18)
axd[1].set_ylabel("$g_{eq}(r/\\langle \\sigma \\rangle) - g_{loc}(r/\\langle \\sigma \\rangle)$", fontsize=18)
#axd[0].set_yticks((-6e-06, -4e-06, -2e-06, 0, 2e-06))
#axd[1].set_yticks((-6e-06, -4e-06, -2e-06, 0, 2e-06))
#axd[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#axd[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axd[0].legend(("equilibrated minus equilibrated log minimized", "equilibrated minus random log minimized"), loc = "upper right", fontsize = 12)
plt.tight_layout()
plt.savefig("/home/farceri/Pictures/paper/gOfRDiff_N" + sys.argv[3] + ".pdf", transparent=True, format = "pdf")
plt.show()
'''
