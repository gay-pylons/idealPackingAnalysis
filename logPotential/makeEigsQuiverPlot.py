'''
Created by Francesco
15 March 2019
'''

import pyCudaPacking as pcp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

nDim = int(sys.argv[1])
numParticles = int(sys.argv[2])
dirName = sys.argv[3]
whichEig = int(sys.argv[4])

pos = pcp.load2DArray(dirName + "positions.dat", dtype = float)
eigenvalues = np.loadtxt(dirName + "eigenvalues.dat", dtype = float)
stableList = np.loadtxt(dirName + "stableList.dat")
numStable = len(stableList[stableList==1])
eigenvector = np.loadtxt(dirName + "eigenvectors.dat", dtype = float, usecols = (whichEig,))
stablePos = pos[stableList==1,:]
eigenvector = eigenvector.reshape(numStable, nDim)
if(nDim == 3):
    fig = plt.figure(dpi = 200)
    ax = fig.gca(projection = '3d')
    ax.set_aspect('equal')
    ax.quiver(stablePos[:,0], stablePos[:,1], stablePos[:,2], eigenvector[:,0], eigenvector[:,1], eigenvector[:,2], color='r', alpha = 0.8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig("/home/farceri/Pictures/paper/quiverPlot3D" + str(whichEig) + ".pdf", transparent=True, format = "pdf")

if(nDim == 2):
    fig = plt.figure(dpi = 200, figsize = (4,4))
    ax = fig.gca()
    ax.set_aspect('equal')
    ax.quiver(stablePos[:,0], stablePos[:,1], eigenvector[:,0], eigenvector[:,1], color = 'r', alpha = 0.8)#0000CD
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("/home/farceri/Pictures/paper/quiverPlot2D" + str(whichEig) + ".pdf", transparent=True, format = "pdf")

plt.show()
