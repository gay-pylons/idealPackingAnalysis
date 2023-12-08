'''
Created by Francesco
4 November 2019
'''

import numpy as np
import pyCudaPacking as pcp
import npquad
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rcParams
import sys

nDim = int(sys.argv[1])
numParticles = int(sys.argv[2])
dirName = sys.argv[3]
whichEig = int(sys.argv[4])

def clearAxes(ax):
    rcParams['savefig.dpi'] = 300
    ax.set_aspect('equal')
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

def setTitle(ax,title):
    rcParams['font.family'] = 'monospace'
    rcParams['axes.titlesize'] = 20
    hfont = {'fontname':'Helvetica'}
    ax.set_title(title)

def makeModeVideo(pos, eigenvector, frequency, numFrames=180,frameTime=20,faceColor=[0.1,0.5,0.8], path='/home/farceri/Pictures/modeVideo' + str(whichEig) + '.gif'):
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))

    def animate(i):
        if(i < 90):
            p.setPositions(pos + eigenvector*(i/90))
        else:
            p.setPositions(pos + eigenvector*(1-(i-90)/90))
        gcf = plt.gcf()
        gcf.clear()
        ax = plt.gca()
        clearAxes(ax)
        setTitle(ax, "$\\omega =$ " + str(np.format_float_scientific(frequency, precision=3)))
        p.draw2DPacking()
        #ax.quiver(pos[:,0], pos[:,1], eigenvector[:,0]*np.cos(i*2*np.pi/360),  10*eigenvector[:,1]*np.cos(i*2*np.pi/360), color = '#0000CD')
        plt.tight_layout()
        return gcf.artists

    anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime)
    anim.save(path, writer='imagemagick', dpi=plt.gcf().dpi)

p = pcp.Packing()
p.load(dirName)
pos = p.getPositions()
eigenvalues = np.loadtxt(dirName + "eigenvalues.dat")
modeFrequency = eigenvalues[whichEig]
stableList = np.loadtxt(dirName + "stableList.dat")
numStable = len(stableList[stableList==1])
eigenvector = np.zeros((numParticles, nDim))
eigvec = np.loadtxt(dirName + "eigenvectors.dat", usecols = (whichEig,))
eigvec = eigvec.reshape(numStable, nDim)
eigenvector[stableList==1] = eigvec

makeModeVideo(pos, eigenvector, modeFrequency)
