'''
Created by Francesco
20 February 2019
'''

import numpy as np
import pyCudaPacking as pcp
import npquad
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rcParams
import sys
import os

def clearAxes(ax):
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

def setPhiTitle(ax,phi):
    maxLength=17

    phiStr="$\\varphi = $"+ str(np.round(phi,10))
    #rounding by a string trim because np.quad is annoying af
    phiStr=phiStr[:maxLength]
    #ax.set_title(phiStr,**hfont)
    phiStr=phiStr.ljust(maxLength,'0')
    setTitle(ax,phiStr)

def setEnergyTitle(ax,energy):
    maxLength=17
    energyStr="$E = $"+ str(np.round(energy,10))
    #rounding by a string trim because np.quad is annoying af
    energyStr=energyStr[:maxLength]
    #ax.set_title(phiStr,**hfont)
    energyStr=energyStr.ljust(maxLength,'0')
    setTitle(ax,energyStr)

def makeLogMinimizationVideo(path, numParticles=256, numFrames=450, frameTime=20, faceColor=[0.1,0.5,0.8]):
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))

    p = pcp.Packing()
    p.load(dirName)

    if(p.getLogZero() == 0):
        p.setPotentialType(pcp.potentialEnum.log)
        p.setNeighborType(pcp.neighborEnum.nList)
        p.setLogZero(np.quad("0.35"))
        p.calcNeighborsCut(cutDistance)
    maxIter = 100

    def animate(i):

        gcf = plt.gcf()
        gcf.clear()
        ax = plt.gca()
        clearAxes(ax)

        #minimize with FIRE
        p.minimizeFIRE(criticalForce = np.quad("1e-10"), maxIterations = maxIter, minCut = cutDistance, dtMax = np.quad("0.01"))
        print("\nEnergy after logarithmic minimization:", p.getEnergy())
        #setEnergyTitle(ax,p.getEnergy().astype(float))
        setPhiTitle(ax,p.getPhi().astype(float))
        p.draw2DPacking(faceColor=faceColor)

        return gcf.artists

    anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=True)

    anim.save(path, writer='imagemagick', dpi=plt.gcf().dpi)


def makeLogCompressionVideo(path, numParticles=256, numFrames=360, frameTime=40, faceColor=[0.1,0.5,0.8]):
    #load the initial packing
    p = pcp.Packing()
    p.load(dirName)
    #set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
    def animate(i):
        particleColor = faceColor
        gcf = plt.gcf()
        gcf.clear()
        ax = plt.gca()
        clearAxes(ax)
        setPhiTitle(ax,p.getPhi().astype(float))
        if(p.getPhi()<np.quad("0.6")):
            compressionFreq = 20
        else:
            compressionFreq = 20
        #minimize with FIRE
        iteration = 0
        p.minimizeFIRE(criticalForce = np.quad("1e-20"), maxIterations = 150, dtMax = np.quad("0.01"))
        p.draw2DPacking(faceColor=faceColor)
        overlaps = p.getSmallOverlaps().astype(float).toarray()
        if(len(overlaps[overlaps>0])>0):
            print("There are", len(overlaps[overlaps>0]), "overlaps")
        #print("\nmaxUnbalancedForce:", p.getMaxUnbalancedForce())
        if(i%compressionFreq==0 and i!=0):
            gaps = p.getNeighborGaps().astype(float).data
            minGap = np.min(gaps)
            radIncrease = np.quad("1") + np.quad("0.5")*minGap
            phiIncrease = radIncrease**nDim
            print("packing fraction increment:", phiIncrease)
            p.setPhi(p.getPhi()*phiIncrease)
            print("packing fraction:", p.getPhi())
            #compute logZero
            gaps = p.getNeighborGaps().astype(float).data
            hist, edges = np.histogram(gaps, np.geomspace(np.min(gaps), np.max(gaps)), density=True)
            p.setLogZero(2*edges[np.argmax(hist)])
            print("logZero", p.getLogZero())
            #particleColor[0] += 0.15
            #particleColor[1] -= 0.05
            #particleColor[2] -= 0.1
            print("current frame:", i)
        return gcf.artists
    anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=True)
    anim.save(path, writer='imagemagick', dpi=plt.gcf().dpi)
    p.save(dirName + os.sep + "endvideo", overwrite=True)

def makeDecompressionVideo(path, numParticles=256, numFrames=360, frameTime=40, faceColor=[0.1,0.5,0.8]):
    #load the initial packing
    p = pcp.Packing()
    p.load(dirName)
    #set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
    def animate(i):
        particleColor = faceColor
        gcf = plt.gcf()
        gcf.clear()
        ax = plt.gca()
        clearAxes(ax)
        setPhiTitle(ax,p.getPhi().astype(float))
        if(p.getPhi()>np.quad("1")):
            decompressionFreq = 20
        else:
            decompressionFreq = 40
        #minimize with FIRE
        p.minimizeFIRE(criticalForce = np.quad("1e-20"), maxIterations = 150)
        p.draw2DPacking(faceColor=faceColor)
        #print("\nmaxUnbalancedForce:", p.getMaxUnbalancedForce())
        if(i%decompressionFreq==0 and i!=0):
            overlaps = p.getSmallOverlaps().astype(float).data
            minOverlap = np.max(overlaps)
            print("minOverlap:", minOverlap)
            radDecrease = np.quad("1") - np.quad("0.2")*minOverlap
            phiDecrease = radDecrease**nDim
            print("packing fraction decrement:", phiDecrease)
            p.setPhi(p.getPhi()*phiDecrease)
            print("packing fraction:", p.getPhi())
            #particleColor[0] += 0.15
            #particleColor[1] -= 0.05
            #particleColor[2] -= 0.1
            print("current frame:", i)
        return gcf.artists
    anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=True)
    anim.save(path, writer='imagemagick', dpi=plt.gcf().dpi)
    p.save(dirName + os.sep + "endvideo", overwrite=True)

def makeLogMinimizedImage(path, numParticles, faceColor=[0.1,0.6,0.3]):
    p = pcp.Packing()
    p.load(dirName)
    # Make the figure
    fig = plt.figure(figsize=(5,5))
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
    ax = plt.gca()
    clearAxes(ax)
    #setPhiTitle(ax,phi)
    p.draw2DPacking(faceColor=faceColor)
    plt.tight_layout()
    plt.savefig(path, transparent=False, format = "pdf", pad_inches=0.01)
    #p.draw2DNetwork(network=p.getNeighbors(),color=[0.1,0.1,0.5],alpha=1,lineWidth=0.6)
    plt.show()

if __name__ == '__main__':
    nDim = 2
    dirName = sys.argv[1]
    which = sys.argv[2]
    cutDistance = np.quad("1")
    gapTh = np.quad("1e-6")

    if(which == "packing"):
        makeLogMinimizedImage(path="/home/farceri/Pictures/ssJammedPacking.pdf", numParticles=256, faceColor=[100/255,100/255,100/255])
    elif(which == "logMin"):
        makeLogMinimizationVideo(path="/home/farceri/Pictures/logMinimization.gif", numParticles=256, faceColor=[34/255,139/255,34/255])
    elif(which == "logComp"):
        makeLogCompressionVideo(path="/home/farceri/Pictures/hsCompression.gif", numParticles=256, faceColor=[100/255,100/255,100/255])
    elif(which == "decomp"):
        makeDecompressionVideo(path="/home/farceri/Pictures/ssDecompression.gif", numParticles=256, faceColor=[100/255,100/255,100/255])
