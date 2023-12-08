'''
Created by Francesco
4 August 2019
'''

import numpy as np
import pandas as pd
import scipy
import scipy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage
np.set_printoptions(precision=5, suppress=True)
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import pyCudaPacking as pcp
import sys

def getIntenseList(eigenvector, factor=1.5):
    intenseList = np.zeros(numParticles)
    intensity = np.sqrt(np.sum(eigenvector**2, axis = 1))
    hist, edges = np.histogram(intensity, np.linspace(np.min(intensity), np.max(intensity)), density=True)
    edges = (edges[:-1] + edges[1:])/2
    firstSaddle = edges[np.argwhere(hist>0.5*np.max(hist))[0,0]]
    width = 2*(edges[np.argmax(hist)])
    tail = edges[-1] - 2*edges[np.argmax(hist)]
    print("width:", width, "tail:", tail)
    if(width < tail):
        threshold = edges[np.argmax(hist)] + 0.5*width
    else:
        print("THE MODE IS VERY PHONON-LIKE")
        threshold = edges[np.argmax(hist)]
    print("threshold:", threshold)
    intenseList[np.argwhere(intensity>threshold).flatten()] = 1
    plt.figure(1)
    plt.plot(edges, hist, '.')
    plt.plot(edges[np.argmax(hist)], np.max(hist))
    plt.plot(np.linspace(edges[0], width, 20), 0.5*np.max(hist)*np.ones(20))
    plt.plot(np.linspace(2*edges[np.argmax(hist)], edges[-1], 20), hist[np.argmax(edges)]*np.ones(20))
    return intenseList

def getLocalAffinityList(eigenvector, contacts, stableList):
    localAffinityList = np.zeros(numParticles)
    localAffinity = np.zeros(numParticles)
    localAffinityCount = np.zeros(numParticles)
    for i in range(numParticles):
        if(stableList[i] == 1):
            contactList = np.argwhere(contacts[i]==True).flatten()
            normi = scipy.linalg.norm(eigenvector[i])
            for j in range(contactList.shape[0]):
                normj = scipy.linalg.norm(eigenvector[j])
                affinity = np.absolute(np.dot(eigenvector[i], eigenvector[j]))/(normi*normj)
                if(affinity > 0):
                    localAffinity[i] += affinity
                    localAffinityCount[i] += 1
    localAffinity[localAffinityCount!=0] /= localAffinityCount[localAffinityCount!=0]
    print("total local affinity:", np.sum(localAffinity)/numParticles)
    localAffinityList[np.argwhere(localAffinity>0).flatten()] = 1
    plt.figure(1)
    plt.plot(localAffinity[excitedList==1], '.')
    return localAffinityList

def getClusters(excitedPos, maxDistance=0.5):
    Z = linkage(excitedPos, 'ward')
    return fcluster(Z, maxDistance, criterion='distance')

def getClusterDistribution(clusters, excitedRad, nDim):
    clusterVolume = np.zeros(np.max(clusters))
    clusterCount = np.zeros(np.max(clusters))
    for i in range(clusterVolume.shape[0]):
        clusterCount[i] = clusters[clusters==i+1].shape[0]
        clusterRad = excitedRad[clusters==i+1]
        if(nDim == 2):
            clusterVolume[i] = np.pi*np.sum(clusterRad**2)
        elif(nDim == 3):
            clusterVolume[i] = np.pi*(4/3)*np.sum(clusterRad**3)
        else:
            print("Function getClusterVolume works only for nDim = 2,3")
    return clusterCount, clusterVolume

def clusterExcitations(positions, stableList, eigenvector):
    #get cluster sizes
    clusters = getClusters(excitedPos)
    maxCluster = np.max(clusters)
    #get cluster populations and volumes
    clusterCount, clusterVolume = getClusterDistribution(clusters, excitedRad, nDim)
    #compute the total volume occupied by clusters
    totalVolume = np.sum(clusterVolume)
    print("total volume:", totalVolume)
    #compute the weighted average of the cluster volume
    meanVolume = np.sum(clusterVolume*clusterCount)/np.sum(clusterCount)
    return np.array([modeFrequency, maxCluster, np.sum(clusterCount), totalVolume, meanVolume, np.mean(clusterVolume), totalVolume*100/phi])

def getWindowListGradient(gridMax, positions, eigenvector):
    windowList = []
    gradIntensityList = []
    excitedWindows = []
    for i in np.linspace(0,1, gridMax):
        for j in np.linspace(0,1, gridMax):
            window = np.argwhere((positions[:,0]-i)**2+(positions[:,1]-j)**2 < 0.05**2)
            intensity = np.sqrt(np.sum(eigenvector[window]**2, axis = 2))
            intensity = intensity.reshape(intensity.shape[0])
            gradIntensityList.append(np.dot(np.gradient(intensity), np.gradient(intensity)))
            windowList.append(window.reshape(window.shape[0]))
    meanGradIntensity = np.mean(gradIntensityList)
    stdGradIntensity = np.std(gradIntensityList)
    for i in range(gridMax*gridMax):
        if(gradIntensityList[i] > meanGradIntensity + stdGradIntensity):
            excitedWindows.append(windowList[i])
    return np.array(excitedWindows)

def getAngle(v1, v2):
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

def getWindowAngles(windowEig, windowRadius):
    angles = np.zeros(windowEig.shape[0])
    axis = np.array([windowRadius/np.sqrt(2),windowRadius/np.sqrt(2)])
    previousAngle = 0
    for i in range(1,windowEig.shape[0]+1):
        angle = getAngle(axis, windowEig[i-1])
        angles[i-1] = angle - previousAngle
        previousAngle = angle
    return angles

def getWindowListAngles(gridMax, positions, eigenvector):
    excitedWindows = []
    maxAngle = np.pi/16
    windowRadius = 0.03
    for i in np.linspace(0,1, gridMax):
        for j in np.linspace(0,1, gridMax):
            window = np.argwhere((positions[:,0]-i)**2+(positions[:,1]-j)**2 < windowRadius**2)
            window = window.reshape(window.shape[0])
            angles = getWindowAngles(eigenvector[window], windowRadius)
            if((np.sum(angles) < maxAngle) and (np.sum(angles) > -maxAngle)):
                excitedWindows.append(window)
    return np.array(excitedWindows)

def getWindowListVortices(positions, eigenvector):
    excitedParticles = []
    maxRad = 0.1
    for i in range(positions.shape[0]):
        center = positions[i]
        centerIntensity = np.sqrt(np.sum(eigenvector[i]**2))
        windowRadius = 0.05
        while windowRadius < maxRad:
            window = np.argwhere(np.sum((positions-center)**2, axis=1) < windowRadius**2)
            window = window.reshape(window.shape[0])
            totalIntensity = []
            for p in window:
                intensity = np.sqrt(np.sum(eigenvector[p]**2))
                if(intensity > centerIntensity):
                    totalIntensity.append(intensity)
                    save = 1
                    for k in range(len(excitedParticles)):
                        if(p == excitedParticles[k]):
                            save = 0
                    if(save==1):
                        excitedParticles.append(p)
            if(np.mean(totalIntensity) > centerIntensity):
                windowRadius += 0.01
                centerIntensity = np.mean(totalIntensity)
            else:
                windowRadius = 0.1
    return excitedParticles

def getExcitedList(eigenvector, positions, stableList):
    indexList = []
    #excitedWindows = getWindowListGradient(gridMax, positions, eigenvector)
    #excitedWindows = getWindowListAngles(gridMax, positions, eigenvector)
    excitedParticles = getWindowListVortices(positions, eigenvector)
    print(len(excitedParticles))
    for i in range(len(excitedParticles)):
        if(stableList[excitedParticles[i]] == True):
            indexList.append(excitedParticles[i])
    print(len(indexList))
    return indexList


nDim = int(sys.argv[1])
numParticles = int(sys.argv[2])
dirName = sys.argv[3]
maxEig = int(sys.argv[4])
plot = sys.argv[5]

p = pcp.Packing()
p.load(dirName)
pos = p.getPositions()
pos = np.array(pos, dtype=np.float64)
rad = p.getRadii()
rad = np.array(rad, dtype=np.float64)
phi = p.getPhi()
contacts = p.getNeighbors().toarray()
stableList = np.loadtxt(dirName + "stableList.dat")
numStable = stableList[stableList==1].shape[0]
eigenvector = np.zeros((numParticles, nDim))
omegas = np.loadtxt(dirName + "omegas.dat", dtype = float)
stableEigenvectors = np.loadtxt(dirName + "eigenvectors.dat")
print(stableEigenvectors.shape[0], numStable)
stableEigenvectors = stableEigenvectors[:,:maxEig]
saveData = []
for whichEig in range(maxEig):
    stableEigenvector = stableEigenvectors[:,whichEig]
    stableEigenvector = stableEigenvector.reshape(numStable, nDim)
    eigenvector[stableList==1] = stableEigenvector
    director = np.absolute(np.mean(np.exp(2*1j*np.arctan2(eigenvector[:,1], eigenvector[:,0]))))
    #director = np.absolute(np.mean(np.sqrt(np.sum(eigenvector**2, axis=1))*np.mean(np.exp(2*1j*np.arctan2(eigenvector[:,1], eigenvector[:,0])))))
    print("mode", whichEig, "mode frequency:", omegas[whichEig], "director average:", director)
    #get particles participating to the localized motions
    #excitedList = getExcitedList(eigenvector, pos, stableList)
    #excitedPos = pos[excitedList]
    #excitedEigenvector = eigenvector[excitedList]
    #print("volume of excited particles:", np.pi*np.sum(rad[excitedList]**2))
    saveData.append([whichEig, omegas[whichEig], director])
    #plot the mode
    if(plot == "plot"):
        if(nDim == 2):
            plt.figure(whichEig, dpi = 100)
            ax = plt.axes()
            ax.set_aspect('equal')
            ax.quiver(pos[:,0], pos[:,1], eigenvector[:,0], eigenvector[:,1], color = 'y', alpha = 0.8)
            ax.quiver(excitedPos[:,0], excitedPos[:,1], excitedEigenvector[:,0], excitedEigenvector[:,1], color = 'k', alpha = 0.5)
            plt.pause(1)
        elif(nDim ==3):
            fig = plt.figure(dpi = 150)
            ax = fig.gca(projection = '3d')
            ax.set_aspect('equal')
            ax.quiver(pos[:,0], pos[:,1], pos[:,2], eigenvector[:,0], eigenvector[:,1], eigenvector[:,2], color = 'k', alpha = 0.5)
            ax.quiver(excitedPos[:,0], excitedPos[:,1], excitedPos[:,2], excitedEigenvector[:,0], excitedEigenvector[:,1], excitedEigenvector[:,2], color = 'r', alpha = 0.2)
        plt.show()
np.savetxt(dirName + "director.dat", saveData)
