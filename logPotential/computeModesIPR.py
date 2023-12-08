'''
Created by Francesco
24 Aprile 2018
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
from scipy.sparse import linalg as spla
from scipy import io
import sys
import os
from os import path

def readPhi(nameDir, namePacking):
    with open(nameDir + os.sep + namePacking + os.sep + "scalars.dat") as file:
        for line in file:
            name, scalarString = line.strip().split("\t")
            if(name == "phi"):
                return float(scalarString)

def readHessian(nameDir):
    hessian = io.mmread(nameDir + os.sep + "stableHessian.mtx")
    #hessian = io.mmread(nameDir + ".cua")
    return hessian

def diagonalizeHessianSparse(hessian):
    eigs_large = spla.eigsh(hessian, N, which="LM", return_eigenvectors = False)
    eigs_small = spla.eigsh(hessian, N, which="SM", return_eigenvectors = False)
    eigenvalues = np.concatenate((eigs_small, eigs_large), axis=0)
    eigenvalues = np.sort(eigenvalues)
    return eigenvalues

def diagonalizeHessianDense(hessian):
    eigenvalues, eigenvectors = la.eigh(hessian.todense())
    return eigenvalues/np.mean(eigenvalues), eigenvectors

def computeIPR(eigenvectors):
    ipr = np.zeros(eigenvectors.shape[0]-nDim)
    for i in range(nDim, eigenvectors.shape[0]):
        eigvecNorm = np.linalg.norm(eigenvectors[:,i].reshape(int(eigenvectors.shape[0]/nDim), nDim), axis=1)
        forthMoment = np.sum(eigvecNorm**4)
        secondMoment = np.sum(eigvecNorm**2)
        ipr[i-nDim] = forthMoment/secondMoment**2
    return ipr

def computeIPRBinned(omegas, ipr):
    numBins = 100
    bins = np.geomspace(np.min(omegas), np.max(omegas), numBins)
    iprBins = np.zeros(numBins)
    iprCount = np.zeros(numBins)
    for i in range(len(omegas)):
        for k in range(1,numBins):
            if(omegas[i] > bins[k-1] and omegas[i] <= bins[k]):
                iprBins[k] += ipr[i]
                iprCount[k] += 1
    saveIPR = np.zeros((numBins, 2))
    for i in range(numBins):
        iprBinned = 0
        if(iprCount[i] > 0):
            iprBinned = iprBins[i]/iprCount[i]
        saveIPR[i] = [bins[i], iprBinned]
    return saveIPR


nDim = int(sys.argv[1])
numParticles = int(sys.argv[2])
numEigs = int(numParticles*nDim) #number of eigenvalues of the hessian given numParticles and nDim
N = int(numEigs/2) #number of eigenvalues to compute using the sparse computation, half eigs from above and half eigs from below
dirPath = sys.argv[3]
dirData = sys.argv[4]
dirName = dirPath + os.sep + dirData
saveEigs = sys.argv[5]
numSamples = int(sys.argv[6])
plot = sys.argv[7]
#whichEig = int(sys.argv[7])
prTh = 5
prExt = 0.2

if(numSamples == 1):
    if(path.exists(dirName + os.sep + "omegas.dat") == True and saveEigs != "saveEigs"):
        print("already diagonalized")
        omegas = np.loadtxt(dirName + os.sep + "omegas.dat")
        ipr = np.loadtxt(dirName + os.sep + "ipr.dat")
        if(path.exists(dirName + os.sep + "eigenvectors.dat") == True and nDim == 2):
            print("eigenvectors are saved")
            eigenvectors = np.loadtxt(dirName + os.sep + "eigenvectors.dat")
            eigenvectors = eigenvectors.reshape(eigenvectors.shape[0], int(eigenvectors.shape[0]/nDim), nDim)
            director = np.mean(np.absolute(np.sqrt(np.sum(eigenvectors**2, axis=2))*np.exp(2*1j*np.arctan2(eigenvectors[:,:,1], eigenvectors[:,:,0]))), axis=1)
            if(plot == "plot"):
                plt.figure(3)
                plt.semilogx(omegas, director[nDim:], 'v')
                np.savetxt(dirName + os.sep + "director.dat", director)
    else:
        hessian = readHessian(dirName)
        eigenvalues, eigenvectors = diagonalizeHessianDense(hessian)
        if(saveEigs == "saveEigs"):
            np.savetxt(dirName + os.sep + "eigenvalues.dat", eigenvalues)
            np.savetxt(dirName + os.sep + "eigenvectors.dat", eigenvectors)
        omegas = np.sqrt(eigenvalues[nDim:])
        print("mean omega:", np.mean(omegas))
        ipr = computeIPR(eigenvectors)
        np.savetxt(dirName + os.sep + "omegas.dat", omegas)
        np.savetxt(dirName + os.sep + "ipr.dat", ipr)
        if(nDim == 2):
            eigenvectors = eigenvectors.reshape(eigenvectors.shape[0], int(eigenvectors.shape[0]/nDim), nDim)
            director = np.mean(np.absolute(np.sqrt(np.sum(eigenvectors**2, axis=2))*np.exp(2*1j*np.arctan2(eigenvectors[:,:,1], eigenvectors[:,:,0]))), axis=1)
            pdf, edges = np.histogram(director, np.geomspace(np.min(director), np.max(director)))
            edges = (edges[1:]+edges[:-1])/2
            if(plot == "plot"):
                plt.figure(3)
                plt.plot(edges, pdf, 'v')
                np.savetxt(dirName + os.sep + "director.dat", director)
    if(plot == "plot"):
        plt.figure(1)
        plt.loglog(omegas, 1/(ipr*numParticles), 'o', alpha=0.5)
        #plt.loglog(omegas[:whichEig], 1/(ipr[:whichEig]*numParticles), 'o', alpha=0.5)
        plt.figure(2)
        pdf, edges = np.histogram(omegas, np.geomspace(np.min(omegas), np.max(omegas)), density=True)
        edges = (edges[:-1] + edges[1:])/2
        plt.loglog(edges, pdf, '^')
        plt.show()

elif(numSamples > 1):
    phiJ = np.loadtxt(dirPath + "phiJ.dat")
    omegasSamples = np.zeros(numEigs*numSamples)
    iprSamples = np.zeros(numEigs*numSamples)
    totalLength = 0
    for dirPacking in os.listdir(dirName):
        if(path.isdir(dirName + os.sep + dirPacking)):
            print("sample:", dirPacking)
            phi = readPhi(dirName, dirPacking)
            #phi = phiJ[int(dirPacking),1]
            if(path.exists(dirName + os.sep + dirPacking + os.sep + "omegas.dat") == True):
                omegas = np.loadtxt(dirName + os.sep + dirPacking + os.sep + "omegas.dat")
                ipr = np.loadtxt(dirName + os.sep + dirPacking + os.sep + "ipr.dat")
                if(plot == "plot"):
                    plt.figure(1)
                    plt.loglog(omegas, 1/(ipr*8192), 'o', alpha=0.5)
                    pdf, edges = np.histogram(omegas, np.geomspace(np.min(omegas), np.max(omegas)), density=True)
                    edges = (edges[:-1] + edges[1:])/2
                    plt.loglog(edges, pdf, '^')
                    plt.pause(2)
            else:
                hessian = readHessian(dirName + os.sep + dirPacking)
                if(os.path.getsize(dirName + os.sep + dirPacking + os.sep + "stableHessian.mtx") > 100):
                    eigenvalues, eigenvectors = diagonalizeHessianDense(hessian)
                    eigenvalues = eigenvalues[nDim:]
                    print(eigenvalues[eigenvalues<0])
                    omegas = np.sqrt(eigenvalues)
                    ipr = computeIPR(eigenvectors)
                    np.savetxt(dirName + os.sep + dirPacking + os.sep + "omegas.dat", omegas)
                    np.savetxt(dirName + os.sep + dirPacking + os.sep + "ipr.dat", ipr)

            #omegas = omegas[1/ipr>prTh]
            #ipr = ipr[1/ipr>prTh]
            #pr = 1/(ipr*numParticles)
            #omegas = omegas/omegas[pr>prExt][-1]
            #omegas /= np.mean(omegas)
            #omegas /= ((phiJ[int(dirPacking)]-phi)/phiJ[int(dirPacking)])**0.5
            print("distance from jamming:",(phiJ[int(dirPacking)]-phi))
            currentLength = len(omegas)
            print(currentLength)
            omegasSamples[totalLength:totalLength+currentLength] = omegas
            iprSamples[totalLength:totalLength+currentLength] = ipr
            totalLength += len(omegas)

    omegasSamples = omegasSamples[omegasSamples!=0]
    iprSamples = iprSamples[iprSamples!=0]
    np.savetxt(dirName + "omegasSamples.dat", omegasSamples)
    np.savetxt(dirName + "iprSamples.dat", iprSamples)
    print(totalLength, len(omegasSamples), len(iprSamples))
    if(plot == "plot"):
        plt.title(dirData, fontsize = 15)
        plt.xlabel("$\omega/\omega^*$", fontsize = 15)
        plt.ylabel("PR, PDF", fontsize = 15)
        plt.figure(2)
        pdf, edges = np.histogram(omegasSamples, np.geomspace(np.min(omegasSamples), np.max(omegasSamples), 100), density=True)
        edges = (edges[:-1] + edges[1:])/2
        plt.loglog(edges, pdf, 'v')
        plt.title(dirData, fontsize = 15)
        plt.xlabel("$\omega/\omega^*$", fontsize = 15)
        plt.ylabel("PDF", fontsize = 15)
        plt.show()
