'''
Created by Francesco
28 May 2019
'''

import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import sys

def calcPolarizationVectors(waveVector):
    '''
    computes polarization vectors from a given wave vector
    args: d-dimensional wave vector

    returns: numPol polarization vectors (numPol = d)
    '''
    # TODO: Fix this to make an actual spanning set
    # This is a bad way to create a spanning set
    polarization = np.zeros((nDim, nDim))
    polarization[0] = waveVector
    # TODO: Turn this into a function!
    norm = np.linalg.norm(polarization[0])
    if(norm != 0):
        polarization[0] /= norm
    polarization[1] = np.array([-waveVector[1], waveVector[0], 0])
    norm = np.linalg.norm(polarization[1])
    if(norm != 0):
        polarization[1] /= norm
    polarization[2] = np.array([-waveVector[2], 0, waveVector[0]])
    norm = np.linalg.norm(polarization[2])
    if(norm != 0):
        polarization[2] /= norm
    polarization[2] -= np.dot(polarization[1], polarization[2]) * polarization[1]
    polarization[2] /= np.linalg.norm(polarization[2])
    norm = np.linalg.norm(polarization[2])
    if(norm != 0):
        polarization[2] /= norm
    return polarization

def calcPhononVectors(polarizationVectors, stablePos):
    '''
    returns a vector representation of a phonon given
    the polarization vectors and the positions of stable particles

    assumes that polarizationVectors[0] is the longitudinal polarization and
    so can be used as the waveVector
    args: numPol polarization vectors and stable positions

    returns: numPol vectors of length numStable*nDim
    '''
    #compute the phonon vector for each of the polarization vectors
    numPol = polarizationVectors.shape[0]
    numStable, nDim = stablePos.shape
    waveVector = polarizationVectors[0]
    modulation = np.exp(1j * np.dot(waveVector,stablePos.T)) / np.sqrt(numStable)
    # Phonons is a d by numStable by d matrix with
    # index0 = polarizationIndex
    # index1 = stableParticleIndex
    # index2 = dimensionIndex
    phonons = np.expand_dims(polarizationVectors,1) * np.expand_dims(modulation,1)
    # Return in flattened form for each polarization as d by numStable*d matrix
    # index0 = polarizationIndex
    # index1 = flattened degree of freedom index
    return phonons.reshape((numPol, numStable*nDim))

def phononProjection(eigenVector, phononVector):
    #compute the projection of each eigenvalue along this phonon
    return np.absolute(np.dot(eigenVector,phononVector))**2

def calcPhononProjection(waveVector, stablePos, eigenVectors):
    '''
    compute the projection of numStable*nDim eigenvectors onto a phonon
    with the same shape
    args: d-dimensional wave vector, positions of stable particles, eigenvectors

    returns: projection as a scalar
    '''
    numStable, nDim = stablePos.shape
    phonons = calcPhononVectors(calcPolarizationVectors(waveVector), stablePos)
    phononProjectionList = np.absolute(np.dot(eigenVectors.T, phonons.T))**2
    #phononProjectionList *= (phononProjectionList > 10/(nDim*numStable - nDim))
    return np.sum(phononProjectionList,1)

def calcPhononParameterList(stablePos, eigenVectors, nMax=5, threshold=2):
    # TODO: make work for all d
    x,y,z = np.meshgrid(range(nMax), range(nMax), range(nMax))
    phononParam = np.zeros(np.shape(eigenVectors[0]))
    for i,j,k in zip(x.flatten(), y.flatten(), z.flatten()):
        if (i>1 and j>1 and k>1):
            waveVector = 2*np.pi*np.array([i,j,k])
            phononParam += calcPhononProjection(waveVector, stablePos, eigenVectors, threshold)
    return phononParam


if __name__ == '__main__':
    nDim = int(sys.argv[1])
    numParticles = int(sys.argv[2])
    dirName = sys.argv[3]
    indexMaxMode = int(sys.argv[4])

    # Consider loading data using pcp.loadArray() stuff
    pos = np.loadtxt(dirName + "positions.dat")
    eigenvalues = np.loadtxt(dirName + "eigenvalues.dat")
    omegas = np.sqrt(eigenvalues[nDim:])
    omegas = omegas[:indexMaxMode]
    eigenVectors = np.loadtxt(dirName + "eigenvectors.dat")
    eigenVectors = eigenVectors[:,nDim:]
    eigenVectors = eigenVectors[:,:indexMaxMode] #consider only the low-frquency eigenmodes
    eigPart = int(len(eigenVectors[:,0])/nDim) #this is just a check
    stableList = np.loadtxt(dirName + "stableList.dat")
    numStable = sum(stableList==1)
    stablePos = pos[stableList==1,:]
    print("numStable:", numStable, "lenEigs:", eigPart)

    phononParam = calcPhononParameterList(stablePos, eigenVectors, nMax=7, threshold=1)

    savePhononParam = np.zeros((len(phononParam), 2))
    for i in range(len(phononParam)):
        savePhononParam[i] = [omegas[i], phononParam[i]]
    # TODO: Use pcp save array
    np.savetxt(dirName + "phononOrderParameter.dat", savePhononParam)
    plt.semilogx(omegas, phononParam, '.', alpha = 0.5)
    plt.show()
