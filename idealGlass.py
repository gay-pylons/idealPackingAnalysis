# -*- coding: utf-8 -*-
# @Author: Eric Corwin <ecorwin>
# @Date:   2022-01-09T13:18:40-08:00
# @Email:  eric.corwin@gmail.com
# @Filename: idealGlass.py
# @Last modified by: Viola Bolton-Lum @ vlum@uoregon.edu
# @Last modified time: 2023-09-18T18:02:54-07:00

import imp
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import logPotential.computeStructureFactor as struct
import numpy as np
import npquad
import os
import matplotlib.pyplot as plt


def getVoronoiContactVectors(p, radical=False):
	voronoiNeighbors = p.delaunayNeighbors(radical=True)


def makePhi6Colors(p):
	phi = p.getPhi()
	p.setPhi(np.quad(.1)+phi)
	cv = p.getContactVectors(normalized=True, stable=True)
	print(cv)
	p.setPhi(phi)
	contactVecs = cv.data.reshape(len(cv.data)//2,2).astype(float)
	thetas = np.arctan2(contactVecs[:,1], contactVecs[:,0])
	#NOTE: This will fail if there are rattlers!
	iVals = cv.row.reshape(len(cv.row)//2,2)[:,0]
	numContacts = np.bincount(iVals, minlength = p.getNumParticles())
	phi6 = np.zeros(len(numContacts), dtype=complex)
	np.add.at(phi6, iVals, np.exp(6j*thetas))
	phi6 /= numContacts
	return [[(x.real+1)/2, (x.imag+1)/2, np.abs(x)] for x in phi6]


def readAllScalars(topDir):
	allDirs = next(os.walk(topDir))[1]
	phi = []
	pressure = []
	excessContacts = []
	for d in allDirs:
		phi.append(np.quad(pcp.fileIO.loadScalar(f'{topDir}/{d}', 'phi')))
		pressure.append(np.quad(pcp.fileIO.loadScalar(f'{topDir}/{d}', 'pressure')))
		excessContacts.append(np.quad(pcp.fileIO.loadScalar(f'{topDir}/{d}', 'excessContacts')))
	return np.array(phi), np.array(pressure), np.array(excessContacts)

def fitPhiC(phi, pressure):
	prefactor, phiC = np.polyfit(pressure.astype(float), phi.astype(float), 1, w = pressure.astype(float)**-2)
	# phi = prefactor*pressure + phiC
	return phiC
import pyCudaPacking as pcp
def computePhiC(topDir):
	phi, pressure, _ = readAllScalars(topDir)
	return fitPhiC(phi, pressure)

def computeAllPhiC(topDir, N, polydispersity):
	phiC = []
	for pd in polydispersity:
		phiC.append(computePhiC(f'{topDir}/{N}-{pd:.2f}'))
	return np.array(phiC)

def makeStartingConfigs(topDir, nList = None, pdList = None, hexRatio = False, deviceNumber=2, idList = False):
	saveList = ['pressure', 'energy', 'excessContacts']
	if not nList:
		nList = [128, 1024, 8192, 65536]
	if not pdList:
		pdList = np.arange(0.02, 0.32, 0.02)
	for n in nList:
		for idx, pd in enumerate(pdList):
			print(n,pd)
			p = pcp.Packing(nDim = 2, deviceNumber=deviceNumber, numParticles=n)
			if hexRatio:
				p.setBoxSize(np.array([2, np.sqrt(np.quad(3))]))
			p.setRandomPositions()
			p.setLogNormalRadii(polyDispersity=f'{pd:.2f}')
			p.setPhi('0.95')
			p.minimizeFIRE('1e-20')
			if not idList:
				p.save(f'{topDir}/{n}-{pd:.2f}', saveList)
			else:
				p.save(f'{topDir}/{n}-{pd:.2f}-{idx}', saveList)

def runRegular(topDir, configList, deviceNumber):
	saveList = ['pressure', 'energy', 'excessContacts']
	for config in configList:
		p = pcp.Packing(deviceNumber=deviceNumber)
		p.load(f'{topDir}/{config}')
		phiStart = p.getPhi()
		phiCEstimate = np.quad('.91')
		nStepsPerDecadePhi = 10

		for stableList, excess, phiC in p.isostaticitySearch(phiCEstimate, phiStart, nStepsPerDecadePhi=nStepsPerDecadePhi, maxDeltaZ=0):
			# print(p.getPhi(), p.getEnergy(), p.getPressure(), excess, phiC)
			print(p.getPressure(), p.getPhi(), phiC, p.getPhi() - phiC, p.getPressure()/(p.getPhi() - phiC))
			#if p.getPressure() < np.quad('0.0045'):
			#	break
		# p.save(f'{topDir}/regularGlass/{config}/{p.getPhi()}')

def runToBottom(topDir, startingConfig, deviceNumber = 2, pBottom = '1e-12', phiC = '0.91'):
	saveList = ['pressure', 'energy', 'excessContacts']
	p = pcp.Packing(deviceNumber=deviceNumber)
	p.load(startingConfig)
	# moments = [-3, -2, -1, 1, 2, 3, 6]
	moments = [-2, 2, 4]
	p.setRadConstraintMoments(moments)
	phiStart = p.getPhi()
	phiCEstimate = np.quad(phiC)
	nStepsPerDecadePhi = 10

	# for stableList, excess, phiC in p.isostaticitySearch(phiCEstimate, phiStart, nStepsPerDecadePhi=nStepsPerDecadePhi, maxDeltaZ=p.getNumParticles() - len(moments)):
	for stableList, excess, phiC in p.isostaticitySearch(phiCEstimate, phiStart, nStepsPerDecadePhi=nStepsPerDecadePhi, maxDeltaZ=0):
		# Reset the radii to force a recalculation of Phi
		r = p.getRadii()
		p.setRadii(r)
		currentPhi = p.getPhi()
		p.save(f'{topDir}/{currentPhi}', saveList)
		print(p.getPhi(), p.getEnergy(), p.getPressure(), excess, np.std(r.astype(float))/np.mean(r.astype(float)), phiC)
		if p.getPressure() < np.quad(pBottom):
			break



def runFromStartingConfigs(topDir, nList, pdList, deviceNumber = 2, idList = False):
	saveList = ['pressure', 'energy', 'excessContacts']
	for n in nList:
		for idx, pd in enumerate(pdList):
			if not os.path.isdir(f'{topDir}/{n}-{pd:.2f}-{idx}'):
				print(n,pd)
				p = pcp.Packing(deviceNumber=deviceNumber)
				if not idList:
					p.load(f'{topDir}/startingConfigs/{n}-{pd:.2f}/')
				else:
					p.load(f'{topDir}/startingConfigs/{n}-{pd:.2f}-{idx}/')
				moments = [-3, -2, -1, 1, 2, 3, 6]
				p.setRadConstraintMoments(moments)
				phiStart = p.getPhi()
				phiCEstimate = np.quad('.91')
				nStepsPerDecadePhi = 10

				# for stableList, excess, phiC in p.isostaticitySearch(phiCEstimate, phiStart, nStepsPerDecadePhi=nStepsPerDecadePhi, maxDeltaZ=p.getNumParticles() - len(moments)):
				numSteps = 0
				for stableList, excess, phiC in p.isostaticitySearch(phiCEstimate, phiStart, nStepsPerDecadePhi=nStepsPerDecadePhi, maxDeltaZ=0):
					numSteps += 1
					r = p.getRadii()
					p.setRadii(r)
					currentPhi = p.getPhi()
					if not idList:
						p.save(f'{topDir}/{n}-{pd:.2f}/{currentPhi}', saveList)
					else:
						p.save(f'{topDir}/{n}-{pd:.2f}-{idx}/{currentPhi}', saveList)
					print(p.getPhi(), p.getEnergy(), p.getPressure(), excess, np.std(r.astype(float))/np.mean(r.astype(float)), phiC)
					if p.getPressure() < np.quad('1e-5') or numSteps > 10:
						break

def computeAverageSofK(packingList, kbin = None, deviceNumber=2, latticeVectors=None,wibbled=None):
	if kbin is None:
		kbin = np.logspace(np.log10(5), np.log10(1000),100, dtype = np.quad)
	structureFactors = []
	for packingDir in packingList:
		meanRadius=np.mean(np.loadtxt(f'{packingDir}/radii.dat')).astype(float)
		try:
			sFactors=pcp.fileIO.load2DArray(f'{packingDir}/{len(kbin)}sOfK.dat',np.quad)
		except:
			p = pcp.Packing(deviceNumber=deviceNumber)
			p.load(packingDir)
			if(wibbled):
				pos = p.getPositions()
				meanRad = np.mean(p.getRadii())
				scale = np.quad('.1') # Or whatever scale you want to set for the displacements
				p.setPositions(pos + pcp.quadMath.gaussianRandomQuad(shape=pos.shape) * meanRad * scale)
				plt.clf()
				p.draw2DPacking()
				plt.savefig('../idealPackingLibrary/figures/wibbledPack.png')
			if(latticeVectors):
				p.setLatticeVectors(pcp.fileIO.load2DArray( f'{packingDir}/latticeVectors.dat' , dtype=np.quad))
				ilv = p.getInverseLatticeVectors()
				lvPos = np.dot(ilv, p.getPositions().T).T
				p.setPositions(lvPos.copy())
			sFactors=struct.computeBinnedStructureFactor(p, kbin).astype(float)
			pcp.fileIO.save2DArray(f'{packingDir}/{len(kbin)}sOfK.dat',sFactors)
		print(sFactors[:,1])
		sFactors=sFactors*np.array([meanRadius,2*np.pi/meanRadius])
		structureFactors.append(sFactors)
	return structureFactors
