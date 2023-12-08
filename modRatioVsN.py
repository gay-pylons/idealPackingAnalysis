#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 14:32:14 2022

@author: violalum

Sweeps idealPackingLibrary for moduli, saves stiffness matrices, moduli, and visual for moduli. 
Attempted to minimize duplicate dynamical matrix calculations.
"""
import numpy as np
import matplotlib.pyplot as plt
import pyCudaPacking as pcp
#import matplotlib.axes.Axes as ax
#import moduliTest as mt

def readProcessedPacking(packing,filename): #assumes packing that has easily readable moduli, correct # contacts, etc.
	packing.load(f'{filename}-processed')
	try:
		latVecs=np.loadtxt(f'{filename}-processed/latticeVectors.dat').astype(np.quad)
	except:
		latVecs=np.loadtxt(f'{filename}/latticeVectors.dat').astype(np.quad)
		np.savetxt(f'{filename}-processed/latticeVectors.dat',latVecs)
	packing.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
	packing.setLatticeVectors(latVecs)

def readUnprocessedIdealPacking(packing,filename): #assumes packing with incorrect number of contacts
	packing.load(filename)
	latVecs=np.loadtxt(f'{filename}/latticeVectors.dat').astype(np.quad)
	packing.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
	packing.setLatticeVectors(latVecs)
	packing.minimizeFIRE('1e-20')
	while(np.size(p.getContacts()) < 6*n):
		while np.size(p.getContacts()) != 6*n:
			packing.setPhi(packing.getPhi()+np.quad('1e-8')) #change to adaptive algorithm when possible
		packing.minimizeFIRE('1e-20')
	packing.save(f'{filename}-processed',overwrite=True)

def readUnprocessedPacking(packing,filename,dev=0):
	packing.load(filename)
	packing.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
	packing.setLatticeVectors(np.array([[1,0],[0,1]],dtype=np.quad))
	packing.minimizeFIRE('1e-20')
	for stableList, excess, phiC in  packing.isostaticitySearch(np.quad('.911'), packing.getPhi(), nStepsPerDecadePhi=2,deltaZ=0):
		if packing.getPressure() < np.quad('1e-6'):
			break
		print(2*(packing.getNumParticles()+1)-np.size(packing.getContacts())/2)
	packing.save(f'{filename}-processed',optionalData=['latticeVectors'])

def moduliPerParticle(packing,pName):
	shearStrain = np.array([0,1,0,0])
	bulkStrain = np.array([1,0,0,1])
	try: #save stiffness matrices to cut down on duped work
		stiffness = np.load(f'{pName}.stiffnessMatrix.npy')
	except:
		stiffness = packing.getStiffnessMatrix()
		np.save(f'{pName}.stiffnessMatrix.npy',stiffness)
	shear=np.matmul(stiffness.astype(float), shearStrain.astype(float))[1]/p.getNumParticles()
	bulk= np.sum(np.matmul(stiffness.astype(float), bulkStrain.astype(float))[[0,3]])/p.getNumParticles()/2
	return shear, bulk

def moduliPerParticle(n,pName):
	shearStrain = np.array([0,1,0,0])
	bulkStrain = np.array([1,0,0,1])
	stiffness = np.load(f'{pName}.stiffnessMatrix.npy')
	shear=np.matmul(stiffness.astype(float), shearStrain.astype(float))[1]/n
	bulk= np.sum(np.matmul(stiffness.astype(float), bulkStrain.astype(float))[[0,3]])/n/2
	return shear, bulk

def moduliPerParticlePressure(packing,pName,pressure):
	shearStrain = np.array([0,1,0,0])
	bulkStrain = np.array([1,0,0,1])
	try: #save stiffness matrices to cut down on duped work
		stiffness = np.load(f'{pName}-{pressure}.stiffnessMatrix.npy')
	except:
		stiffness = packing.getStiffnessMatrix()
		np.save(f'{pName}-{pressure}.stiffnessMatrix.npy',stiffness)
	shear=np.matmul(stiffness.astype(float), shearStrain.astype(float))[1]/packing.getNumParticles()
#	print(shear)
	bulk= np.sum(np.matmul(stiffness.astype(float), bulkStrain.astype(float))[[0,3]])/packing.getNumParticles()/2
	return shear, bulk

if __name__ == '__main__':
	dev= 1 #device number
	indexMax=10
	nList=np.array([16384,8196,4096,1024,256,128])
	idealK=[]
	idealG=[]
	idealP=[]
	for j in range(len(nList)):
		n=nList[j]
		if(j>=2):
			indexMax=10
		idealG.append([])
		idealK.append([])
		idealP.append([])
		for i in range(indexMax):
			fname=f'../idealPackingLibrary/{n}/finishedPackings/idealPack{n}-{i}'
# =============================================================================
# 			try:
# 				p=pcp.Packing()
# 				readProcessedPacking(p, fname)
# 			except:
# 				p=pcp.Packing(deviceNumber=dev)
# 				readUnprocessedIdealPacking(p, fname)
# 			print(np.size(p.getContacts())/n)
# 			print(n)
# =============================================================================
			shearMod, bulkMod = moduliPerParticle(n,fname)
			print(shearMod)
			print(bulkMod)
			idealG[j].append(bulkMod)
			idealK[j].append(shearMod)
#			idealP[j].append(p.getPressure())
			print(f'{n} complete')
	idealK=np.array(idealK)
	idealG=np.array(idealG)
#	idealP=np.array(idealP)
#	np.savez('ideal.data.npz',bulk=idealG,shear=idealK,pressure=idealP,n=nList)
	#flatten
	longNList=np.array([nList]*indexMax)
	nList=longNList.transpose().flatten()
	idealK=idealK.flatten()
	idealG=idealG.flatten()
#	plt.loglog(nList,idealG, '.', label='bulk Modulus Per Particle')
#	plt.loglog(nList,idealK, 'x', label='shear Modulus Per Particle')
	camN=np.array([64.0,128.0,256,512.0,1024.0,2048.0,4096,8192.0])
	camG=np.array([1.5694834756273817,1.529547625487689,1.4915473627886475,1.4755290891162551,1.4754508476908808,1.4708635492264133,1.469547980717028,1.4689983821064698])
	camK=np.array([0.1241439353440653,0.0672302428196559,0.04550735679581425,0.019702208538422136,0.010693519725198346,0.006332945122617293,0.002577659594771929,0.0015121321449186372])
	plt.figure(figsize=(6,5))
	plt.loglog(np.array([100,16000]),np.array([0.741566527095123,0.7415665270951238])/2.98649730664251,'--',color='darkred')
	plt.loglog(nList,idealK/idealG, '-o',label='circlePack Packings',color=[.5,.3,.75],linewidth=.5)
	plt.loglog(camN,camK/camG,'-o',label='marginal (approx)',color=[.3,.75,.5],linewidth=.5)
	plt.xlabel('N',fontsize=20)
	plt.ylabel('shear/bulk ratio',fontsize=20)
#	plt.legend()
	plt.tick_params(length=5)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.tight_layout()
	plt.savefig('finiteSizeScaling.svg')

