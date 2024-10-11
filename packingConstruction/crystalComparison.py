#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:37:06 2024

@author: violalum
"""
import matplotlib
#matplotlib.use('agg')
import imp
from importlib.machinery import SourceFileLoader
pcp = SourceFileLoader("pyCudaPacking","/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py").load_module()
import numpy as np
import matplotlib.pyplot as plt

def wrapPosIntoBox(p):
	pos=p.getPositions()
	ilv = p.getInverseLatticeVectors()
	lv=p.getLatticeVectors()
	newPos = np.dot(np.dot(pos, ilv.T) % 1, lv.T)
	p.setPositions(newPos)

def truePacking(p):
	rota0=p.getLatticeVectors()[0]
	rota0=rota0.astype(float)/np.linalg.norm(rota0.astype(float))
	rotaMatrix=np.array([rota0,[-rota0[1],rota0[0]]],dtype=np.quad)
	newPos=np.dot(p.getPositions(),rotaMatrix)
	newLVs=np.dot(rotaMatrix.T,p.getLatticeVectors())
	print(newLVs)
	p.setPositions(newPos)
	p.setLatticeVectors(newLVs)

def centers(p,cIndex):
	position=p.getPositions()
	if type(cIndex)==int:
		center = position[cIndex]
	else:
		center = np.mean(position[cIndex],axis=0)
	return center

def distances(p,cIndex):
	distance=p.getDistances()
	if type(cIndex)==int:
		distance = distance[cIndex]
	else:
		distance = np.mean(distance[cIndex],axis=0)
	return distance

def getDisplacements(p,geometryType,cIndex):
	if geometryType==pcp.enums.geometryEnum.latticeVectors:
		truePacking(p)
	else:
		box=p.getBoxSize()
		lvs=np.array([[box[0],0],[0,box[1]]])
		p.setLatticeVectors(lvs)
	wrapPosIntoBox(p)
	positions=p.getPositions()-centers(p,cIndex)+np.mean(p.getInverseLatticeVectors(),axis=0)
	p.setPositions(positions)
	wrapPosIntoBox(p)
	p.minimizeFIRE('1e-20')
	positions=p.getPositions()-centers(p,cIndex)
	return positions

def get3Displacements(directory,cIndex,geometryType=pcp.enums.geometryEnum.latticeVectors):
	p=pcp.Packing()
	p.load(directory)
	p.setGeometryType(geometryType)
	distance = distances(p,cIndex)
	radii=p.getRadii()
	positions=getDisplacements(p,geometryType,cIndex)
	print(centers(p,cIndex))

	return distance, radii, positions

if __name__ == '__main__':
	directories= ['../idealPackingLibrary/986/hexSeeds/singleDefect','../idealPackingLibrary/986/finishedPackings/singleDefect']
	centerIndices=np.array([5, 7]) #circlePack indices-1 since CP doesn't have zeroes
	seedDistance,seedRadii,seedPositions=get3Displacements(directories[0],cIndex=centerIndices,geometryType=pcp.enums.geometryEnum.nonSquare)
	defectDistance,defectRadii,defectPositions=get3Displacements(directories[1],cIndex=centerIndices)
	defectPositions=np.dot(defectPositions,np.array([[0,-1],[1,0]]))
	arrows=(defectPositions-seedPositions).T.astype(float)
	print(arrows.T[6]+arrows.T[8])
	print(np.max(np.abs(arrows)))
	arrows[np.abs(arrows)>.5]=0
	locations=seedPositions.T.astype(float)
	plt.quiver(locations[0],locations[1],arrows[0],arrows[1],angles='xy')
	plt.gca().set_aspect('equal')
	plt.savefig('../idealPackingLibrary/figures/dispArrows.png')
	plt.clf()
	radiiChange=np.abs((defectRadii-seedRadii).astype(float))
	positionChange=np.linalg.norm((defectPositions-seedPositions).astype(float),axis=1)
	plt.loglog(seedDistance,positionChange,'x',alpha=.5)
	plt.loglog(seedDistance,radiiChange,'^',fillstyle='none',alpha=.5)
	plt.loglog(seedDistance,defectDistance,'.',fillstyle='none',alpha=.5)
	guidex=np.array([.1,.5])
	guidey=1e-5*guidex**(-1)
	plt.plot(guidex,guidey,'--',color='black')
	plt.xlabel('distance')
	plt.ylabel('displacement')
	plt.savefig('../idealPackingLibrary/figures/singleDefectDisplacements.png')
#	plt.show()
