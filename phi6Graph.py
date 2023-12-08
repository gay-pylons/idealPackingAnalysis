#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 18:00:15 2023

@author: violalum
"""
import imp
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import numpy as np
import npquad
import idealGlass
import cpRandomDelaunayTriangulation as cp
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import math
import colorsys

def colorMapBasic(phi6Vec):
	length=np.exp(np.dot(phi6Vec,phi6Vec)*np.log(2))-1
	angle=np.arctan2(phi6Vec[0],phi6Vec[1])
# =============================================================================
# 	if(angle < np.pi/3):
# 		return length*np.array([1,angle*3/np.pi,0])
# 	elif(angle < 2*np.pi/3):
# 		return length*np.array([2-3*angle/np.pi,1,0])
# 	elif(angle < np.pi):
# 		return length*np.array([0,1,3*angle/np.pi-2])
# 	elif(angle < 4*np.pi/3):
# 		return length*np.array([0,4-angle*3/np.pi,1])
# 	elif(angle < 5*np.pi/3):
# 		return length*np.array([angle*3/np.pi-4,0,1])
# 	else:
# 		return length*np.array([1,0,6-angle*3/np.pi])
# =============================================================================
	return np.clip((np.array(colorsys.hsv_to_rgb(angle/(2*np.pi)-.1,length,.7))+1)/2,0,1)
#	return colorMaprTheta2(length, angle)

def colorMaprTheta(length,angle):
#	hsv_to_rgb(angle, length, 1)
	if(angle < np.pi/3):
		return length*np.array([1,(angle*3./np.pi),0])
	elif(angle < 2*np.pi/3):
		return length*np.array([(2-3.*angle/np.pi),1,0])
	elif(angle < np.pi):
		return length*np.array([0,1,(3.*angle/np.pi-2)])
	elif(angle < 4*np.pi/3):
		return length*np.array([0,(4-angle*3./np.pi),1])
	elif(angle < 5*np.pi/3):
		return length*np.array([(angle*3./np.pi-4),0,1])
	else:
		return length*np.array([	1,0,(6-angle*3./np.pi)])
	
def colorMaprTheta2(length,angle):
#	hsv_to_rgb(angle, length, 1)
	if(angle < np.pi/2):
		return length*np.array([1,(angle*2./np.pi),1-(angle*2./np.pi)])
	elif(angle < np.pi):
		return length*np.array([(2-(angle*2./np.pi))**2,1,(angle*2./np.pi)-1])
	elif(angle < 3*np.pi/2):
		return length*np.array([0,(3-(angle*2./np.pi)),1])
	else:
		return length*np.array([((angle*2./np.pi)-3)**2,0,1])

def drawBox(corner1,corner2):
	xCoords=np.array([corner1[0],corner2[0],corner2[0],corner1[0],corner1[0]])
	yCoords=np.array([corner1[0],corner1[0],corner2[0],corner2[0],corner1[0]])
	plt.plot(xCoords,yCoords,'-',color=[0,0,0])

def makePhi6Colors2(p):
	phi = p.getPhi()
	p.setPhi(np.quad(.1)+phi)
	cv = cp.delaunayVectors(p)
#	print(cv)
	p.setPhi(phi)
	contactVecs = cv.data.reshape(len(cv.data)//2,2).astype(float)
	thetas = np.arctan2(contactVecs[:,1], contactVecs[:,0])
	#NOTE: This will fail if there are rattlers!
	iVals = cv.row.reshape(len(cv.row)//2,2)[:,0]
	numContacts = np.bincount(iVals, minlength = p.getNumParticles())
	phi6 = np.zeros(len(numContacts), dtype=complex)
	np.add.at(phi6, iVals, np.exp(6j*thetas))
	phi6 /= np.max(np.abs(phi6))
#	print(np.max(np.abs(phi6)))
#	np.array([colorMapBasic([x.real,x.imag]) for x in phi6])
	return [colorMapBasic([x.real,x.imag]) for x in phi6]

def makeSizeColors(p):
	radii=p.getRadii().astype(float)
	normedRadii=radii-np.min(radii)
	normedRadii=np.clip(normedRadii/np.max(normedRadii),0,1)
	zeroSize=np.array([.85,.95,.85])
	dropScale=np.array([.3,.7,.1])
	return [zeroSize-r*dropScale for r in normedRadii]

def makePhi6Vectors(p):
	phi = p.getPhi()
	p.setPhi(np.quad(.1)+phi)
	cv = cp.delaunayVectors(p)
	p.setPhi(phi)
	contactVecs = cv.data.reshape(len(cv.data)//2,2).astype(float)
	thetas = np.arctan2(contactVecs[:,1], contactVecs[:,0])
	#NOTE: This will fail if there are rattlers!
	iVals = cv.row.reshape(len(cv.row)//2,2)[:,0]
	numContacts = np.bincount(iVals, minlength = p.getNumParticles())
	phi6 = np.zeros(len(numContacts), dtype=complex)
	np.add.at(phi6, iVals, np.exp(6j*thetas))
	phi6 /= np.max(np.abs(phi6))
	return [phi6.real.astype(float),phi6.imag.astype(float)]

def loadPack(p,directory): #Loads packing from directory into p and attempts to set lattice vectors
	p.load(directory)
	try:
		lv=np.loadtxt(f'{directory}/latticeVectors.dat')
		lv1Norm=lv[0]/np.sqrt(np.dot(lv[0],lv[0]))
		#print(lv1Norm)
		rotationMatrix= np.array([[lv1Norm[0],lv1Norm[1]],[-lv1Norm[1],lv1Norm[0]]])
		newVec=np.matmul(rotationMatrix.transpose(),lv)
	#	print(newVec)
		p.setLatticeVectors(newVec)
		p.setPositions(np.dot(p.getPositions(),rotationMatrix))
		p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
		p.minimizeFIRE('1e-20')
	except:
		p.minimizeFIRE('1e-20')

if __name__ == '__main__':
	n=4096
	directories=['finishedPackings/posMin-1',f'finishedPackings/idealPack{n}-1','posMin/posMin-1','radMin/radMin-1']
	for directory in directories:
		p = pcp.Packing()
		loadPack(p,f'../idealPackingLibrary/{n}/{directory}')#switch back to 4k after finishing this test
		print(p.getPhi())
		colorMap=np.array(makePhi6Colors2(p))#/np.array([1.2,1.4,1])
		fig = plt.figure()
		p.draw2DPacking(faceColor=colorMap.astype(float),edgeColor=None,alpha=1)
		xy=p.getPositions().transpose().astype(float)
		arrows=makePhi6Vectors(p)
		plt.axis('off')
		p.draw2DPacking(faceColor=colorMap.astype(float),edgeColor=None,alpha=1)
		plt.quiver(xy[0],xy[1],arrows[0],arrows[1],angles='xy')
		plt.savefig(f'../idealPackingLibrary/{n}/{directory}-Phi6vector.pdf',bbox_inches='tight',pad_inches=0)
		plt.savefig(f'../idealPackingLibrary/{n}/{directory}-Phi6vector.png',bbox_inches='tight',pad_inches=0)
		plt.clf()
# =============================================================================
# 	plt.clf()
# 	fig = plt.figure()
# 	ax = plt.subplot(projection='polar')
# 	rho = np.linspace(0,1,100) # Radius of 1, distance from center to outer edgephi = np.linspace(0, math.pi*2.,1000 ) # in radians, one full circle
# 	phi = np.linspace(0, math.pi*2.,500) # in radians, one full circle
# 	RHO, PHI = np.meshgrid(rho,phi) # get every combination of rho and phi
# 	RHO=RHO.flatten()
# 	PHI=PHI.flatten()
# 	c=np.array([colorMaprTheta(RHO[i],PHI[i]) for i in range(len(RHO))])
# 	c=np.clip((c+1)/2,0,1)
# 	print(len(PHI))
# 	print(len(RHO))
# 	print(len(c))
# 	plt.scatter(PHI, RHO, c=c,linewidth=0)
# 	plt.axis('off')
# 	plt.ylim((0,1))
# #	plt.ylabel('rho')
# #	plt.xlabel('theta')
# 	plt.savefig('cWheel.pdf',bbox_inches='tight',pad_inches=0)
# 	plt.savefig('cWheel.png',bbox_inches='tight',pad_inches=0)
# =============================================================================
# =============================================================================
# 	plt.clf()
# 	fig = plt.figure()
# 	ax = plt.subplot(projection='polar')
# 	rho = np.linspace(0,1,100) # Radius of 1, distance from center to outer edgephi = np.linspace(0, math.pi*2.,1000 ) # in radians, one full circle
# 	phi = np.linspace(0, math.pi*2.,500) # in radians, one full circle
# 	RHO, PHI = np.meshgrid(rho,phi) # get every combination of rho and phi
# 	RHO=RHO.flatten()
# 	PHI=PHI.flatten()
# 	c=np.array([colorMaprTheta2(RHO[i],PHI[i]) for i in range(len(RHO))])
# 	c=np.clip((c+1)/2,0,1)
# 	print(len(PHI))
# 	print(len(RHO))
# 	print(len(c))
# 	plt.scatter(PHI, RHO, c=c,linewidth=0)
# 	plt.axis('off')
# 	plt.ylim((0,1))
# #	plt.ylabel('rho')
# #	plt.xlabel('theta')
# 	plt.savefig('cWheel2.pdf',bbox_inches='tight',pad_inches=0)
# 	plt.savefig('cWheel2.png',bbox_inches='tight',pad_inches=0)
# 
# 
# =============================================================================
