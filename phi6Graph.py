#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 18:00:15 2023

@author: violalum
"""
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import imp
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import numpy as np
import npquad
import idealGlass
import cpPoissonPoints as cp
import matplotlib.cbook
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
	plt.plot(xCoords,yCoords,'-',color=[0,0,0],lineWidth=1)


def makePhi6Colors2(p):
	phi = p.getPhi()
	p.setPhi(np.quad(.1)+phi)
	cv = cp.delaunayVectors(p).tocoo()
	p.setPhi(phi)
	contactVecs = cv.data.reshape(len(cv.data)//2,2).astype(float)
	thetas = np.arctan2(contactVecs[:,1], contactVecs[:,0])
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

def makePhi6Vectors(p): # pulled from idealGlass.py I think: modified to use delaunay vectors instead of contact vectors
	phi = p.getPhi()
	p.setPhi(np.quad(.1)+phi)
	cv = cp.delaunayVectors(p).tocoo()
	p.setPhi(phi)
	contactVecs = cv.data.reshape(len(cv.data)//2,2).astype(float)
	thetas = np.arctan2(contactVecs[:,1], contactVecs[:,0])
	iVals = cv.row.reshape(len(cv.row)//2,2)[:,0]
	numContacts = np.bincount(iVals, minlength = p.getNumParticles())
	phi6 = np.zeros(len(numContacts), dtype=complex)
	np.add.at(phi6, iVals, np.exp(6j*thetas))
	phi6 /= numContacts
	return [phi6.real.astype(float),phi6.imag.astype(float)]

def loadPack(p,directory): #Loads packing from directory into p and attempts to set lattice vectors
	p.load(directory)
	try:
		lv=np.loadtxt(f'{directory}/latticeVectors.dat')
	except:
		lv=np.array([[1,0],[0,1]])
	lv1Norm=lv[0]/np.sqrt(np.dot(lv[0],lv[0]))
	rotationMatrix= np.array([[lv1Norm[0],lv1Norm[1]],[-lv1Norm[1],lv1Norm[0]]])
	newVec=np.matmul(rotationMatrix.transpose(),lv)
	p.setLatticeVectors(newVec)
	p.setPositions(np.dot(p.getPositions(),rotationMatrix))
	p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
	p.minimizeFIRE('1e-20')

if __name__ == '__main__':
	n=1024
	plt.figure(figsize=(4,4))
#	directories=['finishedPackings/posMin-1',f'finishedPackings/idealPack{n}-1','posMin/posMin-1/isostatic','radMin/radMin-1/isostatic']
	directories=[f'finishedPackings/idealPack{n}-1',f'jumbledPackings/idealPack{n}-1/isostatic']
	labels=['A','B']#['C','D','A','B']
	for index in range(len(directories)):
		fig, ax=plt.subplots()
		p = pcp.Packing()
		loadPack(p,f'../idealPackingLibrary/{n}/{directories[index]}')#switch back to 4k after finishing this test
		colorMap=np.array(makePhi6Colors2(p))
		xy=p.getPositions().transpose().astype(float)
		arrows=makePhi6Vectors(p)
		p.draw2DPacking(faceColor=colorMap.astype(float),edgeColor=[.5,.5,.5],alpha=1,axis=ax,lineWidth=.25)
		ax.quiver(xy[0],xy[1],arrows[0],arrows[1],angles='xy',scale=np.sqrt(n))
		ax.axis('off')
		plt.text(.02,.04,labels[index],color='black',bbox= dict(boxStyle='square',alpha=1,facecolor=[.9,.9,.9]),fontsize=28)
		#inset axes: based on matplotlib docs
		x1, x2, y1, y2 = .3, .4, .3, .4 # subregion of the original image
		axins = ax.inset_axes([0.5, 0.5, 0.5, 0.5],xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
		p.draw2DPacking(faceColor=colorMap.astype(float),edgeColor=[.5,.5,.5],alpha=1,axis=axins,xBounds=[x1,x2],yBounds=[y1,y2],lineWidth=1)
		axins.quiver(xy[0],xy[1],arrows[0],arrows[1],angles='xy',scale=np.sqrt(n)*(x2-x1)/.75,width=.06)
		axins.set_aspect('equal')#'box')
		axins.get_xaxis().set_visible(False)
		axins.get_yaxis().set_visible(False)
		ax.indicate_inset_zoom(axins, edgeColor=[.3,.3,.3],lineWidth=2,alpha=1)
#		plt.show()
		figName=directories[index].split('/')[0]+'.'+directories[index].split('/')[1]
		fig.tight_layout(pad=0,w_pad=0,h_pad=0)
		[x.set_linewidth(2) for x in axins.spines.values()]
		[x.set_color([.3,.3,.3]) for x in axins.spines.values()]
#		plt.show()
		fig.savefig(f'../idealPackingLibrary/figures/{n}.{figName}-Phi6vector.pdf',pad_inches=0)
		fig.savefig(f'../idealPackingLibrary/figures/{n}.{figName}-Phi6vector.png',pad_inches=0)#,bbox_inches='tight',pad_inches=0)
		plt.clf()
		plt.close()
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
# 	plt.scatter(PHI, RHO, c=c,lineWidth=0)
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
# 	plt.scatter(PHI, RHO, c=c,lineWidth=0)
# 	plt.axis('off')
# 	plt.ylim((0,1))
# #	plt.ylabel('rho')
# #	plt.xlabel('theta')
# 	plt.savefig('cWheel2.pdf',bbox_inches='tight',pad_inches=0)
# 	plt.savefig('cWheel2.png',bbox_inches='tight',pad_inches=0)
# 
# 
# =============================================================================
