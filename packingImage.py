#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 18:00:15 2023

@author: violalum
"""
import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import imp
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import numpy as np
import npquad
import idealGlass
from packingConstruction import makeTriangulationFromPacking as cp
import matplotlib.cbook
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import math
import colorsys

def colorMapBasic(phi6Vec):
	length=np.diag(np.exp(np.dot(phi6Vec.T,phi6Vec)*np.log(2))-1)
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
	return np.clip((np.array(colorsys.hsv_to_rgb(angle/(2*np.pi)-.1,length,.7*np.ones(len(angle))))+1)/2,0,1)
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

def makeZColors(p):
	colorList=np.array([[.8,.8,.8],[.8,.8,.8],[.8,.8,.8],[.7,.8,.8],[.55,.6,.8],[.4,.55,.7],[.67,.5,.7],[.6,.3,.4],[.2,.38,.4]])
	zList=np.array(np.sum(p.getContacts(),axis=1),dtype=int).flatten()
	print(zList)
	return np.array(colorList[zList])

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
	cv = cp.delaunayVectors(p)[1].tocoo()
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

def packingZImage(directory,ax,fig,label=None):
	p = pcp.Packing(deviceNumber=2)
	loadPack(directory)
	colorMap=makeZColors(p)
	p.draw2DPacking(faceColor=colorMap.astype(float),edgeColor=[.5,.5,.5],alpha=1,axis=ax,lineWidth=0,figure=fig)
	p.draw2DNetwork(p.getContacts(),color=[.2,.2,.2],lineWidth=.5,axis=ax,figure=fig,alpha=1)
	ax.axis('off')
	fig.tight_layout(pad=0,w_pad=0,h_pad=0)
	if label != None:
		fig.text(.17,.1,label,color='black',bbox= dict(boxstyle='square',alpha=.7,facecolor=[.6,.6,.6],linewidth=0),fontsize=28)

if __name__ == '__main__':
	n=256
	directories=[f'{n}/finishedPackings/idealPack{n}-1',f'{n}/jumbledPackings/idealPack{n}-1/isostatic']
	labels=['A','B']#['C','D','A','B']
	for index in range(len(directories)):
		ax=plt.gca()
		fig=plt.gcf()
		packingZImage(p,f'../idealPackingLibrary/{directories[index]}')		figName=directories[index].split('/')[1]+'.'+directories[index].split('/')[2]
		plt.gcf().text(.17,.1,labels[index],color='black',bbox= dict(boxstyle='square',alpha=.7,facecolor=[.6,.6,.6],linewidth=0),fontsize=28)
		fig.savefig(f'../idealPackingLibrary/figures/{n}.{figName}-packingImage.pdf',pad_inches=0)
		fig.savefig(f'../idealPackingLibrary/figures/{n}.{figName}-packingImage.png',pad_inches=0)
		plt.clf()
		plt.close()
