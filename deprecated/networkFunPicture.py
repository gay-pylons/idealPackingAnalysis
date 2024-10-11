#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 22:30:57 2023

@author: violalum
"""
import pyCudaPacking as pcp
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import matplotlib.colors as mcolors
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

def colourize(packing):
	radii=packing.getRadii()
	k=np.zeros_like([],dtype=float,shape=[len(radii),3])
	meanRadius=np.mean(radii).astype(float)
	for i in range(len(k)):
		k[i]=(radii[i]/meanRadius)*np.array([.1,.5,.4])%1
#		k[i]=(radii[i]/meanRadius)*np.array([.15,.6,.5])%1
	return k

p=pcp.Packing(potentialPower=2,nDim=2)
#p.load('showPacking')
#fullContacts=sparse.load_npz('fullcontacts.npz')
p.setLogNormalRadii(polyDispersity='.2')
p.setRandomPositions()
p.set2DHexCrystal((30,17),rescaleRadii=True)
#lv=np.array(np.loadtxt('1024/finishedPackings/radminShowPacking/latticeVectors.dat'),dtype=np.quad)
#p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
#p.setLatticeVectors(lv)
#print(p.getPhi())
#p.setPhi(p.getPhi()+np.quad('1e-6'))
#p.minimizeFIRE('1e-20')
#conList=p.getContacts()
#print(np.size(conList))
#p.setPhi('.915')
p.setPhi('.911')
p.minimizeFIRE('1e-20')
print(moduliPerParticle(p,'1024Crystal'))
#p.setGeometryType(pcp.enums.geometryEnum.normal)
#p.setPositions(np.dot(p.getPositions(),lv.astype(np.quad)))
#colMap=colourize(p)
#p.draw2DNetwork(p.getContacts(),color=[.6,.4,.8])
#p.draw2DNetwork(p.getContacts()!=fullContacts,color='fuchsia')
#p.draw2DPacking(edgeColor=None,faceColor=colMap)
#plt.axis('off')
#plt.tight_layout()
#plt.savefig('256Show.svg',bbox_inches='tight',pad_inches=0)