#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:43:17 2023

@author: violalum
"""
import pyCudaPacking as pcp
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
def colourize(packing):
	radii=packing.getRadii()
	k=np.zeros_like([],dtype=float,shape=[len(radii),3])
	meanRadius=np.mean(radii).astype(float)
	for i in range(len(k)):
		k[i]=(radii[i]/meanRadius)*np.array([.1,.5,.4])%1
	return k

p=pcp.Packing()
p.load('radminShowPacking')
faceColors=colourize(p)
offContacts=sparse.load_npz('offcontacts.npz')
fullContacts=sparse.load_npz('fullcontacts.npz')
fullContacts
p.draw2DNetwork(fullContacts,color=[.5,.3,.75])
plt.axis('off')
plt.savefig('delaunayTriangulation.svg',bbox_inches='tight',pad_inches=0)
plt.clf()
p.draw2DPacking(faceColor=faceColors,edgeColor=None)
p.draw2DNetwork(offContacts,lineWidth=3,alpha=1,color=[.5,.3,.75])
plt.axis('off')
plt.savefig('radminPacking.svg',bbox_inches='tight',pad_inches=0)
plt.savefig('radminPacking.png',bbox_inches='tight',pad_inches=0,dpi=2000)
plt.clf()
p=pcp.Packing()
p.load('showPacking')
p.draw2DPacking(faceColor=colourize(p),edgeColor=None)
plt.axis('off')
plt.savefig('nonRadminPacking.svg',bbox_inches='tight',pad_inches=0)
plt.clf()
p.setRandomPositions()
p.draw2DPacking(faceColor=colourize(p),edgeColor=None)
plt.axis('off')
plt.savefig('randomPacking.svg',bbox_inches='tight',pad_inches=0)
plt.savefig('randomPacking.png',bbox_inches='tight',pad_inches=0,dpi=2000)