#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:43:17 2023

@author: violalum
"""
import pyCudaPacking as pcp
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
def colourize(packing):
	radii=packing.getRadii()
	k=np.zeros_like([],dtype=float,shape=[len(radii),3])
	meanRadius=np.mean(radii).astype(float)
	for i in range(len(k)):
		k[i]=(radii[i]/meanRadius)*np.array([.15,.6,.5])%1
	return k

p=pcp.Packing()
p.load('radminShowPacking')
faceColors=colourize(p)
offContacts=np.load('offcontacts.npy',allow_pickle=True)
fullContacts=np.load('fullcontacts.npy',allow_pickle=True)
fullContacts
p.draw2DNetwork(fullContacts,color=[.5,.3,.75])
plt.axis('off')
plt.savefig('delaunayTriangulation.png',bbox_inches='tight',pad_inches=0,dpi=2000)
plt.clf()
p.draw2DPacking(faceColor=faceColors,edgeColor=None)
p.draw2DNetwork(offContacts,linewidth=5)
plt.axis('off')
plt.savefig('radminPacking.png',bbox_inches='tight',pad_inches=0,dpi=2000)
p=pcp.Packing()
p.load('showPacking')
p.draw2DPacking(faceColor=colourize(p),edgeColor=None)
plt.axis('off')
plt.savefig('nonRadminPacking.png',bbox_inches='tight',pad_inches=0,dpi=2000)