#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 18:35:38 2024

@author: violalum
"""
import sys
import numpy as np
import npquad
import imp
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import rigidpy as rp
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse
#mport cpRandomDelaunayTriangulation as cp
import plotColorList as pcl

def makeMonoPacking(directory,N,phi):
	try:
		p = pcp.Packing()
		p.load(directory)
	except:
		p=pcp.Packing(numParticles=N,nDim=2)
		p.setMonoRadii()
		p.setRadii(p.getRadii())
		p.setRandomPositions()
		p.setPhi(phi)
		p.minimizeFIRE('1e-20')
		print(p.getPressure())
		while(p.getPressure()==0):
			p.setRandomPositions()
			p.minimizeFIRE('1e-20')
		p.save(directory,overwrite=True)
if __name__ == '__main__':
	nList=[64,128,256,512,1024,2048,4096]#,8192]
	for n in nList:
		for i in range(30):
			makeMonoPacking(f'../idealPackingLibrary/{n}/monoDisperse/monoDisperse-{i}',n,np.quad(.9))
