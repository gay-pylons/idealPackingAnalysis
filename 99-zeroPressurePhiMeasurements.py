#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:29:14 2024

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
import cpRandomDelaunayTriangulation as cp
import plotColorList as pcl

def getPhi(path):
# =============================================================================
# 	p = pcp.Packing(deviceNumber=1)
# 	p.load(posTriangPath)
# 	try:
# 		lv=np.loadtxt(f'{path}/latticeVectors.dat')
# 		p.setLatticeVectors(lv)
# 		p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
# 	except:
# 		print('noLV')
# 	return p.getPhi()
# =============================================================================
	radii=np.loadtxt(f'{path}/radii.dat')
	try:
		lv=np.loadtxt(f'{path}/latticeVectors.dat')
		volume=np.det(lv)
	except:
		volume=1
	return np.pi*np.dot(radii,radii)/volume

nList=np.array([128,256,512,1024,2048,4096])
posTriangJamm=[]
posRadTriangJamm=[]
posIso=[]
posRadIso=[]
for n in nList:
	posTriang=[]
	posRadTriang=[]
	for index in range(10):
		posTriangPath=f'../idealPackingLibrary/{n}/finishedPackings/posMin-{index}'
		phi=getPhi(posTriangPath)
		if(phi<.9):
			print(posTriangPath)
		posTriang.append(phi)
	for index in range(10):
		posRadTriangPath=f'../idealPackingLibrary/{n}/finishedPackings/idealPack{n}-{index}'
		phi=getPhi(posRadTriangPath)
		if(phi<.9):
			print(posRadTriangPath)
		posRadTriang.append(phi)
	posTriang=np.array(posTriang,dtype=float)
	posRadTriang=np.array(posRadTriang,dtype=float)
	posTriangJamm.append(np.mean(posTriang))
	posRadTriangJamm.append(np.mean(np.array(posRadTriang)))
posTriangJamm=np.array(posTriangJamm)
posRadTriangJamm=np.array(posRadTriangJamm)
plt.semilogx(nList,posRadTriangJamm,'--o',label='pos rad min traingulated packing')
plt.semilogx(nList,posTriangJamm,'--x',label='pos-min triangulated packing')
print(np.mean(posTriangJamm))
print(np.mean(posRadTriangJamm))
for n in nList:
	pos=[]
	posRad=[]
	for index in range(10):
		posPath=f'../idealPackingLibrary/{n}/posMin/posMin-{index}/isostatic'
		phi=getPhi(posPath)
		pos.append(phi)
	for index in range(10):
		posRadPath=f'../idealPackingLibrary/{n}/radMin/radMin-{index}/isostatic'
		phi=getPhi(posRadPath)
		posRad.append(phi)
	pos=np.array(pos,dtype=float)
	posRad=np.array(posRad,dtype=float)
	posIso.append(np.mean(pos))
	posRadIso.append(np.mean(np.array(posRad)))
posIso=np.array(posIso)
posRadIso=np.array(posRadIso)
plt.semilogx(nList,posRadIso,'--1',label='pos rad min isostatic packing')
plt.semilogx(nList,posIso,'--^',label='pos-min isostatic packing')
plt.xlabel('$N$')
plt.ylabel('$\phi$')
plt.legend()
#plt.show()
plt.savefig('../idealPackingLibrary/figures/jammingDensity.png')