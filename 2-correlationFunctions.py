#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:12:29 2023
tool to get correlation functions
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

def makePhi6(p):
	phi = p.getPhi()
	p.setPhi(np.quad(.1)+phi)
	cv = cp.delaunayVectors(p)
	p.setPhi(phi)
	contactVecs = cv.data.reshape(len(cv.data)//2,2).astype(float)
	print(np.size(contactVecs))
	thetas = np.arctan2(contactVecs[:,1], contactVecs[:,0])
	#NOTE: This will fail if there are rattlers!
	iVals = cv.row.reshape(len(cv.row)//2,2)[:,0]
	numContacts = np.bincount(iVals, minlength = p.getNumParticles())
	phi6 = np.zeros(len(numContacts), dtype=complex)
	np.add.at(phi6, iVals, np.exp(6j*thetas))
#	phi6 /= np.max(np.abs(phi6))
	return phi6

def correlationFunctions(packing):
	distance=p.getDistances()
	print(np.size(distance))
	phi6=makePhi6(packing)
	print(np.size(phi6))
	r=[]
	correlation=[]
	for i in range(packing.getNumParticles()):
		for j in range(i+1,packing.getNumParticles()):
			r.append(distance[i,j].astype(float))
			correlation.append((phi6[i]*phi6[j].conjugate()).re)
	return np.array(r),np.array(correlation)

def loadCorrelations(corrPath):
	try:
		cind=np.loadtxt(f'{corrPath}-correlations.dat')
		rind=np.loadtxt(f'{corrPath}-distances.dat')
	except:
		p = pcp.Packing(deviceNumber=1)
		p.load(f'{corrPath}') #switch back to 4k after finishing this test
		lv=np.loadtxt(f'{corrPath}/latticeVectors.dat')
		lv1Norm=lv[0]/np.sqrt(np.dot(lv[0],lv[0]))
		rotationMatrix= np.array([[lv1Norm[0],lv1Norm[1]],[-lv1Norm[1],lv1Norm[0]]])
		newVec=np.matmul(rotationMatrix.transpose(),lv)
		p.setLatticeVectors(newVec)
		p.setPositions(np.dot(p.getPositions(),rotationMatrix))
		p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
		p.minimizeFIRE('1e-20')
		rind, cind =correlationFunctions(p)
		np.savetxt(f'{corrPath}-correlations.dat',cind)
		np.savetxt(f'{corrPath}-distances.dat',rind)
	return rind, cind
			
if __name__ == '__main__':
	nList=[128,256,512,1024,2048,4096]#,8192]
	cIt=0
	for n in nList:
		nbins=(np.log2(n)).astype(int)
		r=[]
		c=[]
		radii=[]
		for index in range(10):
			corrPath=f'../idealPackingLibrary/{n}/finishedPackings/idealPack{n}-{index}'
			rind, cind=loadCorrelations(corrPath)
		radii.append(np.loadtxt(f'../idealPackingLibrary/{n}/finishedPackings/idealPack{n}-{index}/radii.dat'))
		r.append(rind)
		c.append(cind)
		r=np.array(r).flatten().astype(float)
		r=r/np.mean(np.array(radii).flatten().astype(float))
		c=np.array(c).flatten().astype(float)
		rbins=np.linspace(np.min(r)*.999,np.max(r)*1.001,nbins+1)
		print(len(rbins))
		cbins=[]
		cerror=[]
		rplots=[]
		for i in range(nbins):
			print(i)
			conditions=np.logical_and(r >= rbins[i],r < rbins[i+1])
			sublist=c[conditions]
			print(len(sublist))
			cbins.append(np.mean(sublist))
			cerror.append(np.std(sublist)/np.sqrt(sublist.shape))
			rplots.append(.5*rbins[i]+.5*rbins[i+1])
		cerror=np.array(cerror)
		cbins=np.array(cbins)
		print(len(cerror))
		print(len(cbins))
		rplots=np.array(rplots)
		plt.plot(rplots,cbins,pcl.markerList[cIt],color=pcl.posRadTriangColor)
		cIt+=1
	cIt=0
	for n in nList:
		nbins=(10*np.log2(n)).astype(int)
		r=[]
		c=[]
		radii=[]
		for index in range(10):
			corrPath=f'../idealPackingLibrary/{n}/finishedPackings/idealPack{n}-{index}'
			rind, cind=loadCorrelations(corrPath)
			radii.append(np.loadtxt(f'../idealPackingLibrary/{n}/finishedPackings/posMin-{index}/radii.dat'))
			r.append(rind)
			c.append(cind)
		r=np.array(r).flatten().astype(float)
		r=r/np.mean(np.array(radii).flatten().astype(float))
		c=np.array(c).flatten().astype(float)
		rbins=np.linspace(np.min(r)*.999,np.max(r)*1.001,nbins+1)
		print(len(rbins))
		cbins=[]
		cerror=[]
		rplots=[]
		for i in range(nbins):
			print(i)
			conditions=np.logical_and(r >= rbins[i],r < rbins[i+1])
		#		print(np.sum(conditions))
			sublist=c[conditions]
			print(len(sublist))
			cbins.append(np.mean(sublist))
			cerror.append(np.std(sublist)/np.sqrt(sublist.shape))
			rplots.append(.5*rbins[i]+.5*rbins[i+1])
		cerror=np.array(cerror)
		cbins=np.array(cbins)
		print(len(cerror))
		print(len(cbins))
		rplots=np.array(rplots)
		plt.plot(rplots,cbins,pcl.markerList[cIt],color=pcl.posTriangColor)
		cIt+=1
	plt.xlabel('d/<r>')
	plt.ylabel('C(d/<r>)')
	plt.savefig(f'../idealPackingLibrary/figures/posMinAvgCorrelations.png')
	plt.savefig(f'../idealPackingLibrary/figures/posMinAvgCorrelations.pdf')
