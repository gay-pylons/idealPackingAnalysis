#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:12:29 2023
tool to get correlation functions
@author: violalum
"""
import sys
from scipy.stats import gmean
import numpy as np
import npquad
import imp
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import rigidpy as rp
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse
import cpPoissonPoints as cp
import plotColorList as pcl

def delaunayVectors(packing):
	longVectors=packing.getContactVectors(gap=np.quad(1)).astype(float).tocsr()
	delaunay=packing.delaunayNeighbors()
	delVectors=scipy.sparse.csr_matrix((packing.getNumParticles(),2*packing.getNumParticles()), dtype=float)
	for i in range(packing.getNumParticles()):
		for j in delaunay[i].indices:
			delVectors[i,2*j]=longVectors[i,2*j]
			delVectors[i,2*j+1]=longVectors[i,2*j+1]
	print(np.size(delVectors))
	return delVectors.astype(float).tocoo()

def makePhi6(p):
	phi = p.getPhi()
	cv = delaunayVectors(p)
	p.setPhi(phi)
	contactVecs = cv.data.reshape(len(cv.data)//2,2).astype(float)
#cr	print(np.size(contactVecsr/\sqrt{N}',size='xx-large'))
	thetas = np.arctan2(contactVecs[:,1], contactVecs[:,0])
	#NOTE: This will fail if there are rattlers!
	iVals = cv.row.reshape(len(cv.row)//2,2)[:,0]
	numContacts = np.bincount(iVals, minlength = p.getNumParticles())
#	print(numContacts)
	phi6 = np.zeros(len(numContacts), dtype=complex)
	np.add.at(phi6, iVals, np.exp(6j*thetas))
#	print(np.max(phi6.real))
	return phi6/numContacts

def correlationFunctions(packing):
	distance=packing.getDistances().flatten()
#	print(distance)
	phi6=makePhi6(packing)
	correlation= (np.outer(phi6.real,phi6.real)+np.outer(phi6.imag,phi6.imag)).flatten()
	return np.array(distance,dtype=float),np.array(correlation)

def dirCorrelationFunctions(packing):
	distance=packing.getDistances()

	phi6=makePhi6(packing)
	r=[]
	correlation=[]
	for i in range(packing.getNumParticles()):
		for j in range(i+1,packing.getNumParticles()):
			r.append(distance[i,j].astype(float))
			correlation.append((phi6[i]*phi6[j].conjugate()).real/np.abs(phi6[i]*phi6[j]))
	print('dirCorrelated')
	return np.array(r),np.array(correlation)

def loadCorrelations(corrPath):
	try:
		cind=np.loadtxt(f'{corrPath}-correlations.dat')
		rind=np.loadtxt(f'{corrPath}-distances.dat')
	except:
		p = pcp.Packing(deviceNumber=0)
		p.load(f'{corrPath}') #switch back to 4k after finishing this test
		try:
			lv=np.loadtxt(f'{corrPath}/latticeVectors.dat')
			lv1Norm=lv[0]/np.sqrt(np.dot(lv[0],lv[0]))
			rotationMatrix= np.array([[lv1Norm[0],lv1Norm[1]],[-lv1Norm[1],lv1Norm[0]]])
			newVec=np.matmul(rotationMatrix.transpose(),lv)
			p.setLatticeVectors(newVec)
			p.setPositions(np.dot(p.getPositions(),rotationMatrix))
			p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
		except:
			print('noLV')
		p.minimizeFIRE('1e-20')
		rind, cind =correlationFunctions(p)
		np.savetxt(f'{corrPath}-correlations.dat',cind)
		np.savetxt(f'{corrPath}-distances.dat',rind)
#	print(rind)
	return rind, cind

def loadDirCorrelations(corrPath):
	try:
		cind=np.loadtxt(f'{corrPath}-dirCorrelations.dat')
		rind=np.loadtxt(f'{corrPath}-distances.dat')
	except:
		p = pcp.Packing(deviceNumber=1)
		p.load(f'{corrPath}') #switch back to 4k after finishing this test
		try:
			lv=np.loadtxt(f'{corrPath}/latticeVectors.dat')
			lv1Norm=lv[0]/np.sqrt(np.dot(lv[0],lv[0]))
			rotationMatrix= np.array([[lv1Norm[0],lv1Norm[1]],[-lv1Norm[1],lv1Norm[0]]])
			newVec=np.matmul(rotationMatrix.transpose(),lv)
			p.setLatticeVectors(newVec)
			p.setPositions(np.dot(p.getPositions(),rotationMatrix))
			p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
		except:
			print('noLV')
		p.minimizeFIRE('1e-20')
		rind, cind =dirCorrelationFunctions(p)
		np.savetxt(f'{corrPath}-dirCorrelations.dat',cind)
		np.savetxt(f'{corrPath}-distances.dat',rind)

	return rind, cind

def binCorrelations(path,maxIndex=10,nbins=100,directionOnly=False,subDir=None):
	r=[]
	c=[]
#	radii=[]
	try:
		if subDir != None:
			cerror=np.loadtxt(f'{path}.{subDir}.dirEq{directionOnly}x{maxIndex}x{nbins}bins.cerror.rNorm.dat')
			cbins=np.loadtxt(f'{path}.{subDir}.dirEq{directionOnly}x{maxIndex}x{nbins}bins.cbins.rNorm.dat')
			rplots=np.loadtxt(f'{path}.{subDir}.dirEq{directionOnly}x{maxIndex}x{nbins}bins.rplots.rNorm.dat')
		else:
			cerror=np.loadtxt(f'{path}.dirEq{directionOnly}x{maxIndex}x{nbins}bins.cerror.rNorm.dat')
			cbins=np.loadtxt(f'{path}.dirEq{directionOnly}x{maxIndex}x{nbins}bins.cbins.rNorm.dat')
			rplots=np.loadtxt(f'{path}.dirEq{directionOnly}x{maxIndex}x{nbins}bins.rplots.rNorm.dat')
	except:
		for index in range(maxIndex):
			if subDir != None:
				corrPath=f'{path}-{index}/{subDir}' #entirely so I can use isostatic for subdir
			else:
				corrPath=f'{path}-{index}'
			if(directionOnly):
				rind, cind=loadDirCorrelations(corrPath)
			else:
				rind, cind=loadCorrelations(corrPath)
#			radii.append(np.loadtxt(f'{corrPath}/radii.dat'))
			r.append(rind/np.mean(np.loadtxt(f'{corrPath}/radii.dat')))
			c.append(cind)
		r=np.array(r).flatten().astype(float)
		c=np.array(c).flatten().astype(float)
		rbins=np.linspace(np.min(r)*.999,np.max(r)*1.001,nbins+1)
		cbins=[]
		cerror=[]
		rplots=[]
		for i in range(nbins):
			conditions=np.logical_and(r >= rbins[i],r < rbins[i+1])
			sublist=c[conditions]
			cbins.append(np.mean(sublist))
			cerror.append(np.std(sublist)/np.sqrt(len(sublist)))
			rplots.append(.5*rbins[i]+.5*rbins[i+1])
		cerror=np.array(cerror)
		cbins=np.array(cbins)
		rplots=np.array(rplots)
		if subDir != None:
			np.savetxt(f'{path}.{subDir}.dirEq{directionOnly}x{maxIndex}x{nbins}bins.cerror.rNorm.dat',cerror)
			np.savetxt(f'{path}.{subDir}.dirEq{directionOnly}x{maxIndex}x{nbins}bins.cbins.rNorm.dat',cbins)
			np.savetxt(f'{path}.{subDir}.dirEq{directionOnly}x{maxIndex}x{nbins}bins.rplots.rNorm.dat',rplots)
		else:
			np.savetxt(f'{path}.dirEq{directionOnly}x{maxIndex}x{nbins}bins.cerror.rNorm.dat',cerror)
			np.savetxt(f'{path}.dirEq{directionOnly}x{maxIndex}x{nbins}bins.cbins.rNorm.dat',cbins)
			np.savetxt(f'{path}.dirEq{directionOnly}x{maxIndex}x{nbins}bins.rplots.rNorm.dat',rplots)
	return cerror,cbins,rplots

def boxcarAvg(path,binWidth,maxIndex=10,nsteps=100):
	r=[]
	c=[]
	radii=[]
	for index in range(maxIndex):
		corrPath=f'{path}-{index}'
		if(directionOnly):
			rind, cind=loadDirCorrelations(corrPath)
		else:
			rind, cind=loadCorrelations(corrPath)
		radii.append(np.loadtxt(f'{corrPath}/radii.dat'))
		r.append(rind)
		c.append(cind)
	r=np.array(r).flatten().astype(float)
#		r=r/np.mean(np.array(radii).flatten().astype(float))
	c=np.array(c).flatten().astype(float)
	rbins=np.linspace(np.min(r)*.999,np.max(r)*1.001,nbins+1)
	cbins=[]
	cerror=[]
	rplots=[]
	for i in range(nbins):
		conditions=np.logical_and(r >= rbins[i],r < rbins[i+1])
		sublist=c[conditions]
		cbins.append(np.mean(sublist))
		cerror.append(np.std(sublist)/np.sqrt(len(sublist)))
		rplots.append(.5*rbins[i]+.5*rbins[i+1])
	cerror=np.array(cerror)
	cbins=np.array(cbins)
	rplots=np.array(rplots)
	np.savetxt(f'{path}.dirEq{directionOnly}x{maxIndex}x{nbins}bins.cerror.dat',cerror)
	np.savetxt(f'{path}.dirEq{directionOnly}x{maxIndex}x{nbins}bins.cbins.dat',cbins)
	np.savetxt(f'{path}.dirEq{directionOnly}x{maxIndex}x{nbins}bins.rplots.dat',rplots)
	return cerror,cbins,rplots

if __name__ == '__main__':
	cIt=5
	nList= [2048,4096]#[64,128,256,512,1024,2048,4096]
	nBins= [20, 25, 30, 40, 50, 50,100]
	plt.figure(figsize=(4,3.5))
# =============================================================================
# 	for n in nList:
# 		print(n)
# 		nbins=2*(np.log2(n)).astype(int)
# 		path=f'../idealPackingLibrary/{n}/finishedPackings/posMin'
# 		cerror, cbins, rplots =binCorrelations(path,nbins=nbins)
# 		plt.errorbar(rplots,np.abs(cbins),yerr=cerror,marker=pcl.markerList[cIt],color=pcl.posTriangColor,capsize=4,linewidth=0,alpha=.7,fill=None)
# 		cIt+=1
# =============================================================================
	for n in nList:
		print(n)
		nbins=nBins[cIt]
		path=f'../idealPackingLibrary/{n}/jumbledPackings/idealPack{n}'
		cerror, cbins, rplots =binCorrelations(path,nbins=nbins,subDir='isostatic')
		plt.errorbar(rplots,np.abs(cbins),yerr=cerror,fmt=pcl.markerList[cIt],marker=pcl.markerList[cIt],color=pcl.posColor,capsize=4,linewidth=1,alpha=.7,fillstyle='none')
		path=f'../idealPackingLibrary/{n}/finishedPackings/idealPack{n}'
		cerror, cbins, rplots =binCorrelations(path,nbins=nbins)
		plt.errorbar(rplots,np.abs(cbins),yerr=cerror,fmt=pcl.markerList[cIt],color=pcl.posRadTriangColor,capsize=4,linewidth=1,alpha=.7,fillstyle='none')
		cIt+=1

# =============================================================================
# 	cIt=0
# 	for n in nList:
# 		print(n)
# 		nbins=(np.log2(n)).astype(int)
# 		path=f'../idealPackingLibrary/{n}/radMin/radMin'
# 		cerror, cbins, rplots =binCorrelations(path,nbins=nbins)
# 		plt.errorbar(rplots,np.abs(cbins),yerr=cerror,marker=pcl.markerList[cIt],color=pcl.posRadTriangColor,capsize=4,linewidth=0,alpha=.7,fill=None)
# =============================================================================
	plt.tick_params(axis='x',which='major',direction='inout',length=14,labelsize='xx-large')
	plt.tick_params(axis='x',which='minor',direction='inout',length=10)
	plt.tick_params(axis='y',which='major',direction='in',length=10,labelsize='xx-large')
	plt.tick_params(axis='y',which='minor',direction='in',length=5)
	plt.xlim([0,13])
	plt.xlabel(r'$r_{ij}/ \left<r\right>$',size=22)
	plt.ylabel(r'$|C_6\left(r_{ij}/ \left<r\right>\right)|$',size=22)
	plt.yscale('log')
	plt.ylim([1e-4,1e-0])
#	plt.xscale('log')
	plt.tight_layout()
	plt.gcf().text(.1,.1,'C',color='black',bbox= dict(boxstyle='square',alpha=1,facecolor=[.9,.9,.9]),fontsize=26)
	plt.savefig(f'../idealPackingLibrary/figures/AvgCorrelations.png')
	plt.savefig(f'../idealPackingLibrary/figures/AvgCorrelations.pdf')

