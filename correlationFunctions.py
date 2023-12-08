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



def makePhi6Vectors(p):
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
	phi6 /= np.max(np.abs(phi6))
	return np.array([phi6.real.astype(float),phi6.imag.astype(float)])

def correlationFunctions(packing):
	distance=p.getDistances()
	print(np.size(distance))
	phi6=makePhi6Vectors(packing).transpose()
	print(np.size(phi6))
	pos=p.getPositions()
	r=[]
	correlation=[]
	for i in range(packing.getNumParticles()):
		for j in range(i+1,packing.getNumParticles()):
			r.append(distance[i,j].astype(float))
			correlation.append(np.dot(phi6[i],phi6[j]))
	return np.array(r),np.array(correlation)

if __name__ == '__main__':
	nList=[128,256,512,1024,2048,4096]
	for n in nList:
		nbins=(10*np.log2(n)).astype(int)
		r=[]
		c=[]
		radii=[]
		for index in range(10):
			try:
				cind=np.loadtxt(f'../idealPackingLibrary/{n}/finishedPackings/idealPack{n}-{index}-correlations.dat')
				rind=np.loadtxt(f'../idealPackingLibrary/{n}/finishedPackings/idealPack{n}-{index}-distances.dat')
			except:
				p = pcp.Packing()
				p.load(f'{n}/finishedPackings/idealPack{n}-{index}') #switch back to 4k after finishing this test
				lv=np.loadtxt(f'{n}/finishedPackings/idealPack{n}-{index}/latticeVectors.dat')
				lv1Norm=lv[0]/np.sqrt(np.dot(lv[0],lv[0]))
				rotationMatrix= np.array([[lv1Norm[0],lv1Norm[1]],[-lv1Norm[1],lv1Norm[0]]])
				newVec=np.matmul(rotationMatrix.transpose(),lv)
				p.setLatticeVectors(newVec)
				p.setPositions(np.dot(p.getPositions(),rotationMatrix))
				p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
				p.minimizeFIRE('1e-20')
				rind, cind =correlationFunctions(p)
				np.savetxt(f'{n}/finishedPackings/idealPack{n}-{index}-correlations.dat',cind)
				np.savetxt(f'{n}/finishedPackings/idealPack{n}-{index}-distances.dat',rind)
			radii.append(np.loadtxt(f'{n}/finishedPackings/idealPack{n}-{index}/radii.dat'))
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
		plt.errorbar(rplots,cbins,yerr=cerror,marker='.')
	plt.xlabel('d/<r>')
	plt.ylabel('C(d/<r>)')
	plt.savefig(f'radMinAvgCorrelations.png')
	plt.savefig(f'radMinAvgCorrelations.pdf')
	plt.clf()
	for n in nList:
		nbins=(10*np.log2(n)).astype(int)
		r=[]
		c=[]
		radii=[]
		for index in range(10):
			try:
				cind=np.loadtxt(f'{n}/finishedPackings/posMin-{index}-correlations.dat')
				rind=np.loadtxt(f'{n}/finishedPackings/posMin-{index}-distances.dat')
			except:
				p = pcp.Packing()
				p.load(f'{n}/finishedPackings/posMin-{index}') #switch back to 4k after finishing this test
				lv=np.loadtxt(f'{n}/finishedPackings/posMin-{index}/latticeVectors.dat')
				lv1Norm=lv[0]/np.sqrt(np.dot(lv[0],lv[0]))
				rotationMatrix= np.array([[lv1Norm[0],lv1Norm[1]],[-lv1Norm[1],lv1Norm[0]]])
				newVec=np.matmul(rotationMatrix.transpose(),lv)
				p.setLatticeVectors(newVec)
				p.setPositions(np.dot(p.getPositions(),rotationMatrix))
				p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
				p.minimizeFIRE('1e-20')
				rind, cind =correlationFunctions(p)
				np.savetxt(f'{n}/finishedPackings/posMin-{index}-correlations.dat',cind)
				np.savetxt(f'{n}/finishedPackings/posMin-{index}-distances.dat',rind)
			radii.append(np.loadtxt(f'{n}/finishedPackings/posMin-{index}/radii.dat'))
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
		plt.errorbar(rplots,cbins,yerr=cerror,marker='.')
	plt.xlabel('d/<r>')
	plt.ylabel('C(d/<r>)')
	plt.savefig(f'posMinAvgCorrelations.png')
	plt.savefig(f'posMinAvgCorrelations.pdf')
