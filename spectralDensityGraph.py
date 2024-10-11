#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:18:11 2023

@author: violalum
"""
import os
import matplotlib
matplotlib.use('Agg')
import imp
import numpy as np
from scipy.stats import gmean
import npquad
import matplotlib.pyplot as plt
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import idealGlass
import plotColorList
from scipy.stats import binned_statistic
from scipy.stats import gmean

def homebrew2DSpecDense(packing):
	weights=packing.getRadii()**2 
	positions=packing.getPositions()
	lvs=packing.getLatticeVectors()
	k1=lvs[0]/np.dot(lvs[0],lvs[0])
	k2=lvs[1]/np.dot(lvs[1],lvs[1])
	def smoothSofK(k):
		rho=0
		for i in range(packing.getNumParticles()):
			rho += weights[i]*np.exp(1j*np.dot(k,positions[i]))
		return np.sqrt(rho.real**2+rho.imag**2)
	#td: create lattice wector grid
	#multiply lattice wector grid b

def getSpectralDensity(path,saveOverride=False):
	try:
		if saveOverride:
			np.loadtxt(path)
		knn=1/(np.mean(np.loadtxt(f'{path}/radii.dat')))
		k=np.loadtxt(f'{path}/kList.dat')
		chi=np.loadtxt(f'{path}/specList.dat')
	except:
		p = pcp.Packing(nDim=2,deviceNumber=2)
		p.load(path)
		try:
			lvs=pcp.load2DArray(f'{path}/latticeVectors.dat',np.quad)
		except:
			lvs=np.array([[1,0],[0,1]],dtype=np.quad)
#		print(lvs)
		p.setLatticeVectors(lvs)
		p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
		knn=int(np.trunc(2*np.pi/(np.mean(p.getRadii().astype(float)))))
#		knn=np.round(np.sqrt(p.getNumParticles))
		print(knn)
		chi,k=p.getSpectralDensity(knn)
		np.savetxt(f'{path}/kList.dat',k.astype(float))
		np.savetxt(f'{path}/specList.dat',chi.astype(float))
		knn=1/(np.mean(p.getRadii().astype(float)))
	return k.astype(float)/knn, chi.astype(float)*knn**2

markerList=plotColorList.markerList
posColor=plotColorList.posColor
posRadColor=plotColorList.posRadColor
posTriangColor=plotColorList.posTriangColor
posRadTriangColor=plotColorList.posRadTriangColor

mSize=6

overList=[False,False,False,False,False,False,False,False,False]
nArray=[64,128,256,512,1024,2048,4096,8192]

fig = plt.figure(figsize=(6.4,6))

cIt=0
for n in nArray:
	try:
		print(n)
		kList=[]
		specList=[]
		directory=f'../idealPackingLibrary/{n}/jumbledPackings/idealPack{n}'
		packingDirs= [f'{directory}-{i}/isostatic' for i in range(10)]
		for direct in packingDirs:
			k, chi= getSpectralDensity(direct,saveOverride=False)
			kList.append(k)
			specList.append(chi)
		kList=np.concatenate(kList).astype(float)
		specList=np.concatenate(specList).astype(float)
		bins = np.geomspace(np.min(k),np.max(k), num=101)
		chi,k,no=binned_statistic(kList,specList,statistic=gmean,bins=bins)
		print(no)
		kavg,k,no=binned_statistic(kList,kList,statistic=gmean,bins=bins)
		plt.loglog(kavg,chi,markerList[cIt],markersize=mSize,color=posColor,alpha=.5,fillstyle='none')
		cIt+=1
	except:
			print('no data')


# =============================================================================
# cIt=0
# 
# n=8192 #function of indices and path?
# kList=[]
# specList=[]
# #directory=f'../idealPackingLibrary/{n}/finishedPackings/1x8LatticeVectors'
# #packingDirs= [f'{directory}-{i}' for i in range(30)]
# directory=f'../idealPackingLibrary/{n}/finishedSquare'
# packingDirs=[name.path for name in os.scandir(directory) if name.is_dir()]
# for direct in packingDirs:
# 	k, chi= getSpectralDensity(direct,saveOverride=False)
# 	kList.append(k)
# 	specList.append(chi)
# kList=np.concatenate(kList).astype(float)
# specList=np.concatenate(specList).astype(float)
# bins = np.geomspace(np.min(k),np.max(k), num=101)
# chi,k,no=binned_statistic(kList,specList,statistic=gmean,bins=bins)
# print(no)
# kavg,k,no=binned_statistic(kList,kList,statistic=gmean,bins=bins)
# plt.loglog(kavg,chi,markerList[cIt],markersize=mSize,color=[.3,.3,.3],alpha=.5,fillstyle='none')
# cIt+=1
# 
# =============================================================================
cIt=0
for n in nArray:
	print(n)
	kList=[]
	specList=[]
#	directory=f'../idealPackingLibrary/{n}/finishedPackings/idealPack{n}'
#	packingDirs= [f'{directory}-{i}' for i in range(10)]
	directory=f'../idealPackingLibrary/{n}/finishedSquare'
	packingDirs=[name.path for name in os.scandir(directory) if name.is_dir()]
	for direct in packingDirs:
		k, chi= getSpectralDensity(direct,saveOverride=True)
		kList.append(k)
		specList.append(chi)
	kList=np.concatenate(kList).astype(float)
	specList=np.concatenate(specList).astype(float)
	bins = np.geomspace(np.min(k),np.max(k), num=101)
	chi,k,no=binned_statistic(kList,specList,statistic=gmean,bins=bins)
	print(no)
	kavg,k,no=binned_statistic(kList,kList,statistic=gmean,bins=bins)
	plt.loglog(kavg,chi,markerList[cIt],markersize=mSize,color=posRadTriangColor,alpha=.5,fillstyle='none')
	cIt+=1
eyex=np.array([.065,.7])
eyey=eyex**(1/2)*.5e-4
plt.plot(eyex,eyey,'--',color='black')

eyex=np.array([.065,.7])
eyey=eyex**(2/3)*.5e-4
plt.plot(eyex,eyey,'-.',color='grey')
#plt.text(.16,1.4e-5,r'$\propto k^\frac{1}{2}$',fontsize='x-large',color='black')
plt.tick_params(axis='x',which='major',direction='inout',length=14,labelsize='x-large')
plt.tick_params(axis='x',which='minor',direction='inout',length=10)
plt.tick_params(axis='y',which='major',direction='in',length=10,labelsize='x-large')
plt.tick_params(axis='y',which='minor',direction='in',length=5)
plt.ylim([1e-5,6e-2])
plt.xlabel(r'$\left<r\right> k$',fontsize='xx-large')
plt.ylabel(r'$\tilde\chi\left(k\right)/\left<r\right>^2$',fontsize='xx-large')
plt.tight_layout()
plt.savefig('../idealPackingLibrary/figures/spectralDensity.pdf')
plt.savefig('../idealPackingLibrary/figures/spectralDensity.png')
print('fin')
