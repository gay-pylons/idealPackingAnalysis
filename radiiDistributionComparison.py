#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:12:17 2023

@author: violalum
"""
import numpy as np
import pyCudaPacking as pcp
from modRatioVsN import moduliPerParticlePressure
from matplotlib import pyplot as plt
import matplotlib
import scipy
import plotColorList

markerList=plotColorList.markerList
posColor=plotColorList.posColor
posRadColor=plotColorList.posRadColor
posTriangColor=plotColorList.posTriangColor
posRadTriangColor=plotColorList.posRadTriangColor

def HistogramFromRadii(radii,nBins=10):
	maxRadius=np.max(radii)*1.01
	minRadius=np.min(radii)*.99
	binSize=(maxRadius-minRadius)/nBins
	histogram=np.zeros(nBins)
	truncRadii=((radii-minRadius)/binSize).astype(int)
	for binNum in truncRadii:
		histogram[binNum] += 1
	binCenters=np.arange(nBins)*binSize+binSize/2+minRadius
	return binCenters, histogram/(np.sum(histogram)*binSize)

def logNormalPlot(radii,numParticles,phi=.91,sigma=.2):
	x = np.linspace(min(radii), max(radii), 100)
	mu=np.log(phi/(np.pi*numParticles))/2-sigma**2
	pdf =(np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
	return x,pdf

def logHistDiff(radii,numParticles,centers,verticals,phi=.91,sigma=.2):
	mu=np.log(phi/(np.pi*numParticles))/2-sigma**2
	pdf =(np.exp(-(np.log(centers) - mu)**2 / (2 * sigma**2)) / (centers * sigma * np.sqrt(2 * np.pi)))
	return centers,(verticals-pdf)


poscpRadii=[]
posradcpRadii=[]
posRadii=[]
posradRadii=[]
phi=[]
n=4096
maxIndex=10
binno=100
filename='posMin'
filename2=f'idealPack{n}'
filename3='radMin'
for i in range(maxIndex):
	p = pcp.Packing()
	p.load(f'../idealPackingLibrary/{n}/finishedPackings/{filename}-{i}')
	poscpRadii.append(p.getRadii())
	phi.append(p.getPhi())
phi=np.array(phi,dtype=float)
for i in range(1,maxIndex):
	p = pcp.Packing()
	p.load(f'../idealPackingLibrary/{n}/finishedPackings/{filename2}-{i}')
	p.setPhi(np.mean(phi))
	posradcpRadii.append(p.getRadii())
for i in range(1,maxIndex):
	p = pcp.Packing()
	p.load(f'../idealPackingLibrary/{n}/posMin/{filename}-{i}')
	p.setPhi(np.mean(phi))
	posRadii.append(p.getRadii())
# =============================================================================
# for i in range(1,maxIndex):
# 	p = pcp.Packing()
# 	p.load(f'../idealPackingLibrary/{n}/radMin/{filename3}-{i}')
# 	p.setPhi(np.mean(phi))
# 	posradRadii.append(p.getRadii())
# =============================================================================
normFactor=(np.mean(phi)/(n*np.pi))**.5
poscpRadii=np.array(poscpRadii).flatten().astype(float)
posradcpRadii=np.array(posradcpRadii).flatten().astype(float)
posRadii=np.array(posRadii).flatten().astype(float)
#posradRadii=np.array(posradRadii).flatten().astype(float)
#disp=np.std(poscpRadii)/np.mean(poscpRadii)
sig=.2
polyX,polyY=logNormalPlot(poscpRadii,n, phi=np.mean(phi).astype(float),sigma=.2)
polyX2,polyY2=logNormalPlot(poscpRadii,n, phi=np.mean(phi).astype(float),sigma=.25)

plt.semilogy(polyX/np.mean(posRadii),polyY*np.mean(posRadii),color='black',alpha=.5)
plt.semilogy(polyX2/np.mean(posRadii),polyY2*np.mean(posRadii),color='black',alpha=.5)

hist,bin_edges=np.histogram(posradcpRadii/np.mean(posradcpRadii),density=True,bins=binno)
bin_centers= np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])
plt.semilogy(bin_centers,hist,'-x',alpha=.7,color=posRadTriangColor,label='posRadTriang',fillstyle='none')

# =============================================================================
# hist,bin_edges=np.histogram(poscpRadii/np.mean(poscpRadii),density=True,bins=binno)
# bin_centers= np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])
# plt.semilogy(bin_centers,hist,'-+',alpha=.7,color=posTriangColor,label='posTriang',fillstyle='none')
# 
# =============================================================================
# =============================================================================
# hist,bin_edges=np.histogram(posRadii/np.mean(posRadii),density=True,bins=binno)
# bin_centers= np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])
# plt.semilogy(bin_centers,hist,'-o',alpha=.7,color=posColor,label='pos',fillstyle='none')
# =============================================================================

# =============================================================================
# hist,bin_edges=np.histogram(posradRadii/np.mean(posradRadii),density=True,bins=binno)
# bin_centers= np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])
# plt.semilogy(bin_centers,hist,'-^',alpha=.7,color=posRadColor,label='posRad',fillstyle='none')
# 
# =============================================================================
plt.tick_params(axis='x',which='major',direction='inout',length=14,labelsize='x-large')
plt.tick_params(axis='x',which='minor',direction='inout',length=10)
plt.tick_params(axis='y',which='major',direction='in',length=10,labelsize='x-large')
plt.tick_params(axis='y',which='minor',direction='in',length=5)

plt.xlabel('$R/R_{rms}$',fontsize='xx-large')
plt.ylabel('$P(R/R_{rms})$',fontsize='xx-large')
plt.tight_layout()
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.15),
#          fancybox=True, shadow=True, ncol=10)
#plt.legend()
#plt.title(f'{maxIndex} systems of {n} particles, pos->CirclePack')
plt.xlim(0,2.5)
plt.yscale('log')
plt.savefig(f'../idealPackingLibrary/figures/{n}DistComp.pdf')
plt.savefig(f'../idealPackingLibrary/figures/{n}DistComp.png')
plt.show()
