#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:12:17 2023

@author: violalum
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pyCudaPacking as pcp
from modRatioVsN import moduliPerParticlePressure
from matplotlib import pyplot as plt
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
binno=60
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
for i in range(1,maxIndex):
	p = pcp.Packing()
	p.load(f'../idealPackingLibrary/{n}/seedPackings(radMin)/{filename2}-{i}')
	p.setPhi(np.mean(phi))
	posradRadii.append(p.getRadii())
normFactor=(np.mean(phi)/(n*np.pi))**.5
poscpRadii=np.array(poscpRadii).flatten().astype(float)
posradcpRadii=np.array(posradcpRadii).flatten().astype(float)
posRadii=np.array(posRadii).flatten().astype(float)
posradRadii=np.array(posradRadii).flatten().astype(float)
disp=np.std(posradcpRadii)/np.mean(posradcpRadii)
plt.figure(figsize=(8,4))
print(disp)
sig=.2
polyX,polyY=logNormalPlot(posradcpRadii,n, phi=np.mean(phi).astype(float),sigma=.2)
polyX2,polyY2=logNormalPlot(posradcpRadii,n, phi=np.mean(phi).astype(float),sigma=disp)

plt.semilogy(polyX/np.mean(posradcpRadii),polyY*np.mean(posradcpRadii),'--',color=[.3,.3,.3],alpha=1)
plt.semilogy(polyX2/np.mean(posradcpRadii),polyY2*np.mean(posradcpRadii),color='black',alpha=1)

hist,bin_edges=np.histogram(posradRadii/np.mean(posradRadii),density=True,bins=binno)
bin_centers= np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])
plt.semilogy(bin_centers,hist,'-x',alpha=.7,color=posRadColor,label='posRad',fillstyle='none')


hist,bin_edges=np.histogram(posradcpRadii/np.mean(posradcpRadii),density=True,bins=binno)
bin_centers= np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])
plt.semilogy(bin_centers,hist,'-^',alpha=.7,color=posRadTriangColor,label='posRadTriang',fillstyle='none')

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


plt.tick_params(axis='x',which='major',direction='inout',length=14,labelsize='x-large')
plt.tick_params(axis='x',which='minor',direction='inout',length=10)
plt.tick_params(axis='y',which='major',direction='in',length=10,labelsize='x-large')
plt.tick_params(axis='y',which='minor',direction='in',length=5)
plt.xlabel(r'$\sigma/\left<\sigma\right>$',fontsize=22)
plt.ylabel(r'$P(\sigma/\left<\sigma\right>)$',fontsize=22)
plt.tight_layout()
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.15),
#          fancybox=True, shadow=True, ncol=10)
#plt.legend()
#plt.title(f'{maxIndex} systems of {n} particles, pos->CirclePack')
plt.gcf().text(.05,.1,'E',color='black',bbox= dict(boxStyle='square',alpha=1,facecolor=[.9,.9,.9]),fontsize=26)
plt.xlim(0,2.2)
plt.ylim(9e-5,10)
plt.text(1.4,2,'26% polydispersity',fontsize='xx-large',color='black',alpha=1,va='top',rotation=-20)
plt.text(.45,2e-4,'20% polydispersity',fontsize='xx-large',color=[.3,.3,.3],alpha=1,rotation=65)
plt.yscale('log')

plt.savefig(f'../idealPackingLibrary/figures/{n}DistComp.pdf')
plt.savefig(f'../idealPackingLibrary/figures/{n}DistComp.png')
#plt.show()
