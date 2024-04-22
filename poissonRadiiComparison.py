#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:12:17 2023

@author: violalum
"""
import matplotlib
#matplotlib.use('Agg')
import numpy as np
import pyCudaPacking as pcp
from modRatioVsN import moduliPerParticlePressure
from matplotlib import pyplot as plt
import scipy
import plotColorList

markerList=plotColorList.markerList
poissonColor1=np.array([.8,.3,1])
poissonColor2=np.array([.4,.3,.8])

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
poissonRadii=[]
phi=[]
nList=[64,128,256,512,1024,2048]
maxIndex=10
binno=30
filename='poissonPoints'
mark=-1
for n in nList:
	mark+=1
	print(n)
	poissonRadii.append([])
	phi.append([])
	for i in range(maxIndex):
		p = pcp.Packing()
		p.load(f'../idealPackingLibrary/{n}/finishedPackings/{filename}-{i}')
		poissonRadii[mark].append(p.getRadii().astype(float))
		phi[mark].append(p.getPhi().astype(float))
	phi[mark]=np.array(phi[mark],dtype=float)
	normFactor=(np.mean(phi)/(np.pi))**.5
	poissonRadii[mark]=np.array(poissonRadii[mark]).flatten().astype(float)
	hist,bin_edges=np.histogram(poissonRadii[mark]/normFactor,density=True,bins=binno)#/np.mean(poissonRadii[mark]),density=True,bins=binno)
	bin_centers= np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])
	poissonColor=poissonColor1*mark/2+poissonColor2*(2-mark)/2
	plt.semilogy(bin_centers,hist,markerList[mark+3],alpha=.7,color=poissonColor,fillstyle='none')
	
	print(n)
disp=np.std(poissonRadii[mark])/np.mean(poissonRadii[mark])
#plt.figure(figsize=(8,4))
print(disp)
sig=.2
polyX2,polyY2=logNormalPlot(poissonRadii[mark],nList[mark], phi=np.mean(phi).astype(float),sigma=disp)
#plt.semilogy(polyX2/np.mean(poissonRadii[mark]),polyY2*np.mean(poissonRadii[mark]),color='black',alpha=1)
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
#plt.xlim(0,2.2)
#plt.ylim(1e-4,10)
#plt.text(1.4,2,f'$\sigma={disp}$',fontsize='xx-large',color='black',alpha=1,va='top')
plt.yscale('log')
plt.xscale('log')
plt.savefig(f'../idealPackingLibrary/figures/{n}PoissonRadii.pdf')
plt.savefig(f'../idealPackingLibrary/figures/{n}PoissonRadii.png')
plt.show()
