#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:12:17 2023

@author: violalum
"""
import matplotlib
import numpy as np
import pyCudaPacking as pcp
#from modRatioVsN import moduliPerParticlePressure
from matplotlib import pyplot as plt
import scipy
import plotColorList
import math

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

def radiiFromDirectories(directoryList):
	phi, radii = [], []
	for directory in directoryList:
		p = pcp.Packing()
		p.load(directory)
		radii.append(p.getRadii().astype(float))
		phi.append(p.getPhi().astype(float))
	phi=np.array(phi,dtype=float)
	normFactor=(np.mean(phi)/(np.pi))**.5
	radii=np.array(radii).flatten().astype(float)
	return phi, radii, normFactor

def radiiGraph(directoryList,color,marker,numBins=30): #update to take desired fig. in future: use ax.semilogy
	phi, radii, normFactor = radiiFromDirectories(directoryList)
	print(f'disp = {np.std(radii)/np.mean(radii)}')
	hist,bin_edges=np.histogram(radii/normFactor,density=True,bins=numBins)
	bin_centers= np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])
	plt.semilogy(bin_centers,hist, marker, alpha=.7, color=color,fillstyle='none')
	return np.mean(phi)

if __name__=="__main__":

	nList= [4096]#[64,128,256,512,1024,2048]
	maxIndex=10
	binno=30
	filename=f'idealPack{n}'
	for n in nList:
		nIndex=int(np.round(math.log2(n)-5))
		marker=markerList[nIndex]
		directories = [f'../idealPackingLibrary/{n}/finishedPackings/{filename}-{i}' for i in range(maxIndex)]
		radiiGraph(directories,poissonColor2,marker)
		print(n)
#	disp=np.std(poissonRadii[mark])/np.mean(poissonRadii[mark])
#	plt.figure(figsize=(8,4))
#	sig=.2
	polyX2,polyY2=logNormalPlot(polyX,polyY, phi=np.mean(phi).astype(float),sigma=disp) 
	plt.tick_params(axis='x',which='major',direction='inout',length=14,labelsize='x-large')
	plt.tick_params(axis='x',which='minor',direction='inout',length=10)
	plt.tick_params(axis='y',which='major',direction='in',length=10,labelsize='x-large')
	plt.tick_params(axis='y',which='minor',direction='in',length=5)
	plt.xlabel(r'$\sigma/\left<\sigma\right>$',fontsize=22)
	plt.ylabel(r'$P(\sigma/\left<\sigma\right>)$',fontsize=22)
	plt.tight_layout()
	plt.yscale('log')
#	plt.xscale('log')
	plt.savefig(f'../idealPackingLibrary/figures/{n}triRadii.pdf')
	plt.savefig(f'../idealPackingLibrary/figures/{n}triRadii.png')
	plt.show()
