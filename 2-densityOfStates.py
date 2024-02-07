#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:30:49 2023

@author: violalum
"""
import numpy as np
import imp
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import sys
import matplotlib.pyplot as plt
import plotColorList

markerList=plotColorList.markerList
posColor=plotColorList.posColor
posRadColor=plotColorList.posRadColor
posTriangColor=plotColorList.posTriangColor
posRadTriangColor=plotColorList.posRadTriangColor

def getCPDoS(directory,numPackings,peakPressure=np.quad('1e-2')): #direct towards 
	try:
		states=np.loadtxt(f'{directory}.DoS.xmd')
		return states
	except:
		states=[]
		for packno in range(numPackings):
			p = pcp.Packing()
			p.load(f'{directory}-{packno}')
			lv=np.loadtxt(f'{directory}-{packno}/latticeVectors.dat')
			p.setLatticeVectors(np.array(lv,dtype=np.quad))
			p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
			p.setPhi(p.getPhi()+peakPressure-p.getPressure())
			print(np.size(p.getContacts())/6)
			p.minimizeFIRE('1e-20')
			states.append(p.getOmegas())
		states=np.array(states)
		np.savetxt(f'{directory}.DoS.xmd',states)
		return states

def getOrdinaryDoS(directory,numPackings,pressureCutoff=np.quad('1e-6')):
	try:
		states=np.load(f'{directory}.DoS.npy')
		return states
	except:
		states=[]
		for packno in range(numPackings):
			try:
				p=pcp.Packing()
				p.load(f'{directory}-{packno}/isostatic')
			except:
				p = pcp.Packing()
				p.load(f'{directory}-{packno}')
				p.minimizeFIRE('1e-20')
				for stableList, excess, phiC in p.isostaticitySearch(np.quad('.9'), p.getPhi(), nStepsPerDecadePhi=2, maxDeltaZ=0):
					if p.getPressure() < pressureCutoff:
						break
				p.minimizeFIRE('1e-20')
				p.save(f'{directory}-{packno}/isostatic')
				print(p.getPressure())
			omegas=p.getOmegas()
			states.append(omegas)
		states=np.concatenate(states)
		np.save(f'{directory}.DoS.npy',states)
		return states

if __name__ == '__main__':
	posCPColor=plotColorList.posTriangColor
	posRadCPColor=plotColorList.posRadTriangColor
	posColor=plotColorList.posColor
	posRadColor=plotColorList.posRadColor
	n=int(sys.argv[1]) #first command is number of particles
	numPackings= int(sys.argv[2])
	binSize= int(sys.argv[3])
	postriangdir=f'../idealPackingLibrary/{n}/finishedPackings/posMin'
	posradtriangdir=f'../idealPackingLibrary/{n}/finishedPackings/idealPack{n}'
	posdir=f'../idealPackingLibrary/{n}/posMin/posMin'
	posraddir=f'../idealPackingLibrary/{n}/radMin/radMin'
	states=getOrdinaryDoS(posdir,numPackings)
	print(len(states))
	hist, bin_edges=np.histogram(states,density=True,bins=np.unique(np.hstack([np.sort(states)[::binSize], np.max(states)])))
	bin_centers= np.array([np.sqrt(bin_edges[i]*bin_edges[i+1]) for i in range(len(bin_edges)-1)])
	plt.loglog(bin_centers,hist,'-o',alpha=.5,color=posColor,fillstyle='none')
	
	states=getOrdinaryDoS(posraddir,numPackings)
	hist, bin_edges=np.histogram(states,density=True,bins=np.unique(np.hstack([np.sort(states)[::binSize], np.max(states)])))
	bin_centers= np.array([np.sqrt(bin_edges[i]*bin_edges[i+1]) for i in range(len(bin_edges)-1)])
	plt.loglog(bin_centers,hist,'-^',alpha=.5,color=posRadColor,fillstyle='none')
	
	states=getCPDoS(postriangdir,numPackings)
	states=states.flatten()
	hist, bin_edges=np.histogram(states,density=True,bins=np.unique(np.hstack([np.sort(states)[::binSize], np.max(states)])))
	bin_centers= np.array([np.sqrt(bin_edges[i]*bin_edges[i+1]) for i in range(len(bin_edges)-1)])
	plt.loglog(bin_centers,hist,'-+',alpha=.5,color=posTriangColor,fillstyle='none')
	
	states=getCPDoS(posradtriangdir,numPackings)
	states=states.flatten()
	hist, bin_edges=np.histogram(states,density=True,bins=np.unique(np.hstack([np.sort(states)[::binSize], np.max(states)])))
	bin_centers= np.array([np.sqrt(bin_edges[i]*bin_edges[i+1]) for i in range(len(bin_edges)-1)])
	plt.loglog(bin_centers,hist,'-x',alpha=.5,color=posRadTriangColor,fillstyle='none')
	
	plt.tick_params(axis='x',which='major',direction='inout',length=14,labelsize='x-large')
	plt.tick_params(axis='x',which='minor',direction='inout',length=10)
	plt.tick_params(axis='y',which='major',direction='in',length=10,labelsize='x-large')
	plt.tick_params(axis='y',which='minor',direction='in',length=5)
	
	plt.xscale('log')
	plt.xlabel('$\omega$',size='xx-large')
	plt.ylabel('$D(\omega)$',size='xx-large')
#	plt.title(f'$N={n}$')
	plt.tight_layout()
	plt.savefig(f'../idealPackingLibrary/figures/{n}DoS.pdf')
	plt.savefig(f'../idealPackingLibrary/figures/{n}DoS.png')
#	plt.show()
	plt.clf()
