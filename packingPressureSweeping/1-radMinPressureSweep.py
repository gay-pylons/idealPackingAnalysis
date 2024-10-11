#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:48:25 2023
posMinPressureSweep (isostaticitySearch)
@author: violalum
"""
import numpy as np
import imp
pcp=imp.load_source('pyCudaPacking','/home/vlum/code/pyCudaPacking/pyCudaPacking/__init__.py')
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
	n=int(sys.argv[1]) #first command is number of particles
	name = str(sys.argv[2])
	p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=1, numParticles=n)
	p.load(f'../idealPackingLibrary/{n}/radMin/{name}')
	p.minimizeFIRE('1e-20')
	p.save(f'../idealPackingLibrary/{n}/radMinPressureSweep2/{name}/{p.getPressure()}',overwrite=True)
	phiStart = p.getPhi()
	phiCEstimate = np.quad('.9')
	nStepsPerDecadePhi = 10
	pressureCutoff = np.quad('1e-7')
	print(p.getPressure())
	for stableList, excess, phiC in p.isostaticitySearch(phiCEstimate, phiStart, nStepsPerDecadePhi=nStepsPerDecadePhi, maxDeltaZ=0):
		p.save(f'../idealPackingLibrary/{n}/radMinPressureSweep2/{name}/{p.getPressure()}',overwrite=True)
		if p.getPressure() < pressureCutoff:
			break
		#data = delaunayPeriodicAngularSort(p)
		#writeCPShortSimple(data, p.getNumParticles(),f'cpInputs',f'{name}-{packno}')
