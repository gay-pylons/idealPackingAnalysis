#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:26:03 2024
Jumbled Pressure Sweep
@author: violalum
"""
import numpy as np
import imp
pcp=imp.load_source('pyCudaPacking','/home/vlum/code/pyCudaPacking/pyCudaPacking/__init__.py') #I'm Locust Ready!
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
	n=int(sys.argv[1]) #first command is number of particles
	name = str(sys.argv[2])
	p = pcp.Packing()
	p.load(f'../idealPackingLibrary/{n}/jumbledPackings/{name}')
	p.minimizeFIRE('1e-20')
	p.save(f'../idealPackingLibrary/{n}/jumblePressureSweep/{name}/{p.getPressure()}',overwrite=True)
	phiStart = p.getPhi()
	phiCEstimate = np.quad('.9')
	nStepsPerDecadePhi = 10
	pressureCutoff = np.quad('1e-7')
	print(p.getPressure())
	for stableList, excess, phiC in p.isostaticitySearch(phiCEstimate, phiStart, nStepsPerDecadePhi=nStepsPerDecadePhi, maxDeltaZ=0):
		p.save(f'../idealPackingLibrary/{n}/posMinPressureSweep2/{name}/{p.getPressure()}',overwrite=True)
		if p.getPressure() < pressureCutoff:
			break
		#data = delaunayPeriodicAngularSort(p)
		#writeCPShortSimple(data, p.getNumParticles(),f'cpInputs',f'{name}-{packno}')
