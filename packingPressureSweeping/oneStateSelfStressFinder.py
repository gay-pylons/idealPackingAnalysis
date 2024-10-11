#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 13:48:48 2023
Script to find correct pressure for 1 state self stressss
@author: violalum
"""

import numpy as np
import imp
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import sys
import matplotlib.pyplot as plt

nList=np.array([64, 128, 256,512,1024])

posMinPressure=[]
radMinPressure=[]
graphingNList=[]

for n in nList:
	for i in range(30):
		phiCEstimate = np.quad('.91')
		nStepsPerDecadePhi = 4
		pressureCutoff = np.quad('1e-9')
# =============================================================================
# 		p = pcp.Packing()
# 		p.load(f'../idealPackingLibrary/{n}/seedPackings(radMin)/idealPack{n}-{i}')
# 		phiStart = p.getPhi()
# 		for stableList, excess, phiC in p.isostaticitySearch(phiCEstimate, phiStart, nStepsPerDecadePhi=nStepsPerDecadePhi, maxDeltaZ=1):
# 			p.save(f'{n}/radPressureSweep/idealPack{n}-{i}/{p.getPressure()}',overwrite=True)
# 			if p.getPressure() < pressureCutoff:
# 				break
# 		radMinPressure.append(p.getPressure())
# 		print(p.getPressure())
# 		graphingNList.append(n)
# =============================================================================
		p = pcp.Packing(deviceNumber=1)
		p.load(f'../idealPackingLibrary/{n}/seedPackings(posMin)/posMin-{i}')
		phiStart = p.getPhi()
		for stableList, excess, phiC in p.isostaticitySearch(phiCEstimate, phiStart, nStepsPerDecadePhi=nStepsPerDecadePhi, maxDeltaZ=1):
			p.save(f'../idealPackingLibrary/{n}/posPressureSweep/posMin-{i}/{p.getPressure()}',overwrite=True)
			if p.getPressure() < pressureCutoff:
				break
		radMinPressure.append(p.getPressure())
		graphingNList.append(n)
		print(p.getPressure())
posMinPressure=np.array(posMinPressure)
radMinPressure=np.array(radMinPressure)
graphingNList=np.array(graphingNList)
#np.savez('1ssspressures.npz',pPress=posMinPressure,rPress=radMinPressure,n=graphingNList)
plt.semilogx(graphingNList,posMinPressure,'.')
plt.semilogx(graphingNList,radMinPressure,'.')
plt.show()
