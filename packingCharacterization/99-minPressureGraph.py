#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 10:47:05 2023
1 state-self-stress-collectOr
@author: violalum
"""

import numpy as np
import imp
import os
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import sys
import matplotlib.pyplot as plt
posMinPressure=[]
radMinPressure=[]
nPackings=10
nArray=np.array([64,128,256,512,1024,2048,4096])
for n in nArray:
	radMinPressure.append([])
	posMinPressure.append([])
	directory=f'../idealPackingLibrary/{n}/posPressureSweep/posMin'
	for i in range(nPackings):
		firstFlag=True
		for filename in os.scandir(f'{directory}-{i}'):
			if(filename.is_dir()):
				p=pcp.Packing(deviceNumber=1)
				p.load(filename.path)
				if(p.getPressure()>0):
					if(firstFlag==True):
						minPressure=p.getPressure()
						firstFlag=False
					elif(p.getPressure()<minPressure):
						minPressure=p.getPressure()
		posMinPressure[len(posMinPressure)-1].append(minPressure)
	directory=f'../idealPackingLibrary/{n}/radPressureSweep/idealPack{n}'
	for i in range(nPackings):
		firstFlag=True
		for filename in os.scandir(f'{directory}-{i}'):
			if(filename.is_dir()):
				p=pcp.Packing(deviceNumber=1)
				p.load(filename.path)
				if(p.getPressure()>0):
					if(firstFlag==True):
						minPressure=p.getPressure()
						firstFlag=False
					elif(p.getPressure()<minPressure):
						minPressure=p.getPressure()
		radMinPressure[len(radMinPressure)-1].append(minPressure)
	print(f'{n} complete')
posMinPressure=np.array(posMinPressure)
radMinPressure=np.array(radMinPressure)
np.savez('1ssspressures.npz',pPress=posMinPressure,rPress=radMinPressure,n=nArray)
plt.loglog(nArray,np.mean(posMinPressure,axis=1),'.')
plt.loglog(nArray,np.mean(radMinPressure,axis=1),'.')
plt.show()
