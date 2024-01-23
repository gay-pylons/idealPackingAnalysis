#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:18:11 2023

@author: violalum
"""
import os
import imp
import numpy as np
from scipy.stats import gmean
import npquad
import matplotlib.pyplot as plt
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import idealGlass
import plotColorList

markerList=plotColorList.markerList
posColor=plotColorList.posColor
posRadColor=plotColorList.posRadColor
posTriangColor=plotColorList.posTriangColor
posRadTriangColor=plotColorList.posRadTriangColor

nArray=[64,128,256,512,1024,2048,4096]#,8192]#,16384,32768]

cIt=0
for n in nArray:
	directory=f'../idealPackingLibrary/{n}/posMin/posMin'
	#packingDirs= [filename.path for filename in os.scandir(directory) if filename.is_dir()]
	packingDirs= [f'{directory}-{i}/isostatic' for i in range(10)]
	sOfK=idealGlass.computeAverageSofK(packingDirs,latticeVectors=False)
#	print(sOfK)
	sOfKAvg=gmean(np.array(sOfK).astype(float),axis=0)
#	sOfKStd=
	plt.loglog(sOfKAvg.T[0].astype(float)/np.sqrt(n),sOfKAvg.T[1].astype(float)*np.sqrt(n),markerList[cIt],color=posColor,alpha=.5)
	cIt+=1
cIt=0
for n in nArray:
	directory=f'../idealPackingLibrary/{n}/radMin/radMin'
	#packingDirs= [filename.path for filename in os.scandir(directory) if filename.is_dir()]
	packingDirs= [f'{directory}-{i}/isostatic' for i in range(10)]
	sOfK=idealGlass.computeAverageSofK(packingDirs,latticeVectors=False)
#	print(sOfK)
	sOfKAvg=gmean(np.array(sOfK).astype(float),axis=0)
#	sOfKStd=
	plt.loglog(sOfKAvg.T[0].astype(float)/np.sqrt(n),sOfKAvg.T[1].astype(float)*np.sqrt(n),markerList[cIt],color=posRadColor,alpha=.5)
	cIt+=1
cIt=0
for n in nArray:
	directory=f'../idealPackingLibrary/{n}/finishedPackings/idealPack{n}'
	#packingDirs= [filename.path for filename in os.scandir(directory) if filename.is_dir()]
	packingDirs= [f'{directory}-{i}' for i in range(10)]
	#print(packingDirs)
	sOfK=idealGlass.computeAverageSofK(packingDirs,latticeVectors=True)
#	print(sOfK)
	sOfKAvg=gmean(np.array(sOfK).astype(float),axis=0)
#	sOfKStd=
	plt.loglog(sOfKAvg.T[0].astype(float)/np.sqrt(n),sOfKAvg.T[1].astype(float)*np.sqrt(n),markerList[cIt],color=posRadTriangColor,alpha=.5)
	cIt+=1
cIt=0
for n in nArray:
	directory=f'../idealPackingLibrary/{n}/finishedPackings/posMin'
	#packingDirs= [filename.path for filename in os.scandir(directory) if filename.is_dir()]
	packingDirs= [f'{directory}-{i}' for i in range(10)]
	sOfK=idealGlass.computeAverageSofK(packingDirs,latticeVectors=True)
#	print(sOfK)
	sOfKAvg=gmean(np.array(sOfK).astype(float),axis=0)
#	sOfKStd=
	plt.loglog(sOfKAvg.T[0].astype(float)/np.sqrt(n),sOfKAvg.T[1].astype(float)*np.sqrt(n),markerList[cIt],color=posTriangColor,alpha=.5)
	cIt+=1

plt.tick_params(axis='x',which='major',direction='inout',length=14,labelsize='x-large')
plt.tick_params(axis='x',which='minor',direction='inout',length=10)
plt.tick_params(axis='y',which='major',direction='in',length=10,labelsize='x-large')
plt.tick_params(axis='y',which='minor',direction='in',length=5)

plt.xlabel('$|k|/\sqrt{N}$',fontsize='xx-large')
plt.ylabel('$S( k )*\sqrt{N}$',fontsize='xx-large')
plt.xlim([8e-2,6])
plt.tight_layout()
plt.savefig('../idealPackingLibrary/figures/sOfK.pdf')
plt.savefig('../idealPackingLibrary/figures/sOfK.png')
plt.show()
