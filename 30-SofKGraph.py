#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:18:11 2023

@author: violalum
"""
import os
import matplotlib
matplotlib.use('Agg')
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

nArray=[64,128,256,512,1024,2048,4096,8192]#,16384,32768]

rMean=[]
cIt=0
kbin=np.logspace(np.log10(5), np.log10(1000), 100, dtype = np.quad)
for n in nArray:
	directory=f'../idealPackingLibrary/{n}/jumbledPackings/idealPack{n}'
	packingDirs= [f'{directory}-{i}/isostatic' for i in range(10)]
	sOfK=np.array(idealGlass.computeAverageSofK(packingDirs,latticeVectors=None,kbin=kbin)).astype(float)
	sOfKAvg=gmean(sOfK,axis=0)
	plt.loglog(sOfKAvg.T[0].astype(float)/(np.pi),sOfKAvg.T[1].astype(float),markerList[cIt],color=posColor,alpha=.5,fillstyle='none')
	cIt+=1
# =============================================================================
# for n in nArray:
# 	directory=f'../idealPackingLibrary/{n}/radMin/radMin'
# 	#packingDirs= [filename.path for filename in os.scandir(directory) if filename.is_dir()]
# 	packingDirs= [f'{directory}-{i}/isostatic' for i in range(10)]
# 	sOfK=idealGlass.computeAverageSofK(packingDirs,latticeVectors=False)
# #	print(sOfK)
# 	sOfKAvg=gmean(np.array(sOfK).astype(float),axis=0)
# #	sOfKStd=
# 	plt.loglog(sOfKAvg.T[0].astype(float)/np.sqrt(n),sOfKAvg.T[1].astype(float)*np.sqrt(n),markerList[cIt],color=posRadColor,alpha=.5)
# 	cIt+=1
# cIt=0
# =============================================================================
cIt=0
for n in nArray:
	directory=f'../idealPackingLibrary/{n}/finishedPackings/idealPack{n}'
	packingDirs= [f'{directory}-{i}' for i in range(10)]
	sOfK=np.array(idealGlass.computeAverageSofK(packingDirs,latticeVectors=True,kbin=kbin)).astype(float)
	sOfKAvg=gmean(sOfK,axis=0)
	plt.loglog(sOfKAvg.T[0].astype(float)/(np.pi),sOfKAvg.T[1].astype(float),markerList[cIt],color=posRadTriangColor,alpha=.5,fillstyle='none')
	cIt+=1
	
packingDirs= ['../idealPackingLibrary/8232/finishedPackings/crystal']
sOfK=np.array(idealGlass.computeAverageSofK(packingDirs,wibbled=True)).astype(float)
sOfKAvg=gmean(sOfK,axis=0)
#plt.loglog(sOfKAvg.T[0].astype(float),sOfKAvg.T[1].astype(float),markerList[7],color=[.3,.3,.3],alpha=1,fillstyle='none')
# =============================================================================
# for n in nArray:
# 	directory=f'../idealPackingLibrary/{n}/finishedPackings/posMin'
# 	#packingDirs= [filename.path for filename in os.scandir(directory) if filename.is_dir()]
# 	packingDirs= [f'{directory}-{i}' for i in range(10)]
# 	sOfK=idealGlass.computeAverageSofK(packingDirs,latticeVectors=True)
# #	print(sOfK)
# 	sOfKAvg=gmean(np.array(sOfK).astype(float),axis=0)
# #	sOfKStd=
# 	plt.loglog(sOfKAvg.T[0].astype(float)/np.sqrt(n),sOfKAvg.T[1].astype(float)*np.sqrt(n),markerList[cIt],color=posTriangColor,alpha=.5,fillstyle=None)
# 	cIt+=1
# =============================================================================

#plt.plot(1.5*np.array([2e-1,2]),np.array([2e-1,2])**2/100,'--',color='black')

plt.tick_params(axis='x',which='major',direction='inout',length=14,labelsize='x-large')
plt.tick_params(axis='x',which='minor',direction='inout',length=10)
plt.tick_params(axis='y',which='major',direction='in',length=10,labelsize='x-large')
plt.tick_params(axis='y',which='minor',direction='in',length=5)

plt.xlabel(r'$\left|k\right|/ k_{nn}$',fontsize='xx-large')
plt.ylabel(r'$S\left(\left|k\right|\right)$',fontsize='xx-large')
#plt.yscale('linear')
#plt.ylim([1e-4,1])
plt.xlim(np.array([1e-1,6])/(np.pi))
plt.tight_layout()
#plt.text(1,1e-3,r'$|k|^2$',rotation=34,color='black',fontsize='x-large')
plt.savefig('../idealPackingLibrary/figures/sOfK.pdf')
plt.savefig('../idealPackingLibrary/figures/sOfK.png')
#plt.show()
