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


nArray=[64,128,256,512,1024,2048,4096]#,8192]#,16384,32768]
gradStart=np.array([.3,.7,.9])
gradEnd=np.array([.8,.1,.7])
cIt=0
for n in nArray:
	nColor=gradStart*(len(nArray)-(1+cIt))/(len(nArray)-1)+gradEnd*cIt/(len(nArray)-1)
	cIt+=1
	directory=f'../idealPackingLibrary/{n}/finishedPackings/idealPack{n}'
	#packingDirs= [filename.path for filename in os.scandir(directory) if filename.is_dir()]
	packingDirs= [f'{directory}-{i}' for i in range(10)]
	#print(packingDirs)
	sOfK=idealGlass.computeAverageSofK(packingDirs,latticeVectors=True)
#	print(sOfK)
	sOfKAvg=gmean(np.array(sOfK).astype(float),axis=0)
#	sOfKStd=
	plt.loglog(sOfKAvg.T[0].astype(float)/np.sqrt(n),sOfKAvg.T[1].astype(float)*np.sqrt(n),'x',color=nColor,alpha=.5)
for n in nArray:
	nColor=gradStart*(len(nArray)-(1+cIt))/(len(nArray)-1)+gradEnd*cIt/(len(nArray)-1)
	cIt+=1
	directory=f'../idealPackingLibrary/{n}/finishedPackings/posMin'
	#packingDirs= [filename.path for filename in os.scandir(directory) if filename.is_dir()]
	packingDirs= [f'{directory}-{i}' for i in range(10)]
	sOfK=idealGlass.computeAverageSofK(packingDirs,latticeVectors=True)
#	print(sOfK)
	sOfKAvg=gmean(np.array(sOfK).astype(float),axis=0)
#	sOfKStd=
	plt.loglog(sOfKAvg.T[0].astype(float)/np.sqrt(n),sOfKAvg.T[1].astype(float)*np.sqrt(n),'+',color=nColor,alpha=.5)
for n in nArray:
	nColor=gradStart*(len(nArray)-(1+cIt))/(len(nArray)-1)+gradEnd*cIt/(len(nArray)-1)
	cIt+=1
	directory=f'../idealPackingLibrary/{n}/posMinPackings'
	#packingDirs= [filename.path for filename in os.scandir(directory) if filename.is_dir()]
	packingDirs= [f'{directory}-{i}/isostatic' for i in range(10)]
	sOfK=idealGlass.computeAverageSofK(packingDirs,latticeVectors=True)
#	print(sOfK)
	sOfKAvg=gmean(np.array(sOfK).astype(float),axis=0)
#	sOfKStd=
	plt.loglog(sOfKAvg.T[0].astype(float)/np.sqrt(n),sOfKAvg.T[1].astype(float)*np.sqrt(n),'o',color=nColor,alpha=.5)
for n in nArray:
	nColor=gradStart*(len(nArray)-(1+cIt))/(len(nArray)-1)+gradEnd*cIt/(len(nArray)-1)
	cIt+=1
	directory=f'../idealPackingLibrary/{n}/radMinPackings'
	#packingDirs= [filename.path for filename in os.scandir(directory) if filename.is_dir()]
	packingDirs= [f'{directory}-{i}/isostatic' for i in range(10)]
	sOfK=idealGlass.computeAverageSofK(packingDirs,latticeVectors=True)
#	print(sOfK)
	sOfKAvg=gmean(np.array(sOfK).astype(float),axis=0)
#	sOfKStd=
	plt.loglog(sOfKAvg.T[0].astype(float)/np.sqrt(n),sOfKAvg.T[1].astype(float)*np.sqrt(n),'^',color=nColor,alpha=.5)

plt.xlabel('$|k|/\sqrt{n}$')
plt.ylabel('$S( k )*\sqrt{n}$')
plt.xlim([8e-2,7])
plt.savefig('../idealPackingLibrary/figures/sOfK.pdf')
plt.savefig('../idealPackingLibrary/figures/sOfK.png')
plt.show()
