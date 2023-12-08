#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:43:31 2023

@author: violalum
"""
#import pyCudaPacking as pcp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pylab import rcParams
#rcParams['figure.figsize'] = 10, 30

maxIndex=4
n=1024
posMinCP=f'../idealPackingLibrary/{n}/posCPPressureSweep/posMin'
radMinCP=f'../idealPackingLibrary/{n}/radCPPressureSweep/idealPack{n}'
posMin=f'../idealPackingLibrary/{n}/posMinPressureSweep/posMin'
radMin=f'../idealPackingLibrary/{n}/radMinPressureSweep/idealPack{n}'
pmcpD2Min=[]
rmcpD2Min=[]
rmD2Min=[]
pmD2Min=[]
pmcpData=[]
rmcpData=[]
pmData=[]
rmData=[]
for i in range(maxIndex):
#	pmcpData.append(np.load(f'{posMinCP}-{i}-pressureWalk.npz'))
#	rmcpData.append(np.load(f'{radMinCP}-{i}-pressureWalk.npz'))
#	pmData.append(np.load(f'{posMin}-{i}-pressureWalk.npz'))
#	rmData.append(np.load(f'{radMin}-{i}-d2MinField.npz'))
	pmcpD2Min.append(np.load(f'{posMinCP}-{i}-d2MinField.npz'))
	rmcpD2Min.append(np.load(f'{radMinCP}-{i}-d2MinField.npz'))
#	pmD2Min.append(np.load(f'{posMin}-{i}-d2MinField.npz'))
	rmD2Min.append(np.load(f'{radMin}-{i}-d2MinField.npz'))
#for i in range(maxIndex):
#plt.semilogx(rmcpData[0]["press"],rmcpData[0]["nContacts"]/1024,'x')
#plt.hist(rmcpData[0]['press'].astype(float),n_bins=10)
histBins=np.geomspace(1e-12,1e+1,num=40)
plt.hist(rmcpD2Min[1]['d2Min'][7].astype(float),bins=histBins,alpha=.5,density=True)
plt.hist(pmcpD2Min[2]['d2Min'][3].astype(float),bins=histBins,alpha=.5,density=True)
plt.hist(rmD2Min[3]['d2Min'][4].astype(float),alpha=.5,bins=histBins,density=True)
#plt.hist(np.log(pmD2Min[0]['d2Min'][].astype(float)),alpha=.5)
plt.xlabel('D2Min')
plt.ylabel('counts')
plt.xscale('log')
plt.title('')
# =============================================================================
# radMinPath0='1024/posCPPressureSweep/posMin-0/0.0055023261688460334861367953994068436'
# lv=np.loadtxt(f'{radMinPath0}/latticeVectors.dat')
# p=pcp.Packing()
# p.load(radMinPath0)
# p.setLatticeVectors(lv)
# colors = np.log(rmcpD2Min[0]['d2Min'][0])
# # Then linearlize it
# colors = (colors - np.min(colors)) / (-np.min(-colors) - np.min(colors))
# # Finally, get the colors:
# colors = cm.plasma(colors.astype(float))
# p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
# p.draw2DPacking(faceColor = colors)
# =============================================================================
