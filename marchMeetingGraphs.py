#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:05:41 2023

@author: violalum
"""
import numpy as np
import pyCudaPacking as pcp
from modRatioVsN import moduliPerParticlePressure
from matplotlib import pyplot as plt
import matplotlib
import scipy
n=1024
marginalData=np.load(f'{n}marginalWalk.npz')
idealData=np.load(f'{n}idealWalk.npz')

plt.figure(figsize=(6,6))
plt.subplot(2, 1, 1)
plt.loglog(np.array([np.quad('1e-6'),np.quad('1e-2')]),np.array([2.98649730664251,2.98649730664251]),'--',color='sienna')
plt.semilogx(idealData['press'],idealData['bMod'], '-s', label='circlePack bulk',color='darkviolet',linewidth=.5,alpha=.7)
plt.semilogx(marginalData['press'],marginalData['bMod'], 's', label='marginal bulk',color='navy',alpha=.5)
plt.xlim((np.quad('1e-6'),np.quad('1e-2')))
#plt.axes().get_xaxis()
#plt.loglog(pList,sList/bList,'+',label='circlePack K/G')
#plt.xlabel('pressure',fontsize=20)
plt.ylabel('G',fontsize=20)
plt.gca().set_xticks([])
plt.ylim((0,4))
plt.yticks(fontsize=12)

plt.tick_params(length=5)
#plt.legend()
plt.subplot(2, 1, 2)
plt.loglog(np.array([np.quad('1e-6'),np.quad('1e-2')]),np.array([0.741566527095123,0.7415665270951238]),'--',color='sienna')
plt.loglog(idealData['press'],idealData['sMod'], '-x', label='circlePack shear',color='darkviolet',linewidth=.5)
plt.loglog(marginalData['press'],marginalData['sMod'], 'x', label='marginal shear',color='navy')
plt.xlim((np.quad('1e-6'),np.quad('1e-2')))
plt.ylim((np.quad('1e-4'),np.quad('4e+0')))
plt.xlabel('pressure',fontsize=20)
plt.ylabel('K',fontsize=20)
#plt.legend()
plt.tight_layout()
plt.tick_params(length=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=10)
#.gca().set(xlabel=None)
plt.tight_layout()
plt.subplots_adjust(hspace=0)
#plt.tight_layout()
plt.savefig(f'{n}modulivPressure.svg')
plt.clf()
plt.figure(figsize=(6,5))
plt.semilogx(np.array([np.quad('1e-6'),np.quad('1e-2')]),np.array([6,6]),'--',color='indigo')
plt.semilogx(idealData['press'],idealData['nContacts'],'-o',label='CirclePack packings',color='chocolate',linewidth=.5)
plt.semilogx(marginalData['press'],marginalData['nContacts']/1024,'x',label='marginal packings',color=[.3,.75,.5],alpha=.5)
plt.xlabel('pressure',fontsize=20)
plt.ylabel('<z>',fontsize=20)
plt.tick_params(length=5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim((3.5,6.5))
plt.xlim((np.quad('1e-6'),np.quad('1e-2')))
plt.tight_layout()
plt.savefig(f'{n}ContactsvPressure.svg')