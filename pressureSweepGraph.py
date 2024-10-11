#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:54:09 2023

Collects moduli pressure sweeps from directories and makes graphs
@author: violalum
"""
import os
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import npquad
import matplotlib
import imp
import plotColorList
from packingCharacterization.densityOfStates import getCPDoS
from packingCharacterization.densityOfStates import getOrdinaryDoS
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')

def unAveragedKGPeZ(directory,strEnd='pressureWalk.npz'):
	pressureList,bulkList,shearList,contactList = [np.array([],dtype=np.quad)]*4
	for file in os.scandir(directory):
		if file.name.endswith(strEnd):
			f=np.load(f'{directory}/{file.name}')
			pressureList=np.concatenate((pressureList,f['press']))
			bulkList=np.concatenate((bulkList,f['bMod']))
			shearList=np.concatenate((shearList,f['sMod']))
			contactList=np.concatenate((contactList,f['ZmZiso']))
	return pressureList,bulkList,shearList,contactList

def averagedKGPeZ(directory,strEnd='pressureWalk.npz'):
	pressureList,bulkList,shearList,contactList=[],[],[],[]
	for file in os.scandir(directory):
		if file.name.endswith(strEnd):
			f=np.load(f'{directory}/{file.name}')
			pressOrder=np.argsort(f['press'])
			press=f['press'][pressOrder]
			bMod=f['bMod'][pressOrder]
			sMod=f['sMod'][pressOrder]
			nContacts=(f['ZmZiso'][pressOrder])
			for i in range(len(f['press'])):
				try:
					pressureList[i].append(press[i])
					bulkList[i].append(bMod[i])
					shearList[i].append(sMod[i])
					contactList[i].append(nContacts[i])
				except:
					pressureList.append([])
					bulkList.append([])
					shearList.append([])
					contactList.append([])
					pressureList[i].append(press[i])
					bulkList[i].append(bMod[i])
					shearList[i].append(sMod[i])
					contactList[i].append(nContacts[i])
	for i in range(len(pressureList)):
		pressureList[i]=np.mean(np.array(pressureList[i]))
		bulkList[i]=np.mean(np.array(bulkList[i]))
		shearList[i]=np.mean(np.array(shearList[i]))
		contactList[i]=np.mean(np.array(contactList[i]))
	pressureList=np.array(pressureList)
	bulkList=np.array(bulkList)
	shearList=np.array(shearList)
	contactList=np.array(contactList)
	return pressureList,bulkList,shearList,contactList

if __name__ == "__main__":
	
	markerList=plotColorList.markerList
	posColor=plotColorList.posColor
	posRadColor=plotColorList.posRadColor
	posTriangColor=plotColorList.posTriangColor
	posRadTriangColor=plotColorList.posRadTriangColor
	strEnd='pressureWalk.npz'
	
	nArray=np.array([64,128,256,512,1024,2048,4096])
	
	#pieces to make tripel panel graph
	
	gs_top = plt.GridSpec(3, 1, hspace=0,bottom=.175,top=1,right=.95,left=.15)
	gs_base = plt.GridSpec(3, 1, bottom=.1,top=.925,right=.95,left=.15)
	fig = plt.figure(figsize=(6.5,10))
	
	axs = [fig.add_subplot(gs_top[0,:]),fig.add_subplot(gs_top[1,:]),fig.add_subplot(gs_base[2,:])]
	
	axs[0].sharex(axs[1])
	cIt = 0
	for n in nArray:
		cIt=np.log2(n).astype(int)-6
		print(cIt)
		pScale=1#n**2
		bScale=1#n
		sScale=1#n
		zScale=1#n
		Ziso=4*(1-1/n)
		pressureList,bulkList,shearList,contactList=averagedKGPeZ(f'../idealPackingLibrary/{n}/jumblePressureSweep')
		axs[0].semilogx(pressureList*pScale,bulkList*bScale,markerList[cIt],alpha=.7,fillstyle='none',color=posColor, label=f'$N={n}$')
		axs[1].loglog(pressureList*pScale,shearList*sScale,markerList[cIt],alpha=.7,fillstyle='none',color=posColor, label=f'$N={n}$')
		pressureList,bulkList,shearList,contactList=averagedKGPeZ(f'../idealPackingLibrary/{n}/radMinPressureSweep2')
		axs[0].semilogx(pressureList*pScale,bulkList*bScale,markerList[cIt],alpha=.7,fillstyle='none',color=posRadColor, label=f'$N={n}$')
		axs[1].loglog(pressureList*pScale,shearList*sScale,markerList[cIt],alpha=.7,fillstyle='none',color=posRadColor, label=f'$N={n}$')
		pressureList, bulkList, shearList, contactList=unAveragedKGPeZ(f'../idealPackingLibrary/{n}/radCPPressureSweep')
		pOrder=np.argsort(pressureList)
		axs[0].semilogx(pressureList[pOrder]*pScale,bulkList[pOrder]*bScale,markerList[cIt],alpha=.7,color=posRadTriangColor,fillstyle='none')
		axs[1].loglog(pressureList[pOrder]*pScale,shearList[pOrder]*sScale,markerList[cIt],alpha=.7,color=posRadTriangColor,fillstyle='none')
		f=np.load('../idealPackingLibrary/1554/hexCrystalPressureSweep/crystal-pressureWalk.npz')
		pOrder=np.argsort(f['press'])
		axs[0].semilogx(f['press'][pOrder],f['bMod'][pOrder],'-',alpha=.7,fillstyle='none',color='black',linewidth=3, label='$N=4012$')
		axs[1].loglog(f['press'][pOrder],f['sMod'][pOrder],'-',alpha=.7,fillstyle='none',color='black',linewidth=3, label='$N=4012$')
	axs[0].set_ylabel('$K/N$',fontsize='xx-large')
	axs[1].set_ylabel('$G/N$',fontsize='xx-large')
	axs[1].set_xlabel('$P$',fontsize='xx-large')
	#fig[2].set_ylabel('$(Z-Z_{iso})/N$',fontsize='xx-large')
	axs[0].set_ylim((0,3.3))
	axs[1].set_ylim((1e-4,2))
	axs[0].set_xlim((5e-7,2e-1))
	axs[1].set_xlim((5e-7,2e-1))
	nListStr=','.join(nArray.astype(str))
	#plt.tight_layout()
	plt.subplots_adjust(hspace=0)
	plt.subplots_adjust(wspace=.1)
	axs[0].tick_params(axis='x',which='major',direction='inout',length=14,labelsize='x-large')
	axs[0].tick_params(axis='x',which='minor',direction='inout',length=10)
	axs[0].set_label
	axs[1].tick_params(axis='x',which='major',direction='inout',length=14,labelsize='x-large')
	axs[1].tick_params(axis='x',which='minor',direction='inout',length=10)
	axs[0].tick_params(axis='y',which='major',direction='in',length=10,labelsize='x-large')
	axs[0].tick_params(axis='y',which='minor',direction='in',length=5)
	axs[1].tick_params(axis='y',which='major',direction='in',length=10,labelsize='x-large')
	axs[1].tick_params(axis='y',which='minor',direction='in',length=5)
	
	n=4096 #first command is number of particles
	numPackings= 10
	binSize= 100
	postriangdir=f'../idealPackingLibrary/{n}/finishedPackings/posMin'
	posradtriangdir=f'../idealPackingLibrary/{n}/finishedPackings/idealPack{n}'
	posdir=f'../idealPackingLibrary/{n}/jumbledPackings/idealPack{n}'
	posraddir=f'../idealPackingLibrary/{n}/radMin/radMin'
	crysdir='../idealPackingLibrary/4012/finishedPackings/crystal'
	states=getOrdinaryDoS(posdir,numPackings)
	print(len(states))
	hist, bin_edges=np.histogram(states,density=True,bins=np.unique(np.hstack([np.sort(states)[::binSize], np.max(states)])))
	bin_centers= np.array([np.sqrt(bin_edges[i]*bin_edges[i+1]) for i in range(len(bin_edges)-1)])
	fig.align_ylabels(axs)
	axs[2].loglog(bin_centers,hist,f'-{markerList[np.log2(n).astype(int)-6]}',alpha=.7,color=posColor,fillstyle='none')
	
	# =============================================================================
	# states=getOrdinaryDoS(posraddir,numPackings)
	# hist, bin_edges=np.histogram(states,density=True,bins=np.unique(np.hstack([np.sort(states)[::binSize], np.max(states)])))
	# bin_centers= np.array([np.sqrt(bin_edges[i]*bin_edges[i+1]) for i in range(len(bin_edges)-1)])
	# axs[2].loglog(bin_centers,hist,f'--{markerList[np.log2(n).astype(int)-6]}',alpha=.7,color=posRadColor,fillstyle='none')
	# 
	# states=getCPDoS(postriangdir,numPackings)
	# states=states.flatten()
	# hist, bin_edges=np.histogram(states,density=True,bins=np.unique(np.hstack([np.sort(states)[::binSize], np.max(states)])))
	# bin_centers= np.array([np.sqrt(bin_edges[i]*bin_edges[i+1]) for i in range(len(bin_edges)-1)])
	# axs[2].loglog(bin_centers,hist,f'-{markerList[np.log2(n).astype(int)-6]}',alpha=.7,color=posTriangColor,fillstyle='none')
	# 
	# =============================================================================
	
	
	states=getCPDoS(posradtriangdir,numPackings)
	states=states.flatten()
	hist, bin_edges=np.histogram(states,density=True,bins=np.unique(np.hstack([np.sort(states)[::binSize], np.max(states)])))
	bin_centers= np.array([np.sqrt(bin_edges[i]*bin_edges[i+1]) for i in range(len(bin_edges)-1)])
	axs[2].loglog(bin_centers,hist,f'-{markerList[np.log2(n).astype(int)-6]}',alpha=.7,color=posRadTriangColor,fillstyle='none')
	
	states=getCPDoS(crysdir,0)
	states=states.flatten()
	hist, bin_edges=np.histogram(states,density=True,bins=np.unique(np.hstack([np.sort(states)[::binSize], np.max(states)])))
	bin_centers= np.array([np.sqrt(bin_edges[i]*bin_edges[i+1]) for i in range(len(bin_edges)-1)])
	axs[2].loglog(bin_centers,hist,'-',alpha=.7,color='black',linewidth=3,fillstyle='none')
	
	plt.tick_params(axis='x',which='major',direction='inout',length=14,labelsize='x-large')
	plt.tick_params(axis='x',which='minor',direction='inout',length=10)
	plt.tick_params(axis='y',which='major',direction='in',length=10,labelsize='x-large')
	plt.tick_params(axis='y',which='minor',direction='in',length=5)
	
	fig.text(0.19,.13,'C',color='black',fontsize=20)#,bbox= dict(boxstyle='square',alpha=.7,facecolor=[.85,.85,.85],linewidth=0))
	fig.text(0.19,.48,'B',color='black',fontsize=20)#,bbox= dict(boxstyle='square',alpha=.7,facecolor=[.85,.85,.85],linewidth=0))
	fig.text(0.19,.76,'A',color='black',fontsize=20)#,bbox= dict(boxstyle='square',alpha=.7,facecolor=[.85,.85,.85],linewidth=0))
	
	plt.xscale('log')
	plt.xlabel('$\omega$',size='xx-large')
	plt.ylabel('$D(\omega)$',size='xx-large')#.set_rotation(270)
	#axs[2].yaxis.set_label_position("right")
	#ax.yaxis.tick_right()
	#plt.title('pos moduli are PLACEHOLDER')
	
	fig.savefig('../idealPackingLibrary/figures/moduliWalkPerParticle.pdf')
	fig.savefig('../idealPackingLibrary/figures/moduliWalkPerParticle.png')
	
