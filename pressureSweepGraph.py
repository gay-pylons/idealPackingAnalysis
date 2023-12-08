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

strEnd='pressureWalk.npz'
nArray=np.array([64,128,256,512,1024,2048])#,4096])#,8192])#,|16384])
#nArray=np.array([4096])
ax,fig=plt.subplots(3,1,sharex=True,figsize=(6,9))
cIt = 0
gradStart=np.array([.3,.7,.9])
gradEnd=np.array([.8,.1,.7])
for n in nArray:
	nColor=gradStart*(len(nArray)-(1+cIt))/(len(nArray)-1)+gradEnd*cIt/(len(nArray)-1)
	print(cIt/(len(nArray)-1)+(len(nArray)-(1+cIt))/(len(nArray)-1))
	cIt += 1
	Ziso=4*(1-1/n)
	pressureList=np.array([],dtype=np.quad)
	bulkList=np.array([],dtype=np.quad)
	shearList=np.array([],dtype=np.quad)
	contactList=np.array([],dtype=np.quad)
	for file in os.scandir(f'../idealPackingLibrary/{n}/radCPPressureSweep'):
		if file.name.endswith(strEnd):
			f=np.load(f'../idealPackingLibrary/{n}/radCPPressureSweep/{file.name}')
			bMax=1#np.max(f['bMod'])
#			plt.loglog(f['press'],f['bMod'],'.')
#			plt.loglog(f['press'],f['sMod'],'x')
			pressureList=np.concatenate((pressureList,f['press']))
			bulkList=np.concatenate((bulkList,f['bMod']/bMax))
			shearList=np.concatenate((shearList,f['sMod']/bMax))
			contactList=np.concatenate((contactList,f['ZmZiso']))
	pOrder=np.argsort(pressureList)
	fig[0].semilogx(pressureList[pOrder],bulkList[pOrder],'x',alpha=.2,color=nColor,fillstyle='none')
	fig[1].loglog(pressureList[pOrder],shearList[pOrder],'x',alpha=.2,color=nColor,fillstyle='none')
	fig[2].loglog(pressureList[pOrder],contactList[pOrder],'x',alpha=.2,color=nColor,fillstyle='none')
	pressureList=[]
	pressureList=np.array([],dtype=np.quad)
	bulkList=np.array([],dtype=np.quad)
	shearList=np.array([],dtype=np.quad)
	contactList=np.array([],dtype=np.quad)
	for file in os.scandir(f'../idealPackingLibrary/{n}/posCPPressureSweep'):
		if file.name.endswith(strEnd):
			f=np.load(f'../idealPackingLibrary/{n}/posCPPressureSweep/{file.name}')
			bMax=1#np.max(f['bMod'])
#			plt.loglog(f['press'],f['bMod'],'.')
#			plt.loglog(f['press'],f['sMod'],'x')
			pressureList=np.concatenate((pressureList,f['press']))
			bulkList=np.concatenate((bulkList,f['bMod']/bMax))
			shearList=np.concatenate((shearList,f['sMod']/bMax))
			contactList=np.concatenate((contactList,f['ZmZiso']))
	pOrder=np.argsort(pressureList)
	fig[0].semilogx(pressureList[pOrder],bulkList[pOrder],'+',alpha=.2,color=nColor,fillstyle='none')
	fig[1].loglog(pressureList[pOrder],shearList[pOrder],'+',alpha=.2,color=nColor,fillstyle='none')
	fig[2].loglog(pressureList[pOrder],contactList[pOrder],'+',alpha=.2,color=nColor,fillstyle='none')
	pressureList=[]
	bulkList=[]
	shearList=[]
	contactList=[]
	for file in os.scandir(f'../idealPackingLibrary/{n}/posMinPressureSweep2'):
		if file.name.endswith(strEnd):
			f=np.load(f'../idealPackingLibrary/{n}/posMinPressureSweep2/{file.name}')
			bMax=1#np.max(f['bMod'])
#			plt.loglog(f['press'],f['bMod'],'.')
#			plt.loglog(f['press'],f['sMod'],'x')
			pressOrder=np.argsort(f['press'])
			press=f['press'][pressOrder]
			bMod=f['bMod'][pressOrder]
			sMod=f['sMod'][pressOrder]
			nContacts=(f['ZmZiso'][pressOrder])
			for i in range(len(f['press'])):
				try:
					pressureList[i].append(press[i])
					bulkList[i].append(bMod[i]/bMax)
					shearList[i].append(sMod[i]/bMax)
					contactList[i].append(nContacts[i])
				except:
					pressureList.append([])
					bulkList.append([])
					shearList.append([])
					contactList.append([])
					pressureList[i].append(press[i])
					bulkList[i].append(bMod[i]/bMax)
					shearList[i].append(sMod[i]/bMax)
					contactList[i].append(nContacts[i])
	for i in range(len(pressureList)):
		pressureList[i]=np.mean(np.array(pressureList[i]))#np.exp(np.mean(np.log(np.array(pressureList[i]))))
		bulkList[i]=np.mean(np.array(bulkList[i]))
		shearList[i]=np.mean(np.array(shearList[i]))
		contactList[i]=np.mean(np.array(contactList[i]))
#	print(contactList)
	pressureList=np.array(pressureList)
	bulkList=np.array(bulkList)
	shearList=np.array(shearList)
	contactList=np.array(contactList)
	fig[0].semilogx(pressureList,bulkList,'o',alpha=.7,fillstyle='none',color=nColor, label=f'$N={n}$',markeredgewidth='2')
	fig[1].loglog(pressureList,shearList,'o',alpha=.7,fillstyle='none',color=nColor, label=f'$N={n}$',markeredgewidth='2')
	fig[2].loglog(pressureList,contactList,'o',alpha=.7,fillstyle='none',color=nColor, label=f'$N={n}$',markeredgewidth='2')
	pressureList=[]
	bulkList=[]
	shearList=[]
	contactList=[]
#	print(shearList)
	for file in os.scandir(f'../idealPackingLibrary/{n}/radMinPressureSweep2'):
		if file.name.endswith(strEnd):
			f=np.load(f'../idealPackingLibrary/{n}/radMinPressureSweep2/{file.name}')
			bMax=1#np.max(f['bMod'])
#			plt.loglog(f['press'],f['bMod'],'.')
#			plt.loglog(f['press'],f['sMod'],'x')
			pressOrder=np.argsort(f['press'])
			press=f['press'][pressOrder]
			bMod=f['bMod'][pressOrder]
			sMod=f['sMod'][pressOrder]
			nContacts=(f['ZmZiso'][pressOrder])
			for i in range(len(f['press'])):
				try:
					pressureList[i].append(press[i])
					bulkList[i].append(bMod[i]/bMax)
					shearList[i].append(sMod[i]/bMax)
					contactList[i].append(nContacts[i])
				except:
					pressureList.append([])
					bulkList.append([])
					shearList.append([])
					contactList.append([])
					pressureList[i].append(press[i])
					bulkList[i].append(bMod[i]/bMax)
					shearList[i].append(sMod[i]/bMax)
					contactList[i].append(nContacts[i])
	for i in range(len(pressureList)):
		pressureList[i]=np.mean(np.array(pressureList[i]))#np.exp(np.mean(np.log(np.array(pressureList[i]))))
		bulkList[i]=np.mean(np.array(bulkList[i]))
		shearList[i]=np.mean(np.array(shearList[i]))
		contactList[i]=np.mean(np.array(contactList[i]))
#	print(contactList)
	pressureList=np.array(pressureList)
	bulkList=np.array(bulkList)
	shearList=np.array(shearList)
	contactList=np.array(contactList)
#	print(shearList)
	fig[0].semilogx(pressureList,bulkList,'^',alpha=.7,fillstyle='none',color=nColor, label=f'$N={n}$',markeredgewidth='2')
	fig[1].loglog(pressureList,shearList,'^',alpha=.7,fillstyle='none',color=nColor, label=f'$N={n}$',markeredgewidth='2')
	fig[2].loglog(pressureList,contactList,'^',alpha=.7,fillstyle='none',color=nColor, label=f'$N={n}$',markeredgewidth='2')
f=np.load('../idealPackingLibrary/1554/hexCrystalPressureSweep/crystal-pressureWalk.npz')
pOrder=np.argsort(f['press'])
fig[0].semilogx(f['press'][pOrder],f['bMod'][pOrder],'--',alpha=.7,fillstyle='none',color=[.3,.3,.3], label='$N=4012$')
fig[1].loglog(f['press'][pOrder],f['sMod'][pOrder],'--',alpha=.7,fillstyle='none',color=[.3,.3,.3], label='$N=4012$')
fig[2].loglog(f['press'][pOrder],f['ZmZiso'][pOrder],'--',alpha=.7,fillstyle='none',color=[.3,.3,.3], label='$N=4012$')
plt.xlabel('$P$')
fig[0].set_ylabel('$K/N$',fontsize='x-large')
fig[1].set_ylabel('$G/N$',fontsize='x-large')
fig[2].set_ylabel('$(Z-Z_{iso})/N$',fontsize='x-large')
fig[0].set_ylim((0,4))
fig[1].set_ylim((1e-4,3))
plt.xlim((4e-7,7e-2))
nListStr=','.join(nArray.astype(str))
#fig[0].title.set_text(f'Pressure Sweep n={nListStr}')
plt.tight_layout()
plt.subplots_adjust(hspace=0)
fig[0].tick_params(axis='x',which='major',direction='inout',length=14)
fig[0].tick_params(axis='x',which='minor',direction='inout',length=10)
fig[1].tick_params(axis='x',which='major',direction='inout',length=14)
fig[1].tick_params(axis='x',which='minor',direction='inout',length=10)
fig[2].tick_params(axis='x',which='major',direction='inout',length=14,labelsize='large')
fig[2].tick_params(axis='x',which='minor',direction='inout',length=10)
fig[0].tick_params(axis='y',which='major',length=10,labelsize='large')
fig[0].tick_params(axis='y',which='minor',length=5)
fig[1].tick_params(axis='y',which='major',length=10,labelsize='large')
fig[1].tick_params(axis='y',which='minor',length=5)
fig[2].tick_params(axis='y',which='major',length=10,labelsize='large')
fig[2].tick_params(axis='y',which='minor',length=5)
#plt.legend(fontsize=11)
ax.savefig('../idealPackingLibrary/figures/moduliWalk.pdf')
ax.savefig('../idealPackingLibrary/figures/moduliWalk.png')

