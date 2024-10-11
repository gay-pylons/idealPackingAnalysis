#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:42:44 2022

@author: violalum
"""
import numpy as np
import pyCudaPacking as pcp
from modRatioVsN import moduliPerParticlePressure
from matplotlib import pyplot as plt
import matplotlib
#matplotlib.use('tkagg')
numList=np.array([1024])

pList=[]
phiList=[]
bList=[]
sList=[]
nContacts=[]
targetPressure=np.quad('1e-2')
for n in numList:
    for i in range(1):#sweep up
        fname=f'../idealPackingLibrary/{n}/finishedPackings/idealPack{n}-{i}'
        p=pcp.Packing(deviceNumber=1)
        p.load(f'{fname}-processed')
        lv=np.loadtxt(f'{fname}/latticeVectors.dat')
        p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
        p.setLatticeVectors(lv)
        while(p.getPressure()<=targetPressure):
            p.minimizeFIRE('1e-20')
            phiList.append(p.getPhi())
            pList.append(p.getPressure())
            s, b = moduliPerParticlePressure(p,fname,p.getPressure())
            bList.append(b)
            sList.append(s)
            nContacts.append(np.size(p.getContacts())/n)
            p.setPhi(p.getPhi()+p.getPressure())
    for i in range(1): #sweep down
        fname=f'../idealPackingLibrary/{n}/finishedPackings/idealPack{n}-{i}'
        p=pcp.Packing(deviceNumber=1)
        p.load(f'{fname}-processed')
        lv=np.loadtxt(f'{fname}/latticeVectors.dat')
        p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
        p.setLatticeVectors(lv)
        pressureCutoff = np.quad('1e-8')
        p.setPhi(p.getPhi()-p.getPressure()**(1.2))
        while(p.getPressure()>=pressureCutoff):
            p.minimizeFIRE('1e-20')
            phiList.append(p.getPhi())
            pList.append(p.getPressure())
            s, b = moduliPerParticlePressure(p,fname,p.getPressure())
            bList.append(b)
            sList.append(s)
            nContacts.append(np.size(p.getContacts())/n)
            p.setPhi(p.getPhi()-p.getPressure()**(1.2))
pList=np.array(pList)
sList=np.array(sList)
bList=np.array(bList)
contacts=np.array(nContacts)
phiList=np.array(phiList)
np.savez(f'{n}idealWalk.npz',sMod=sList,bMod=bList,nContacts=contacts,press=pList)
