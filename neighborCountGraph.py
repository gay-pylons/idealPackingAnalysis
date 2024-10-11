#!/usr/bin/env python3ee on dandylion
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:58:15 2024

@author: violalum
"""
import imp
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import matplotlib.pyplot as plt
import numpy as np

directoryList=[f'../idealPackingLibrary/100/finishedPackings/triDisperse-{i}' for i in range(10)]

zList=[]
neighborCount=np.zeros(10,dtype=int)
for directory in directoryList:
	p = pcp.Packing()
	p.load(directory)
	zList.append([np.size(p.getContacts()[i]) for i in range(100)])
zList=np.array(zList).flatten()
for conNum in zList:
	neighborCount[conNum] += 1
neighborCount=neighborCount/len(zList)
neighborNums=np.array([i for i in range(10)])
plt.xlabel('neighbor count')
plt.ylabel('probability')
plt.plot(neighborNums,neighborCount,'x')
plt.show()

