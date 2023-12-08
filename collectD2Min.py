#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:58:26 2023

@author: violalum
"""
import os
import numpy as np
from importlib.machinery import SourceFileLoader
import sys
import matplotlib.pyplot as plt
# imports the module from the given path
pcp = SourceFileLoader("pyCudaPacking","/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py").load_module()

def d2MinField(packing,strain=np.quad('1e-2')):
	d2MinLischt=[]
	#copied from cDennis d2MinTest
	shearMatrix = np.identity(2, dtype = np.quad)
	shearMatrix[0, 1] = strain
	pos1 = packing.getPositions()
	lv1 = packing.getLatticeVectors()
	newPositions = np.dot(shearMatrix, pos1.T).T.copy()
	newLatticeVectors = np.dot(shearMatrix, packing.getLatticeVectors())
	packing.setPositions(newPositions)
	packing.setLatticeVectors(newLatticeVectors)
#	packing.draw2DPacking()
	# Now minimize this perturbed system:
	packing.minimizeFIRE("1e-20")
	meanDiameter = 2 * np.mean(packing.getRadii())
	cutDistance = meanDiameter * np.quad('1.5')

	# Now the algorithm. First, let's initialize it:
	packing.initD2MinJ(cutDistance)
	packing.calcD2MinJ(pos1, lv1)
	d2Min = packing.getD2Min() / meanDiameter**2
	d2Min.append
#	packing.draw2DPacking()
#	plt.show()
# =============================================================================
# 	try:
	print(np.max(d2Min.astype(float)))
# 	except:
# 		print(d2Min)
# =============================================================================
	return plasticShear, d2Min

if __name__ == '__main__':
	directory=str(sys.argv[1])
	try:
		strainPar=np.quad(sys.argv[3])
	except:
		strainPar=np.quad('1e-2')
	try:
		dev=np.quad(sys.argv[2])
	except:
		dev=0
	d2MinList=[]
	pressure=[]
	contacts=[]
	for filename in os.scandir(directory):
		if(filename.is_dir()):
			p=pcp.Packing(deviceNumber=dev)
			p.load(filename.path)
			try:
				lv=np.loadtxt(f'{filename.path}/latticeVectors.dat')
			except:
				lv=np.identity(2,dtype=np.quad)
			p.setLatticeVectors(np.array(lv,dtype=np.quad))
			p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
			pressure.append(p.getPressure())
			contacts.append(np.size(p.getContacts()))
			d2MinList.append(d2MinField(p,strain=strainPar))
	d2MinList=np.array(d2MinList)
	contacts=np.array(contacts)
	pressure=np.array(pressure)
#	print(pressure)
	np.savez(f'{directory}-d2MinField.npz',d2Min=d2MinList,nContacts=contacts,press=pressure)
