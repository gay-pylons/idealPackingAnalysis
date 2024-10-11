#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2023
sweeps directories of packings in order to produce collection of moduli and save to repository.
@author: violalum
"""
import os
import numpy as np
from importlib.machinery import SourceFileLoader
import sys
import matplotlib.pyplot as plt
# imports the module from the given path
pcp = SourceFileLoader("pyCudaPacking","/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py").load_module()
i=0
def moduliPerParticle(packing,pName):
	shearStrain = np.array([0,1,0,0])
	bulkStrain = np.array([1,0,0,1])
	try:
		stiffness=np.load(f'{pName}.stiffnessMatrix.npy')
	except:
		stiffness = packing.getStiffnessMatrix()
		np.save(f'{pName}.stiffnessMatrix.npy',stiffness)
	shear=np.matmul(stiffness.astype(float), shearStrain.astype(float))[1]/p.getNumParticles()
	bulk= np.sum(np.matmul(stiffness.astype(float), bulkStrain.astype(float))[[0,3]])/p.getNumParticles()/2
	return shear, bulk

def collectGKPeZ(filename): #collects bulk, shear, pressure and Z_excess from pressure sweep dir 
	shear=[]
	bulk=[]
	pressure=[]
	contacts=[]
	zExcess=[]
	for filename in os.scandir(directory):
		if(filename.is_dir()):
			p=pcp.Packing()
			print(filename.path)
			p.load(filename.path)
			if(p.getPressure()>0): #
				try:
					lv=np.loadtxt(f'{filename.path}/latticeVectors.dat')
				except:
					lv=np.array([[1,0],[0,1]],dtype=np.quad)
				p.setLatticeVectors(np.array(lv,dtype=np.quad))
				p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
				s, b = moduliPerParticle(p,f'{filename.path}')
				shear.append(s)
				bulk.append(b)
				pressure.append(p.getPressure())
				contacts.append(np.size(p.getContacts()))
				zExcess.append(2*(p.getStableListAndExcess()[1]+1)/np.sum(p.getStableListAndExcess()[0]))
	contacts=np.array(contacts)
	pressure=np.array(pressure)
	shear=np.array(shear)
	bulk=np.array(bulk)
	zExcess=np.array(zExcess)
	np.savez(f'{directory}-pressureWalk.npz',sMod=shear,bMod=bulk,nContacts=contacts,press=pressure,ZmZiso=zExcess)

if __name__ == '__main__':
	directory=str(sys.argv[1])
	try:
		n=int(sys.argv[2])
		multiplePackings=True
	except:
		multiplePackings=False
#	print(multiplePackings)

	if(multiplePackings==False):
		collectGKPeZ(directory)
	else:
		for i in range(n):
			collectGKPeZ(f'{directory}-{i}')
