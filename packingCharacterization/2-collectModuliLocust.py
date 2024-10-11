#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2023
sweeps directories of packings in order to produce collection of moduli and save to repository.
@author: violalum
"""
import os
import numpy as np
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("pyCudaPacking","/home/vlum/code/pyCudaPacking/pyCudaPacking/__init__.py")
pcp = importlib.util.module_from_spec(spec)
sys.modules["pyCudaPacking"] = pcp
spec.loader.exec_module(pcp)

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
if __name__ == '__main__':
	directory=str(sys.argv[1])
	try:
		n=int(sys.argv[2])
		multiplePackings=True
	except:
		multiplePackings=False
	shear=[]
	bulk=[]
	pressure=[]
	contacts=[]
	zExcess=[]
	if(multiplePackings==False):
		for filename in os.scandir(directory):
			if(filename.is_dir()):
				print(filename.path)
				p=pcp.Packing(deviceNumber=0)
				p.load(filename.path)
				try:
					lv=np.loadtxt(f'{filename.path}/latticeVectors.dat')
				except:
					lv=[[1,0],[0,1]]
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
	else:
		for i in range(n):
			shear=[]
			bulk=[]
			pressure=[]
			contacts=[]
			zExcess
			for filename in os.scandir(f'{directory}-{i}'):
				if(filename.is_dir()):
					p=pcp.Packing(deviceNumber=0)
					p.load(filename.path)
					try:
						lv=np.loadtxt(f'{filename.path}/latticeVectors.dat')
					except:
						lv=[[1,0],[0,1]]
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
			np.savez(f'{directory}-{i}-pressureWalk.npz',sMod=shear,bMod=bulk,nContacts=contacts,press=pressure,ZmZiso=zExcess)
