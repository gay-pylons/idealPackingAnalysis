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

def moduliPerParticle(pName):
	packing=pcp.Packing()
	packing.load(pName)
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
	moduliPerParticle(directory)
