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

if __name__ == '__main__':
	directory=str(sys.argv[1])
	try:
		n=int(sys.argv[2])
		multiplePackings=True
	except:
		multiplePackings=False
	if(multiplePackings==False):
		filename=directory
		p=pcp.Packing()
		print(filename)
		p.load(filename)
		try:
			lv=np.loadtxt(f'{filename}/latticeVectors.dat')
		except:
			lv=np.array([[1,0],[0,1]],dtype=np.quad)
		p.setLatticeVectors(np.array(lv,dtype=np.quad))
		p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
#		p.draw2DPacking()
#		plt.show()
		p.minimizeFIRE('1e-20')
		print(p.getMaxUnbalancedForce())
		while(np.size(p.getContacts())/p.getNumParticles()<6):
			p.setPhi(p.getPhi()+np.quad('1e-6'))
			p.minimizeFIRE('1e-20')
		p.save(filename,overwrite=True)
	else:
		for i in range(n):
			print(i)
			filename =f'{directory}-{i}'
			p=pcp.Packing()
			p.load(filename)
			try:
				lv=np.loadtxt(f'{filename}/latticeVectors.dat')
			except:
				lv=np.array([[1,0],[0,1]],dtype=np.quad)
			p.setLatticeVectors(np.array(lv,dtype=np.quad))
			p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
			p.minimizeFIRE('1e-20')
			print(np.size(p.getContacts())/p.getNumParticles())
			while(np.size(p.getContacts())/p.getNumParticles() < 6):
				p.setPhi(p.getPhi()+np.quad('1e-6'))
				p.minimizeFIRE('1e-20')
			p.save(filename,overwrite=True)
