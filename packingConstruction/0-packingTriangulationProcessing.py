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
pcp = SourceFileLoader("pyCudaPacking","/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py").load_module()
import sys
import matplotlib.pyplot as plt
from crystalComparison import wrapPosIntoBox
# imports the module from the given path

def puffPacking(filename,devNum,preMinPuff=np.quad('1e-7')):
	p=pcp.Packing(deviceNumber=devNum,potentialPower=2)
	p.load(filename)
	if(np.size(p.getContacts())/p.getNumParticles()<6):
		p.setPhi(p.getPhi()+preMinPuff)
	p.minimizeFIRE('1e-20')
	while(np.size(p.getContacts())/p.getNumParticles()<6):
		p.setPhi(p.getPhi()+np.quad('1e-6'))
		p.minimizeFIRE('1e-20')
		print(np.size(p.getContacts())/p.getNumParticles())
	p.save(filename,overwrite=True)

if __name__ == '__main__':
	directory=str(sys.argv[1])
	try:
		n=int(sys.argv[2])
		multiplePackings=True
	except:
		multiplePackings=False
	try:
		devNum=int(sys.argv[3])
	except:
		devNum=0
	if(multiplePackings==False):
		puffPacking(directory,devNum)
	else:
		for i in range(n):
			print(i)
			filename =f'{directory}-{i}'
			puffPacking(filename,devNum)
