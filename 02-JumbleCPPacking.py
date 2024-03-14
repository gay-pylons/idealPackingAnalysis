#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:40:21 2024

@author: violalum
"""
import sys
import numpy as np
import npquad
import imp
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import rigidpy as rp
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse

def jumblePacking(inPath,outPath,phi):#loads path
	try:
		p = pcp.Packing()
		p.load(inPath)
		p.setPhi(phi)
		p.setRandomPositions()
		p.minimizeFIRE('1e-20')
		p.save(outPath)
		print(f'jumble succeded @ {inPath}')
	except:
		print(f'Jumble failed @ {inPath}')

if __name__ == '__main__':
	n=int(sys.argv[1]) #first command is number of particles
	name = str(sys.argv[2])
	try:
		phi=str(sys.argv[3])
	except:
		phi='.915'
	try:
		numPackings= int(sys.argv[4])
	except:
		numPackings= 0
	try:
		i=int(sys.argv[5])
	except:
		i=0
	inPath=f'../idealPackingLibrary/{n}/finishedPackings/{name}'
	outPath=f'../idealPackingLibrary/{n}/jumbledPackings/{name}'
	if numPackings > 1:
		for index in range(numPackings):
			jumblePacking(f'{inPath}-{index}/',f'{outPath}-{index}/',phi)
	else:
		jumblePacking(f'{inPath}-{i}',f'{outPath}-{i}',phi)