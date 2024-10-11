#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:01:26 2024

@author: violalum
"""
import pcpAnalysis.camLM.scheme.LMFunctions as lmf
#import pyCudaPacking as pcp
import imp
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import numpy as np
import npquad
import pcpAnalysis.eric.idealGlass as ig
import matplotlib.pyplot as plt
import sys
n = int(sys.argv[1])
packingDirectories = [f'idealPackingLibrary/{n}/seedPackings(posRadMin)/idealPack{n}-{i}' for i in range(30)]
saveDirectories = [f'idealPackingLibrary/{n}/finishedSquare/idealPack{n}-{i}' for i in range(30)]
for i in range(30):
	p = pcp.Packing()
	p.load(packingDirectories[i])
	z = ig.makeTriangulation(ig.createFlowerList(p.delaunayNeighbors(radical=True), p))
	p = lmf.triangulatePacking(p, z)
	p.save(saveDirectories[i],overwrite=True)
	pcp.fileIO.save2DArray(f'{saveDirectories[i]}/latticeVectors.dat',p.getLatticeVectors())
	
