#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:51:56 2024

@author: violalum
"""
import sys
import numpy as np
import npquad
import imp
#pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import pyCudaPacking as pcp #configuring for radiiConstrainedDOFMinimizationSwitchable
import rigidpy as rp
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse
import random
import makeTriangulationFromPacking as mtfp

n=8192 #first command is number of particles
aspect1=1
aspect2=8
name = f'{aspect1}x{aspect2}LatticeVectors'
boxSize=np.array([aspect1,aspect2],dtype=np.quad)
packingType='posRad'
phi='.915'
numPackings= 30
packingPath=f'../../idealPackingLibrary/{n}'
cpPath=f'../../idealPackingLibrary/{n}/cpInputs'
packingPath=f'../../idealPackingLibrary/{n}'
cpPath=f'../../idealPackingLibrary/{n}/cpInputs'
for packno in range(0,numPackings):
	mtfp.writePackingToCP(n,packingPath,cpPath,f'{name}-{packno}',packingType=packingType,boxSize=boxSize)
