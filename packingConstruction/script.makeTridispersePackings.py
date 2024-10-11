#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:23:36 2024

@author: violalum
"""
import makeTriangulationFromPacking as mtfp

if __name__ == "__main__":
	n=100 #first command is number of particles
	name = '10kT1'
	packingType='poisson'
	seedName='hexCrystal'
	phi='.915'
	numPackings=10
	nT1=10000
	packingPath=f'../../idealPackingLibrary/{n}'
	cpPath=f'../../idealPackingLibrary/{n}/cpInputs'
	for packno in range(numPackings):
		mtfp.writePackingToCP(n,packingPath,cpPath,f'{name}-{packno}',packingType=packingType,nT1=nT1,seedName=seedName)