#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:24:59 2022
Generates seed packings and circlePack input data
@author: violalum
#give n and name of packing: outputs circlePack script to relevant folder.
"""
import sys
import numpy as np
import npquad
import imp
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import matplotlib.pyplot as plt

#needs to run on radialDOFConstrainedMinimization
if __name__ == '__main__':
	n=int(sys.argv[1]) #first command is number of particles
	name = str(sys.argv[2])
	phi=np.quad(sys.argv[3])
	try:
		numPackings= int(sys.argv[4])
		for packno in range(numPackings):
			p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=0, numParticles=n)
			p.setRandomPositions()
			p.setLogNormalRadii(polyDispersity='0.2')
			p.setPhi(phi)
			p.minimizeFIRE('1e-20')
			print(p.getPressure())
			p.save(f'../idealPackingLibrary/{n}/posMin/{name}-{packno}',overwrite=True)
			print(packno)
	except:
		p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=0, numParticles=n)

		p.setRandomPositions()
		p.setLogNormalRadii(polyDispersity='0.2')
		p.setPhi(phi)
		p.minimizeFIRE('1e-20')
		print(p.getPressure())
		print(packno)
		p.save(f'../idealPackingLibrary/{n}/posMin/{name}',overwrite=True)
