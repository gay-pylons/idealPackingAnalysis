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
import pyCudaPacking as pcp
import matplotlib.pyplot as plt

#needs to run on radialDOFConstrainedMinimization
if __name__ == '__main__':
	n=int(sys.argv[1]) #first command is number of particles
	name = str(sys.argv[2])
	phi=np.quad(sys.argv[3])
#	moments=[-3,-2,-1,1,2,3,4,5,6]
	moments=[-2,2,4]
	try:
		numPackings= int(sys.argv[4])
		for packno in range(numPackings):
			p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=1, numParticles=n)
			p.setRadConstraintMoments(moments)
			p.setRandomPositions()
			p.setLogNormalRadii(polyDispersity='0.2')
			p.setPhi(phi)
			p.minimizeFIRE('1e-20')
			print(p.getPressure())
			p.save(f'{n}/radMin/{name}-{packno}',overwrite=True)
			print(packno)
	except:
		p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=1, numParticles=n)
		p.setRadConstraintMoments(moments)
		p.setRandomPositions()
		p.setLogNormalRadii(polyDispersity='0.2')
		p.setPhi(phi)
		p.minimizeFIRE('1e-20')
		print(p.getPressure())
		print(packno)
		p.save(f'{n}/radMin/{name}',overwrite=True)