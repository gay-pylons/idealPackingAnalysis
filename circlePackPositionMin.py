#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:36:22 2023

@author: violalum
"""
import sys
import imp
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')

import matplotlib.pyplot as plt
import numpy as np
import npquad

if __name__ == '__main__':
	n=int(sys.argv[1]) #first command is number of particles
	name = str(sys.argv[2])
	try:
		numPackings= int(sys.argv[3])
		p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=1, numParticles=n)
		for packno in range(numPackings):
			try:
				p.load(f'{n}/seedPackings(radMin)/{name}-{packno}')
			except:
				p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=1, numParticles=n)
				p.setRandomPositions()
				p.setLogNormalRadii(polyDispersity='0.2')
				p.setPhi('.915')
			p.minimizeFIRE('1e-20')
			p.save(f'{n}/seedPackings(radMin)/{name}-{packno}',overwrite=True)
			p.setLatticeVectors(np.array([[1,0],[0,1]],dtype=np.quad))
	except:
		p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=1, numParticles=n)
		try:
			p.load(f'{n}/seedPackings(radMin)/{name}')
		except:
			p.setRandomPositions()
			p.setLogNormalRadii(polyDispersity='0.2')
			p.setPhi('.915')
		p.minimizeFIRE('1e-20')
#		p.draw2DPacking()
		plt.show()
		p.save(f'{n}/seedPackings(radMin)/{name}',overwrite=True)
		p.setLatticeVectors(np.array([[1,0],[0,1]],dtype=np.quad))
