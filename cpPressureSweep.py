#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:48:25 2023
posMinPressureSweep (isostaticitySearch)
@author: violalum
"""
import numpy as np
import imp
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
	n=int(sys.argv[1]) #first command is number of particles
	name = str(sys.argv[2])
	try:
		pressureCutoff=np.quad(float(sys.argv[5]))
		print(pressureCutoff)
	except:
		pressureCutoff=np.quad('1e-6')
	try:
		peakPressure=np.quad(float(sys.argv[6]))
	except:
		peakPressure=np.quad('1e-2')
	numPackings= int(sys.argv[3])
	posRad= str(sys.argv[4])
	for packno in range(numPackings):
		p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=0, numParticles=n)
		p.load(f'{n}/finishedPackings/{name}-{packno}')
#		p.setPhi(p.getPhi()+peakPressure-p.getPressure())
		lv=np.loadtxt(f'{n}/finishedPackings/{name}-{packno}/latticeVectors.dat')
		p.setLatticeVectors(np.array(lv,dtype=np.quad))
		p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
		p.setPhi(p.getPhi()+peakPressure-p.getPressure())
		print(np.size(p.getContacts())/6)
		p.minimizeFIRE('1e-20')
		p.save(f'{n}/{posRad}CPPressureSweep/{name}-{packno}/{p.getPressure()}',overwrite=True)
		np.savetxt(f'{n}/{posRad}CPPressureSweep/{name}-{packno}/{p.getPressure()}/latticeVectors.dat',lv)
		phiStart = p.getPhi()
		phiCEstimate = np.quad('.9')
		while(p.getPressure()>pressureCutoff):
			p.setPhi(p.getPhi()-p.getPressure()**(1.2)) #for most pressures this is unproblematic I think
			p.minimizeFIRE('1e-20')
			if(p.getPressure()>0):
				p.save(f'{n}/{posRad}CPPressureSweep/{name}-{packno}/{p.getPressure()}',overwrite=True)
				np.savetxt(f'{n}/{posRad}CPPressureSweep/{name}-{packno}/{p.getPressure()}/latticeVectors.dat',lv)
		#data = delaunayPeriodicAngularSort(p)
		#writeCPShortSimple(data, p.getNumParticles(),f'cpInputs',f'{name}-{packno}')