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
	n_desired=int(sys.argv[1]) #first command is number of particles
	name = str(sys.argv[2])
	nSteps=int(sys.argv[3])
	try: 
		peakPressure=np.quad(float(sys.argv[4]))
	except: 
		peakPressure=np.quad('5e-2')
	try:
		p = pcp.Packing(deviceNumber=1)
		p.load(f'{p.getNumParticles()}/finishedPackings/{name}')
		lv=np.loadtxt(f'{p.getNumParticles()}/finishedPackings/{name}/latticeVectors.dat')
		p.setLatticeVectors(lv)
		p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
	except:
		p = pcp.Packing(deviceNumber=1)
		p.set2DHexCrystal(np.array([np.sqrt(n_desired*np.sqrt(3)/2),np.sqrt(n_desired/np.sqrt(3)/2).round()],dtype=int))
		lv=np.array([[p.getBoxSize()[0],0],[0,p.getBoxSize()[1]]],dtype=np.quad)
		p.setLatticeVectors(lv)
		p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
		p.save(f'{p.getNumParticles()}/finishedPackings/{name}',overwrite=True)
		np.savetxt(f'{p.getNumParticles()}/finishedPackings/{name}/latticeVectors.dat',lv)
	phiList=p.getPhi()+np.geomspace(1e-8,peakPressure.astype(float),num=nSteps).astype(np.quad)
	print(phiList)
	n=p.getNumParticles()
	for phi in phiList:
		print(phi)
		newName= f'{p.getNumParticles()}/hexCrystalPressureSweep/{name}/{phi}'
		try:
			p = pcp.Packing(deviceNumber=1)
			p.load(newName)
		except:
			p = pcp.Packing(deviceNumber=1)
			p.load(f'{n}/finishedPackings/{name}')
			lv=np.loadtxt(f'{p.getNumParticles()}/finishedPackings/{name}/latticeVectors.dat')
			p.setLatticeVectors(lv)
			p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
			p.setPhi(phi)
			p.minimizeFIRE('1e-20')
			p.save(newName,overwrite=True)
			np.savetxt(f'{newName}/latticeVectors.dat',lv)
