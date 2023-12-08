#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:29:43 2023

@author: violalum
"""

import sys
import numpy as np
import npquad
import imp
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import scipy.sparse
import rigidpy as rp
import networkx as nx
import matplotlib.pyplot as plt
#from cpReadyPacking import writeCPShortSimple

#returns delaunay triangulation vectors
def delaunayVectors(packing):
	longVectors=packing.getContactVectors(gap=np.quad('.5')).astype(float).tocsr()
	delaunay=packing.delaunayNeighbors()
	delVectors=scipy.sparse.csr_matrix((packing.getNumParticles(),2*packing.getNumParticles()), dtype=float)
	for i in range(packing.getNumParticles()):
		for j in delaunay[i].indices:
			delVectors[i,2*j]=longVectors[i,2*j]
			delVectors[i,2*j+1]=longVectors[i,2*j+1]
	print(np.size(delVectors))
	return delVectors.astype(float).tocoo()

def delaunayPeriodicAngularSort(packing): #stripped back version for large packigns and stuff
	contacts=packing.delaunayNeighbors()
	vecList=delaunayVectors(packing)
	connectivity=[]
	for i in range(packing.getNumParticles()):
		neighbors=contacts[i].indices
		neighborsX=vecList[i,2*neighbors].todense()
		neighborsY=vecList[i,2*neighbors+1].todense()
		angList=np.arctan2(neighborsY,neighborsX)
		connectivity.append(neighbors[np.argsort(angList)].flatten())
	return np.array(connectivity,dtype=object)

def writeCPShortSimple(connectivity, N,loc,name):
	blob = f"""<?xml version="1.0"?>
	
<CP_Scriptfile date="Apr 17, 2022">
<CPscript title="manual CP save" >
<text> process packing into circlepack; repack n=100000 </text>
<cmd iconname="amarok.png">act 0;Cleanse;Map 0 1;infile_read {name}.p;set_aim -d;alpha 10;newRed -t;repack 100000;layout -c;set_screen -a;Disp -w -C -Ra; </cmd>
<text> saves packing and lattice vectors </text>
<cmd>output :: Mob ::a b :: -f ~/Documents/code/idealPackingLibrary/{n}/latVecs/{name}-latVecs.dat;Write -cgrz ~/Documents/code/idealPackingLibrary/{n}/cpOutputs/{name}.p-dc</cmd>
	</CPscript>
<CPdata>
	<circlepacking name="{name}.p">\n"""

	with open(f'./{N}/{loc}/{name}-cpfile.xmd', "w") as f:
		f.write (blob)
		f.write('NODECOUNT: {:d}\n'.format(N))
		f.write('FLOWERS:\n')
		# write the connectivity
		
		for i in range(len(connectivity)):
			f.write(str(i+1)+' ')
			f.write(str(len(connectivity[i]))+' ')
			f.write(' '.join(list(map(str,connectivity[i]+1)))+' ')
			f.write(str(connectivity[i][0]+1)+'\n')

		f.write("""	</circlepacking>
  </CPdata>
</CP_Scriptfile>
		""")

if __name__ == '__main__':
	n=int(sys.argv[1]) #first command is number of particles
	name = str(sys.argv[2])
#	moments=[-3,-2,-1,1,2,3,4,5,6]
	moments=[-2,2,4]
	try:
		numPackings= int(sys.argv[3])
		p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=0, numParticles=n)
		for packno in range(numPackings):
			try:
				p.load(f'{n}/poissonPoints/{name}-{packno}')
			except:
				p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=0, numParticles=n)
				p.setRandomPositions()
				p.setLogNormalRadii(polyDispersity='.2')
			p.save(f'{n}/poissonPoints/{name}-{packno}',overwrite=True)
			p.setLatticeVectors(np.array([[1,0],[0,1]],dtype=np.quad))
			data=delaunayPeriodicAngularSort(p)
			writeCPShortSimple(data, p.getNumParticles(),f'cpInputs',f'{name}-{packno}')
			print(packno)
	except:
		p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=1, numParticles=n)
		try:
			p.load(f'{n}/poissonPoints/{name}')
		except:
			p.setRandomPositions()
			p.setRadii(np.zeros(p.getNumParticles(),dtype=np.quad))
		p.save(f'{n}/poissonPoints/{name}',overwrite=True)
		p.setLatticeVectors(np.array([[1,0],[0,1]],dtype=np.quad))
		data = delaunayPeriodicAngularSort(p)
		writeCPShortSimple(data, p.getNumParticles(),f'cpInputs',f'{name}')