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
import rigidpy as rp
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse
#from cpReadyPacking import delaunayPeriodicAngularSort, writeCPShortSimple

#returns delaunay triangulation vectors
def delaunayVectors(packing): #in future: unite delaunayPASort and delaunayVectors (single sweep)
	longVectors=packing.getContactVectors(gap=5).astype(float).tocsr() #.3 is rough metric: may need to increase for lower density
# =============================================================================
# 	try:
# 		rDelaunay=packing.delaunayNeighbors(radical=True)
# 		delaunay=scipy.sparse.coo_matrix(rDelaunay)
# 		print('radical success dude!')
# 	except:
# =============================================================================
	rDelaunay=packing.delaunayNeighbors()
	delaunay=scipy.sparse.coo_matrix(rDelaunay)
	print('radical failure dude!')
	delVectors=scipy.sparse.csr_matrix((packing.getNumParticles(),2*packing.getNumParticles()), dtype=float)
	delVectors[delaunay.row,2*delaunay.col]=longVectors[delaunay.row,2*delaunay.col]
	delVectors[delaunay.row,2*delaunay.col+1]=longVectors[delaunay.row,2*delaunay.col+1]
	return rDelaunay,delVectors

def delaunayPeriodicAngularSort(packing): #stripped back version for large packigns and stuff
	contacts,vecList=delaunayVectors(packing)
	connectivity=[]
	for i in range(packing.getNumParticles()):
		neighbors=contacts[i].indices
		neighborsX=vecList[i,2*neighbors].todense()
		neighborsY=vecList[i,2*neighbors+1].todense()
		angList=np.arctan2(neighborsY,neighborsX)
		connectivity.append(neighbors[np.argsort(angList)].flatten())
	return np.array(connectivity,dtype=object)

#Identical to cpReadyPacking version: copied, since it fails when imported
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

	with open(f'{loc}/{name}-cpfile.xmd', "w") as f:
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

def writePackingToCP(n,packingPath,cpPath,name,index,phi):
	try:
		p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=0, numParticles=n)
		p.load(f'{packingPath}/{name}{phi}-{index}')
	except:
		p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=0, numParticles=n)
		p.setRandomPositions()
		p.setMonoRadii()
		p.setPhi(np.quad(phi))
	p.minimizeFIRE('1e-20')
# =============================================================================
# 	p.draw2DPacking()
# 	plt.show()
# 	plt.clf()
# =============================================================================
	p.save(f'{packingPath}/{name}{phi}-{index}',overwrite=True)
	p.setLatticeVectors(np.array([[1,0],[0,1]],dtype=np.quad))
	data = delaunayPeriodicAngularSort(p)
	writeCPShortSimple(data, p.getNumParticles(),cpPath,f'{name}{phi}-{index}')

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
		i='0'
	packingPath=f'../idealPackingLibrary/{n}/seedPackings(posMin)'
	cpPath=f'../idealPackingLibrary/{n}/cpInputs'
	if numPackings>1:
		for packno in range(numPackings):
			writePackingToCP(n,packingPath,cpPath,name,packno,phi)
	else:
		writePackingToCP(n,packingPath,cpPath,name,i,phi)

