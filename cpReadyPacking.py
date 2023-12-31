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
import rigidpy as rp
import networkx as nx
import matplotlib.pyplot as plt

def writeCPShort(connectivity, N,loc,name):
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
		
		for items in connectivity:
			f.write(str(items+1)+' ')
			f.write(str(len(connectivity[items]))+' ')
			f.write(' '.join(list(map(str,connectivity[items]+1)))+' ')
			f.write(str(connectivity[items][0]+1)+'\n')

		f.write("""	</circlepacking>
  </CPdata>
</CP_Scriptfile>
		""")
		
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

def PeriodicAngularSort(coordinates, bonds, basis):#-np.transpose(vex))
	# Using PBCs, find the correct neighbors and sort them around a vertex.
	sites = {}
	angles = {}
	
	N,M = len(coordinates), len(bonds)
	F = rp.framework(coordinates, bonds, basis)
	L = F.edgeLengths()
	R = F.rigidityMatrix()
	G = nx.Graph(b for b in bonds)

	# dictionary to map a bond to its index in rgidity matrix
	bond_map={}
	for i,bond in enumerate(bonds):
		label = '{},{}'.format(*bond)
		bond_map[label]=i
	
	# loop over all sites
	for index in range(N):
		print(index)
		cols=(2*index,2*index+1)
		neighs = np.array([n for n in G.neighbors(index)])
		arr=[]
		for ne in neighs:
			bond = [index,ne]
			label = '{},{}'.format(*sorted(bond))
			x = R[bond_map[label],cols[0]]
			y = R[bond_map[label],cols[1]]
			arr.append([x,y])
		arr = np.array(arr)
		angle = np.arctan2(arr[:,1], arr[:,0])*180/(np.pi)
		sortedarg = np.argsort(angle)
		sorted_sites = neighs[sortedarg]
		sites[index] = sorted_sites
		angles[index] = angle[sortedarg]
	return sites, angles

def simplePeriodicAngularSort(packing): #stripped back version for large packigns and stuff
	contacts=packing.getContacts()
	vecList=packing.getContactVectors().astype(float).tocsr()
	connectivity=[]
	for i in range(packing.getNumParticles()):
		neighbors=contacts[i].indices
		neighborsX=vecList[i,2*neighbors].todense()
		neighborsY=vecList[i,2*neighbors+1].todense()
		angList=np.arctan2(neighborsY,neighborsX)
		connectivity.append(neighbors[np.argsort(angList)].flatten())
	return np.array(connectivity,dtype=object)

#needs to run on radialDOFConstrainedMinimization
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
				p.load(f'{n}/seedPackings(radMin)/{name}-{packno}')
				p.setRadConstraintMoments(moments)
			except:
				p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=0, numParticles=n)
				p.setRadConstraintMoments(moments)
				p.setRandomPositions()
				p.setLogNormalRadii(polyDispersity='0.2')
				p.setPhi('.915')
			p.minimizeFIRE('1e-20')
			while(np.size(p.getContacts()) < 6*n):
				p.setPhi(p.getPhi()*1.01)
				p.minimizeFIRE('1e-20')
			print(np.size(p.getContacts()))
			p.save(f'{n}/seedPackings(radMin)/{name}-{packno}',overwrite=True)
			p.setLatticeVectors(np.array([[1,0],[0,1]],dtype=np.quad))
			data=simplePeriodicAngularSort(p)
			print(np.size(data))
			writeCPShortSimple(data, p.getNumParticles(),f'cpInputs',f'{name}-{packno}')
			print(packno)
	except:
		p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=1, numParticles=n)
		try:
			p.load(f'{n}/seedPackings(radMin)/{name}')
			p.setRadConstraintMoments(moments)
		except:
			p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=1, numParticles=n)
			p.setRadConstraintMoments(moments)
			p.setRandomPositions()
			p.setLogNormalRadii(polyDispersity='0.2')
			p.setPhi('.915')
		p.minimizeFIRE('1e-20')
		while(np.size(p.getContacts()) < 6*n):
			p.setPhi(p.getPhi()*1.01)
			p.minimizeFIRE('1e-20')
		print(np.size(p.getContacts()))
		p.save(f'{n}/seedPackings(radMin)/{name}',overwrite=True)
		data = simplePeriodicAngularSort(p)
		writeCPShortSimple(data, p.getNumParticles(),f'cpInputs',f'{name}')