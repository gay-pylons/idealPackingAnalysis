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
#pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import pyCudaPacking as pcp #configuring for radiiConstrainedDOFMinimizationSwitchable
import rigidpy as rp
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse
import random
#from cpReadyPacking import delaunayPeriodicAngularSort, writeCPShortSimple

#returns delaunay triangulation vectors
def delaunayVectors(packing,useRadical=True): #in future: unite delaunayPASort and delaunayVectors (single sweep)
	longVectors=packing.getContactVectors(gap=10).astype(float).tocsr() #.3 is rough metric: may need to increase for lower density
	rDelaunay=packing.delaunayNeighbors(radical=useRadical)
	delaunay=scipy.sparse.coo_matrix(rDelaunay)
	print(f'radical = {useRadical}')
	delVectors=scipy.sparse.csr_matrix((packing.getNumParticles(),2*packing.getNumParticles()), dtype=float)
	delVectors[delaunay.row,2*delaunay.col]=longVectors[delaunay.row,2*delaunay.col]
	delVectors[delaunay.row,2*delaunay.col+1]=longVectors[delaunay.row,2*delaunay.col+1]
	return rDelaunay,delVectors

def delaunayPeriodicAngularSort(packing,useRadical=True): #stripped back version for large packigns and stuff
	contacts,vecList=delaunayVectors(packing,useRadical=useRadical)
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
<text> process packing into circlepack; repack N=100000 </text>
<cmd iconname="amarok.png">act 0;Cleanse;Map 0 1;infile_read {name}.p;set_aim -d;alpha 10;newRed -t;repack 100000;layout -c;set_screen -a;Disp -w -C -Ra; </cmd>
<text> saves packing and lattice vectors </text>
<cmd>output :: Mob ::a b :: -f ~/Documents/code/idealPackingLibrary/{N}/latVecs/{name}-latVecs.dat;Write -cgrz ~/Documents/code/idealPackingLibrary/{N}/cpOutputs/{name}.p-dc</cmd>
  </CPscript>
<CPdata>
	<circlepacking name="{name}.p">
"""

	with open(f'{loc}/{name}-cpfile.xmd', "w") as f:
		f.write(blob)
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


def writeCPwithRadiiCenters(connectivity, N,loc,name,radii,centers):
	blob = f"""<?xml version="1.0"?>
	
<CP_Scriptfile date="Apr 17, 2022">
<CPscript title="manual CP save" >
<text> process packing into circlepack; repack N=100000 </text>
<cmd iconname="amarok.png">act 0;Cleanse;Map 0 1;infile_read {name}.p;set_aim -d;alpha 10;newRed -t;repack 100000;layout -c;set_screen -a;Disp -w -C -Ra; </cmd>
<text> saves packing and lattice vectors </text>
<cmd>output :: Mob ::a b :: -f ~/Documents/code/idealPackingLibrary/{N}/latVecs/{name}-latVecs.dat;Write -cgrz ~/Documents/code/idealPackingLibrary/{N}/cpOutputs/{name}.p-dc</cmd>
  </CPscript>
<CPdata>
	<circlepacking name="{name}.p">
"""

	with open(f'{loc}/{name}-cpfile.xmd', "w") as f:
		f.write(blob)
		f.write('NODECOUNT: {:d}\n'.format(N))
		f.write('FLOWERS:\n')
		# write the connectivity
		
		for i in range(len(connectivity)):
			f.write(str(i+1)+' ')
			f.write(str(len(connectivity[i]))+' ')
			f.write(' '.join(list(map(str,connectivity[i]+1)))+' ')
			f.write(str(connectivity[i][0]+1)+'\n')
		f.write('\n')
		f.write('RADII:\n')
		for i in range(len(radii)):
			f.write(f'{radii[i]:e}  ')
			if((i-3)%4 == 0):
				f.write('\n')
		f.write('\n')
		f.write('CENTERS:\n')
		for i in range(len(centers)):
			f.write(f'{centers[i][0]:e} {centers[i][1]:e}  ')
			if((i+1)%2 == 0):
				f.write('\n')
		f.write("""	</circlepacking>
  </CPdata>
</CP_Scriptfile>
		""")
		

def initializePackingHelper(N,name,packingPath,minimizationFlags, dist='.2',boxSize=np.array([1,1],dtype=np.quad)):
	print(f'{packingPath}/{name}')
	try:
		p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=0, numParticles=N)
		p.load(f'{packingPath}/{name}')
	except:
		p = pcp.Packing(nDim=2,potentialPower=np.quad('2'),deviceNumber=0, numParticles=N)
		if dist == 'mono':
			p.setMonoRadii()
		else:
			p.setLogNormalRadii(polyDispersity=np.quad('.2'))
		p.setBoxSize(boxSize)
		p.setRandomPositions()
		p.setLogNormalRadii(polyDispersity='0.2')
		p.setPhi(np.quad('.915'))
		if minimizationFlags != 'poisson':
			p.setDegreesOfFreedomType(minimizationFlags)
			if minimizationFlags != pcp.enums.degreesOfFreedomEnum.positions:
				p.setRadConstraintMoments([-2,2,4])
			p.minimizeFIRE('1e-20')
	p.save(f'{packingPath}/{name}',overwrite=True)
	return p


def initializePacking(N,name,packingPath,packingType='posRad',boxSize=np.array([1,1],dtype=np.quad)):
	if packingType== 'posRad':
		directory=f'{packingPath}/seedPackings(posRadMin)'
		p = initializePackingHelper(N,name,directory,pcp.enums.degreesOfFreedomEnum.positionsAndRadii,boxSize=boxSize)
		p.setRadConstraintMoments([-2,2,4])
	elif packingType== 'pos':
		directory=f'{packingPath}/seedPackings(posMin)'
		p = initializePackingHelper(N,name,directory,pcp.enums.degreesOfFreedomEnum.positions,boxSize=boxSize)
	elif packingType=='mono':
		directory=f'{packingPath}/monoDisperse'
		p = initializePackingHelper(N,name,directory,pcp.enums.degreesOfFreedomEnum.positions,dist='mono',boxSize=boxSize)
	else:
		directory=f'{packingPath}/poissonPoints'
		p = initializePackingHelper(N,name,directory,'poisson',dist='mono',boxSize=boxSize)
	return p

def writeFlowersToCP(n,packingPath,cpPath,name): #not currently in use anywhere
	data= np.loadtxt()
	writeCPShortSimple(data, p.getNumParticles(),cpPath,f'{name}')

def t1Transition(flowers,n1,n2): # execute random t1 transition on network
	if len(flowers[n1]) <= 3 or len(flowers[n2]) <= 3:
		print('will cause radius = 0')
		return flowers, 0
	if n2 in flowers[n1]:
		n2ind=np.nonzero(flowers[n1]==n2)[0][0]
		n1ind=np.nonzero(flowers[n2]==n1)[0][0]
		n3= flowers[n1][(n2ind+1)%len(flowers[n1])]
		n4= flowers[n1][(n2ind-1)%len(flowers[n1])]
		n4ind= np.nonzero(flowers[n3]==n2)[0][0]
		n3ind= np.nonzero(flowers[n4]==n1)[0][0]
		if n3==n2 or n4 == n2 or n3 == n1 or n4==n1:
			print('attempted self-interaction')
			return flowers, 0
		elif n3 in flowers[n4] or n4 in flowers[n3]:
			print('will cause radius = 0')
			return flowers, 0
		else:
			flowers[n3]=np.insert(flowers[n3],n4ind,n4)
			flowers[n4]=np.insert(flowers[n4],n3ind,n3)
			flowers[n1]=np.delete(flowers[n1], n2ind)
			flowers[n2]=np.delete(flowers[n2], n1ind)
			return flowers, 1
	else:
		print('chosen nodes not connected')
		return flowers, 0

def randomT1(flowers, nT1s):
	nSuccess=0
	while nSuccess < nT1s:
		weights=[len(array) for array in flowers]
		weights=weights/np.sum(weights)
		n1 = np.random.choice(range(len(flowers)),1,p=weights)[0]
		n2 = np.random.choice(flowers[n1],1)[0]
		flowers, success = t1Transition(flowers,n1,n2)
		nSuccess += success
	return flowers

def fiveMinRandomT1(flowers, nT1s):
	nSuccess=0
	while nSuccess < nT1s:
		weights=[]
		for array in flowers:
			if len(array) > 7:
				weights.append(100)
			elif len(array) > 5:
				weights.append(1)
			else:
				weights.append(0)
		weights=weights/np.sum(weights)
		n1 = np.random.choice(range(len(flowers)),1,p=weights)[0]
		weights2=[]
		for index in flowers[n1]:
			if len(flowers[index]) > 7:
				weights2.append(100)
			elif len(flowers[index]) > 5:
				weights2.append(1)
			else:
				weights2.append(0)
		weights2=weights2/np.sum(weights2)
		n2 = np.random.choice(flowers[n1],1,p=weights2)[0]
		flowers, success = t1Transition(flowers,n1,n2)
		nSuccess += success
	return flowers

def writePackingToCP(N,packingPath,cpPath,name,packingType='posRad',nT1=0,seedName=None,t1type=0,boxSize=np.array([1,1],dtype=np.quad)):
	if seedName != None:
		p=initializePacking(N,seedName,packingPath,packingType=packingType)
#		p.draw2DPacking()
#		plt.show()
#		plt.clf()
	else:
		p=initializePacking(N,name,packingPath,packingType=packingType,boxSize=boxSize)
	print(p.getNumParticles())
	if packingType == 'poisson' or packingType == 'mono':
		radical=False
	else:
		radical=True
	data = delaunayPeriodicAngularSort(p,useRadical=radical)
	if(nT1>0):
		data=randomT1(data, nT1)
	writeCPwithRadiiCenters(data, p.getNumParticles(),cpPath,f'{name}',p.getRadii().astype(float),p.getPositions().astype(float))

if __name__ == "__main__":
	print('gex')
	n=int(sys.argv[1]) #first command is number of particles
	name = str(sys.argv[2])
	try:
		packingType=str(sys.argv[3]) #delineates packing type:
	except:
		packingType='posRad'
#	if packingType is not pos,posRad or mono defaults to poisson
	try:
		phi=str(sys.argv[4])
	except:
		phi='.915'
	try:
		numPackings= int(sys.argv[5])
		lowerIndex= int(sys.argv[6])
	except:
		numPackings= 0
		lowerIndex= 0
	i=0
	try:
		nT1=int(sys.argv[7])
	except:
		nT1=0
	packingPath=f'../../idealPackingLibrary/{n}'
	cpPath=f'../../idealPackingLibrary/{n}/cpInputs'
	if numPackings>1:
		for packno in range(lowerIndex,numPackings):
			writePackingToCP(n,packingPath,cpPath,f'{name}-{packno}',packingType=packingType,nT1=nT1)
	else:
		writePackingToCP(n,packingPath,cpPath,f'{name}-{i}',packingType=packingType,nT1=nT1)

