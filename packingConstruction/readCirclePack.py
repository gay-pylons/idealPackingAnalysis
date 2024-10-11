#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:51:10 2022
present due to helper functions
CP Test assistance file
@author: violalum
"""
import sys
import pyCudaPacking as pcp
import matplotlib.pyplot as plt
import numpy as np
import npquad
#import trm
from matplotlib import pyplot as plt
import rigidpy as rp
import networkx as nx

# Varda's code
# copied directly from CirclePack notebook

#define color maps used in CirclePack?
def color_map_color(value, cmap_name='Wistia', vmin=0, vmax=1):
	# norm = plt.Normalize(vmin, vmax)
	norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
	cmap = cm.get_cmap(cmap_name)  # PiYG
	rgb = cmap(norm(abs(value))) # will return rgba, we take only first 3 so we get rgb
	colors = mpl.colors.rgb2hex(rgb) #rgb[:,0:3]
	return colors

def getColorListRadii(myList,mapping):
		colorList = np.zeros((myList.shape[0],3))
		listRange = np.linspace(np.mean(myList)-3*np.sqrt(np.var(myList)), np.mean(myList)+8*np.sqrt(np.var(myList)), 256)
		colorRange = np.zeros((256, 3))
		#colorRange[:,0] = np.ones(256)*100/256 #red
		#colorRange[:,1] = np.linspace(1,255,256)/255 #green
		#colorRange[:,1] = np.flip(np.linspace(0,200,256)/255) #green
		#colorRange[:,2] = np.flip(np.linspace(0,250,256)/255) #blue
		colorRange = mapping(np.linspace(0, 2, 256))[:,:3]
		for i in range(myList.shape[0]):
			for j in range(1,256):
				if(myList[i]>listRange[j-1] and myList[i]<listRange[j]):
					colorList[i] = colorRange[j]
		return colorList

def lenref(a1,a2):
	index = np.array([[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]],dtype=np.longdouble)
	transvec = np.outer(index[:,0],a1)+ np.outer(index[:,1],a2)
	return transvec

def readFromCirclePack(filepath):
	d = {'CHECKCOUNT':None, 'GEOMETRY':None, 'RADII':[], 'CENTERS':[]}
	with open(filepath) as f:
		lines = f.readlines()
		d['CHECKCOUNT'] = lines[0].split()[1]
		d['GEOMETRY'] = lines[1].split()[1]
		copy_radius = False
		copy_centers = False
		for i in range(2,len(lines)):
			if lines[i].startswith('RADII:'):
				copy_radius = True
			elif lines[i].startswith('CENTERS:'):
				copy_radius = False
				copy_centers = True
			elif lines[i].startswith('END'):
				copy_centers = False
			elif copy_radius:
				d['RADII'].extend([float(item) for item in lines[i].split()])
			elif copy_centers:
				d['CENTERS'].extend([float(item) for item in lines[i].split()])
	return d

def unit_vector(vector):
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#defunct; primarily holdover from init. version
def writeToCirclePack(connectivity, N):
	blob = """<?xml version="1.0"?>
	
<CP_Scriptfile date="Apr 17, 2022">
<CPscript title="How to lay out a patch of nine cells" >
<text> This script shows how to read the "init_torus.p" data file (internal to this script) and to circle pack and display it. </text>
<cmd iconname="amarok.png">act 0;Cleanse;Map 0 1;infile_read init_torus.p;set_aim -d;alpha 10;newRed -t;repack;layout -c;set_screen -a;Disp -w -C -Ra; </cmd>
<text> The circle packing is unique up to euclidean scaling, side-pairings, and normalization. The "newRed -t" command in the first set of commands insured that the layout has just two side-pairing maps rather than the generic three. (You can remove that command to see the effect (which barely shows up in this particular case). The position of the patch is rather arbitrary, determined by the layout order.

The next command repeats this in p1 and scales it in preparation for layout out some copies in the universal cover. </text>
<cmd iconname="delete.png">act 1;infile_read init_torus.p;set_aim -d;alpha 10;newRed -t;repack;layout -c;set_screen -a;scale .4;Disp -w -C -Ra; </cmd>
<text> The next command applies side-pairing Mobius transformation to get the eight surrounding copies in the covering group. Note that you can start over and carry these or any combination our by putting the mouse in the canvas and pressing one of the letters. </text>
<cmd iconname="kivio.png">[a];[b];[A];[A];[B];[B];[a];[a]; </cmd>
<text> The next command shows two of the side-pairing Mobius transformation, labeled 'a' and 'b'. in the "Messages" window. Each involves four complex numbers a, b, c, d, and the associated Mobius is (az+b)/(cz+d) (for this situation, c=0 and d=1). These two Mobius generate the lattice associated with this torus. The command also displays the conformal modulus tau. </text>
<cmd iconname="xeyes.png">?mob a;?mob b;torus_tau; </cmd>
<text> This next command saves a matrix readable by matlab; each row gives the end vertex numbers and length of one of the edges of the layout (though matlab reads the integers as doubles). Note that these lengths are uniquely determined up to a scale factor by the initial combinatorics of the torus. </text>
<cmd iconname="psi.png">act 0;output edgelengths =[\n :: EI "  "EL "\n" :: a :: ] -f  patch_edgelengths.m </cmd>
<cmd inline="no" name="a" mnemonic="a" iconname="amarok.png">pair_mob a;[d] </cmd>
<cmd inline="no" name="b" mnemonic="b" iconname="apollon.png">pair_mob b;[d] </cmd>
<cmd inline="no" name="c" mnemonic="c" iconname="userconfig.png">pair_mob c;[d] </cmd>
<cmd inline="no" name="A" mnemonic="A" iconname="metacontact_online.png">pair_mob A;[d] </cmd>
<cmd inline="no" name="B" mnemonic="B" iconname="icq_dnd.png">pair_mob B;[d] </cmd>
<cmd inline="no" name="C" mnemonic="C" iconname="format_increaseindent.png">pair_mob C;[d] </cmd>
<cmd inline="no" name="d" mnemonic="d" iconname="psi.png">mark -cw -c a(1 1024);disp -c m -R a </cmd>
  </CPscript>
<CPdata>
	<circlepacking name="init_torus.p">\n"""

	with open(f'./{N}/{name}-cpfile.xmd', "w") as f:
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


# altered slightly for my purposes but fundamentally the same
# set v1 to be mobius b parameters

def readVardaOutput(filename, n,v1): #add more descriptive name soon
	a1 = np.array([ 1, 0])
	a2 = np.array([ 0, 1]) #tau_imaginary])
	data = readFromCirclePack(filename)
	cents = np.array(data['CENTERS'])
	rad = np.array(data['RADII'])
	coordinates = cents.reshape(-1,2)[0:n]
	radii = rad.reshape(-1,1)[0:n]
	sFactor=np.sqrt(np.linalg.det(v1).astype(np.quad))
	radii = np.ndarray.flatten(radii).astype(np.quad)/sFactor
	sites = coordinates.astype(np.quad)/sFactor
	return [radii, sites]

# Viola's code
# enters radii and positions from output of varda's code; may have to check for rotations
def inputVardaOutput(packing,vO):
	packing.setRadii(vO[0].astype(np.quad))
	packing.setPositions(vO[1].astype(np.quad))

def readCPOutput(pA,filename,nParticles,latVectors):
	cpInput = readVardaOutput(filename,nParticles,latVectors)
	pA.setLatticeVectors(latVectors)
	#set thingy to lattice vektor: 2 element array in 2d
	inputVardaOutput(pA,cpInput)

def colourize(packing): #returns color map according to number of contacts
	k=np.zeros_like([],dtype=float,shape=[len(packing.getRadii()),3])
	for i in range(len(k)):
		if np.size(packing.getContacts()[i])==7:
			k[i] = [0.6,0.4,0.9]
		elif np.size(packing.getContacts()[i])>7:
			k[i] = [0.90,0.3,0.5]
		elif np.size(packing.getContacts()[i])==5:
			k[i] = [0,0.5,1.0]
		elif np.size(packing.getContacts()[i])<5:
			k[i] = [0.3,1.0,0.3]
		else:
			k[i] = [0.5,0.5,0.5]
	return k

def sortIndicesByRadii(packing):
	argList=np.argsort(packing.getRadii())
	packing.setRadii(packing.getRadii()[argList])
	packing.setPositions(packing.getPositions()[argList])

def readLatVecs(lvPath):
	file= open(lvPath,'r')
	lines=file.readlines()
	b1=np.array(lines[1].split('= ')[1].split('i')[0].split(' + '))
	b2=np.array(lines[6].split('= ')[1].split('i')[0].split(' + '))
	print(b1)
	print(b2)
	lv=np.transpose(np.array([[np.quad(b1[0]),np.quad(b1[1])],[np.quad(b2[0]),np.quad(b2[1])]]))
	if lv[0,0] < 0: # check to insure positive lattice vectors: only matters for 
		return -lv
	else:
		return lv

def loadCirclePackIntoPcp(directory):
	latVex=readLatVecs(f'../../idealPackingLibrary/{n}/latVecs/{directory}-latVecs.dat').astype(np.quad)
	p=pcp.Packing(nDim=2,numParticles=n,potentialPower=np.quad('2'),deviceNumber=1)
	p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
	p.setLatticeVectors(latVex)
	readCPOutput(p,f'../../idealPackingLibrary/{n}/cpOutputs/{directory}.p-dc',n,latVex)
	print(f'phi={p.getPhi()}')
	print(f'z={np.size(p.getContacts())/n}')
	p.save(f"../../idealPackingLibrary/{n}/finishedPackings/{directory}",overwrite=True)
	pcp.fileIO.save2DArray(f"../../idealPackingLibrary/{n}/finishedPackings/{directory}/latticeVectors.dat",p.getLatticeVectors())

# In[read CirclePack]:
if __name__ == '__main__':
	n=int(sys.argv[1])
	packing=str(sys.argv[2])
	try:
		batchMax=int(sys.argv[3])
	except:
		batchMax=-1
	if(batchMax>=1):
		for packno in range(batchMax):
			packN = f"{packing}-{packno}"
			loadCirclePackIntoPcp(packN)
	else:
		loadCirclePackIntoPcp(packing)