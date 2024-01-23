import pyCudaPacking as pcp
import numpy as np
import scipy.linalg

def angleEnergy(r0,extRadii): #implementation of thurston's algorithm
	mIndex=len(extRadii)
	thetaSum=0
	for i in mIndex:
		c=r0+extRadii[i]
		b=r0+extRadii[(i+1)%mIndex]
		a=extRadii[i]+extRadii[(i+1)%mIndex]
		thetaSum+=np.arccos((c**2+b**2-a**2)/(2*b*c))
	return (2*np.pi-thetaSum)**2

def energySum(packing):
	contacts=packing.getContacts() #replace with delaunay neighbors!i
	radii=packing.getRadii()
	energy=0
	for i in range(packing.getNumParticles())
		r0=radii[i]
		extRadii=radii[contacts[i]] #this is a sketch: needs to be refinagled for data structure
		energy+=angleEnergy(r_0,extRadii)
	return energy

#minimize Angle Energy
