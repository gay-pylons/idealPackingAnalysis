import sys
import numpy as np
import npquad
import imp
pcp=imp.load_source('pyCudaPacking','/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py')
import matplotlib.pyplot as plt
import scipy.sparse
import cpPoissonPoints as cp

def flowerAngleEnergy(radii,i,contacts): #implementation of thurston's algorithm
	neighbors=contacts[i]
	r0=radii[i]
	extRadii1=radii[neighbors]
	extRadii2=np.roll(extRadii1,1)
	c=r0+extRadii1
	b=r0+extRadii2
	a=extRadii1+extRadii2
	thetaSum=np.sum(np.arccos((c**2+b**2-a**2)/(2*b*c)))
	return (2*np.pi-thetaSum)**2

def epsilon(radii,i,contacts): #implementation of thurston's algorithm
	neighbors=contacts[i]
	r0=radii[i]
	extRadii1=radii[neighbors]
	extRadii2=np.roll(extRadii1,1)
	c=r0+extRadii1
	b=r0+extRadii2
	a=extRadii1+extRadii2
	thetaSum=np.sum(np.arccos((c**2+b**2-a**2)/(2*b*c)))
	return 2*np.pi-thetaSum

def diEpsiloni(radii,i,contacts): #implementation of thurston's algorithm
	neighbors=contacts[i]
	ri=radii[i]
	extRadii1=radii[neighbors]
	extRadii2=np.roll(extRadii1,1)
	tri=ri*(ri+extRadii1+extRadii2)
	dEps=np.sqrt(extRadii1*extRadii2/tri)*(2*ri+extRadii1+extRadii2)/(tri+extRadii1*extRadii2)
	return np.sum(dEps)

def diEpsilonj(radii,i,j,contacts):
	neighbors=np.intersect1d(contacts[i],contacts[j])
	ri=radii[i]
	rj=radii[j]
	rk=radii[neighbors]
	trii=ri*(ri+rj+rk)
	trij=rj*(ri+rj+rk)
	dEps=np.sqrt(rj*rk/trii)*(2*ri+rj+rk)/(trij+ri*rk)
	return -np.sum(dEps)

def angleEnergy(radii,contacts):
	energy=0
	for i in range(len(radii)):
		energy=flowerAngleEnergy(radii,i,contacts)
	return energy

def angleEnergyGradient(radii,contacts):
	dU=np.array([0. for i in range(len(radii))])
	for i in range(len(radii)):
		dU[i]=epsilon(radii,i,contacts)*diEpsiloni(radii,i,contacts)
		for j in contacts[i]:
			dU[i]+=epsilon(radii,j,contacts)*diEpsilonj(radii,i,j,contacts)
	return dU

def angleSD(radii,contacts,scale):
	return radii-scale*angleEnergyGradient(radii,contacts)

def triangLayout(radii,contacts):
	positions = np.empty((len(radii,2)))
	packed=np.ones(128,dtype=bool)
	positions[0][0]=0
	positions[0][1]=0 #sets particle 0 at origin
	outerRow=contacts[0] #sets first flower
	positions[outerRow[0]][0]=radii[0]+radii[outerRow[0]] #sets first neighbor of particle at origin on x axis
	packed[outerRow[0]]=True
	while(np.sum(packed)>0)

if __name__ == '__main__':
	p = pcp.Packing()
	p.load('../idealPackingLibrary/128/seedPackings(radMin)/idealPack128-1')
	p.setLatticeVectors(np.array([[1,0],[0,1]],dtype=np.quad))
	p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
	radii=np.ones(p.getNumParticles())
	contacts=cp.delaunayPeriodicAngularSort(p)
#	radiip1=p.getRadii().astype(float)
	print(angleEnergy(radii,contacts))
	print(np.zeros(128,dtype=bool))
# =============================================================================
# 	for k in range(40):
# 		radii=angleSD(radii,contacts,.01)
# 		print(f'angergy={angleEnergy(radii,contacts)}')
# 	p.setRadii(radii.astype(np.quad))
# 	p.minimizeFIRE('1e-20')
# 	p.draw2DPacking()
# 	plt.show()
# =============================================================================
