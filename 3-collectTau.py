from scipy.stats import gmean
import numpy as np
import os
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import npquad
import matplotlib
import plotColorList
from importlib.machinery import SourceFileLoader
pcp = SourceFileLoader("pyCudaPacking","/home/violalum/Documents/code/pcpMaster/pyCudaPacking/__init__.py").load_module()
#from scipy.stats import gmean 

def GofR(packing):
	N=packing.getNumParticles()
	distance=packing.getDistances().astype(float)
	distance=distance[distance!=0]
	Gee, Arr= np.histogram(distance,bins=N,density=True)
	Arr2= np.array([(Arr[i]+Arr[i+1])/2 for i in range(len(Arr)-1)])
	return Arr2,Gee

def hofR(packing):
	arr2,Gee=GofR(packing)
	return arr2, (Gee/(2*np.pi*arr2)-1)

def calcTau(hOfR, centers, N,cutoff=1/2):
	# Only compute out to delta = 1/2
	binSpacing = centers[1]-centers[0]
	integrand = hOfR**2 * centers * binSpacing
	# Is this the correct normalization???
	tau = np.sum(integrand[centers < cutoff]) * N
	return tau

def getTauR(directory):
	p=loadPack(directory)
	c,h=hofR(p)
	return calcTau(h,c,p.getNumParticles())

def getTauRCrystal(N):
	p = pcp.Packing(deviceNumber=0)
	p.set2DHexCrystal(np.array([np.sqrt(N*np.sqrt(3)/2),np.sqrt(N/np.sqrt(3)/2).round()],dtype=int))
	lv=np.array([[p.getBoxSize()[0],0],[0,p.getBoxSize()[1]]],dtype=np.quad)
	cutoff=np.min([p.getBoxSize()[0],p.getBoxSize()[1]])/2
	p.setLatticeVectors(lv)
	p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
	c,h=hofR(p)
	return p.getNumParticles(), calcTau(h,c,p.getNumParticles(),cutoff=cutoff)

def loadPack(directory,deviceNumber=0): #Loads packing from directory into p and attempts to set lattice vectors
	p=pcp.Packing(deviceNumber=0)
	p.load(directory)
	try:
		lv = np.loadtxt(f"{directory}/latticeVectors.dat")
	except:
		lv=np.array([[1,0],[0,1]])
	lv1Norm=lv[0]/np.sqrt(np.dot(lv[0],lv[0]))
	rotationMatrix= np.array([[lv1Norm[0],lv1Norm[1]],[-lv1Norm[1],lv1Norm[0]]])
	newVec=np.matmul(rotationMatrix.transpose(),lv)
	p.setLatticeVectors(newVec)
	p.setPositions(np.dot(p.getPositions(),rotationMatrix))
	p.setGeometryType(pcp.enums.geometryEnum.latticeVectors)
	return p

def getTauK(directory):
	p=loadPack(directory)
	k=1/np.sqrt(p.getPhi()/ (p.getNumParticles()*np.pi))
	try:
		sFactors=pcp.fileIO.load2DArray(f'{directory}/unBinnedsOfK.dat',np.quad)
	except:
		sFactors=np.array(p.getStructureFactor(k))
		pcp.fileIO.save2DArray(f'{directory}/unBinnedsOfK.dat',sFactors)
	Sm1=sFactors-1
	tau=(np.dot(Sm1,Sm1))*(k/(2*np.pi))**2/(p.getNumParticles())**2
	print(tau)
	return tau


if __name__ == '__main__':
	plt.figure(figsize=(4,3.5))
	nArray=np.array([64,128,256,512,1024,2048,4096])#,8192])
	posCPColor=plotColorList.posTriangColor
	posRadCPColor=plotColorList.posRadTriangColor
	posColor=plotColorList.posColor
	posRadColor=plotColorList.posRadColor
	posTau,posRadTau,posTriangTau,posRadTriangTau,crysTau=[],[],[],[],[]
	posTauStd,posRadTauStd,posTriangTauStd,posRadTriangTauStd,crysTauStd=[],[],[],[],[]
	crysN=[]
	
	for n in nArray:
		directory=f'../idealPackingLibrary/{n}/jumbledPackings/idealPack{n}'
		data=np.array([getTauR(f'{directory}-{i}/isostatic') for i in range(10)])
		posTau.append(np.mean(data))
		posTauStd.append(np.std(data))
		directory=f'../idealPackingLibrary/{n}/finishedPackings/idealPack{n}'
		data=np.array([getTauR(f'{directory}-{i}') for i in range(10)])
		posRadTriangTau.append(np.mean(data))
		posRadTriangTauStd.append(np.std(data))
		cN,cTau=getTauRCrystal(n)
		crysN.append(cN)
		crysTau.append(cTau)
	crysN=np.array(crysN,dtype=float)
	posTau,posRadTau,posTriangTau,posRadTriangTau,crysTau=np.array(posTau),np.array(posRadTau),np.array(posTriangTau),np.array(posRadTriangTau),np.array(crysTau)
	posTauStd,posRadTauStd,posTriangTauStd,posRadTriangTauStd,crysTauStd=np.array(posTauStd),np.array(posRadTauStd),np.array(posTriangTauStd),np.array(posRadTriangTauStd),np.array(crysTauStd)
	plt.errorbar(nArray,posTau,yerr=posTauStd,linestyle='--',marker='x',color=posColor,capsize=4)
	plt.errorbar(nArray,posRadTriangTau,yerr=posRadTriangTauStd,linestyle='-',marker='x',color=posRadCPColor,capsize=4,alpha=.8)
	plt.loglog(crysN,crysTau,'-o',color=[.3,.3,.3])
	plt.ylim([.1,8e+3])
	plt.xlabel('$N$',size=22)
	plt.ylabel("$\\tau$",size=22)
	plt.tick_params(axis='x',which='major',direction='inout',length=14,labelsize='xx-large')
	plt.tick_params(axis='x',which='minor',direction='inout',length=10)
	plt.tick_params(axis='y',which='major',direction='in',length=10,labelsize='xx-large')
	plt.tick_params(axis='y',which='minor',direction='in',length=5)
	plt.tight_layout()
	plt.text(64, 200,'hex crystal',fontsize='xx-large',rotation=25)
	plt.text(64, 1,'pos. minimized',fontsize='x-large',color=posColor,rotation=0)
#	plt.text(256, 2,'pos. minimized & triangulated',fontsize='large',color=posCPColor,rotation=5)
#	plt.text(64,4.5e-1,'pos.+rad. minimized',fontsize='large',color=posRadColor,rotation=355,va='top')
	plt.text(64,4e-1, 'triangulated',fontsize='x-large',color=posRadCPColor,va='top')
	plt.gcf().text(.31,.87,'D',color='black',fontsize=23)#,bbox= dict(boxstyle='square',alpha=.7,facecolor=[.85,.85,.85],linewidth=0))
	plt.savefig('../idealPackingLibrary/figures/tauVsN.png')
	plt.savefig('../idealPackingLibrary/figures/tauVsN.pdf')
# =============================================================================
# 	plt.yscale('linear')
# 	plt.ylim([0,1.8])
# 	plt.savefig('../idealPackingLibrary/figures/tauVsNLogLin.png')
# 	plt.savefig('../idealPackingLibrary/figures/tauVsNLogLin.pdf')
# =============================================================================
	plt.show()
