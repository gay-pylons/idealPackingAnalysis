'''
Created by Francesco
16 July 2019
'''

import numpy as np
import matplotlib.pyplot as plt
from mpltools import annotation
import sys

nDim = int(sys.argv[1])
#numParticles = int(sys.argv[2])
#dirName = sys.argv[3]
#prTh = float(sys.argv[4])
'''
fromIso = 1
if(numParticles == "8192"):
    if(nDim == 3):
        packingList = np.array([0.6218438, 0.64460665, 0.65146494, 0.65655583, 0.65966034, 0.6608476, 0.66136605, 0.66143376, 0.66145015])
        phiList = packingList
        phiJ = 0.6614521
    elif(nDim == 2):
        packingList = np.array([0.8376547, 0.84389955, 0.84554935, 0.84766954, 0.84961224, 0.84972686, 0.84980655, 0.84984285, 0.84986544])
        phiList = packingList
        phiJ = 0.849865452
elif(numParticles == "4096"):
    packingList = np.array([0.61199707, 0.6344846, 0.64614004, 0.65292734, 0.65648913, 0.6598868, 0.66004986, 0.66143316, 0.66174644, 0.66182905, 0.6624586859])
    phiList = packingList
    phiJ = 0.66246611921
elif(numParticles == "fromIso"):
    packingList = np.array(["0", "deltaPhi1e-06", "deltaPhi3e-06", "deltaPhi5e-06", "deltaPhi1e-05", "deltaPhi1e-04", "deltaPhi5e-04", "deltaPhi1e-03"])
    phiList = np.array([0.645410391, 0.645409101, 0.6454071645, 0.6454052283, 0.645397483, 0.64528131, 0.644894063, 0.644119571])
    phiJ = 0.6454052283
elif(numParticles == 2048):
    packingList = np.array([0.61539143, 0.62512285, 0.64586645, 0.65432274, 0.65955183, 0.66076213, 0.66084605, 0.66088074, 0.66089207])
    phiList = packingList
    phiJ = 0.660895407
elif(numParticles == 1024):
    packingList = np.array([0.6363179, 0.65836704, 0.65980047, 0.66013575, 0.66140217, 0.66172165, 0.66185397, 0.661953726757])
    phiList = packingList
    phiJ = 0.661963276124
#saveData = np.zeros((len(packingList),2))
'''
fig, ax = plt.subplots(1, dpi=150)
x = np.geomspace(2e-06, 2e-03)
plt.plot(x, x**0.5, linewidth=1.5, color='k')
ax.set_ylabel("$\\omega^*$", fontsize=12)
ax.set_xlabel("$\Delta \\varphi$", fontsize=12)
ax.tick_params(axis='both', labelsize=12)
prThList = np.array([0.2, 0.18, 0.15, 0.12, 0.1, 0.08])
colorList = np.array(["#228B22", "#FFFF00", "#FFA500", "r", "#FF00FF", "b"])
pathName = "/home/farceri/Documents/Data/logCompression/"
sampleList = np.array(["1024", "2048", "4096", "fromIso", "8192"])
for sample in sampleList:
    if(sample == "1024"):
        numParticles = 1024
        dirName = "1024/3D/"
        packingList = np.array([0.6363179, 0.65836704, 0.65980047, 0.66013575, 0.66140217, 0.66172165, 0.66185397, 0.661953726757])
        phiList = packingList
        phiJ = 0.661963276124
        markerShape = "o"
    elif(sample == "2048"):
        numParticles = 2048
        dirName = "2048/3D/"
        packingList = np.array([0.61539143, 0.62512285, 0.64586645, 0.65432274, 0.65955183, 0.66076213, 0.66084605, 0.66088074, 0.66089207])
        phiList = packingList
        phiJ = 0.660895407
        markerShape = "^"
    elif(sample == "4096"):
        numParticles = 4096
        dirName = "4096/3D/"
        packingList = np.array([0.61199707, 0.6344846, 0.64614004, 0.65292734, 0.65648913, 0.6598868, 0.66004986, 0.66143316, 0.66174644, 0.66182905, 0.6624586859])
        phiList = packingList
        phiJ = 0.66246611921
        markerShape = "v"
    elif(sample == "fromIso"):
        numParticles = 4096
        dirName = "4096/3D/fromIso/"
        packingList = np.array(["0", "deltaPhi1e-06", "deltaPhi3e-06", "deltaPhi5e-06", "deltaPhi1e-05", "deltaPhi1e-04", "deltaPhi5e-04", "deltaPhi1e-03"])
        phiList = np.array([0.645410391, 0.645409101, 0.6454071645, 0.6454052283, 0.645397483, 0.64528131, 0.644894063, 0.644119571])
        phiJ = 0.6454052283
        markerShape = "s"
    elif(sample == "8192"):
        numParticles = 8192
        dirName = "8192/3D/compression1e-06/"
        packingList = np.array([0.6218438, 0.64460665, 0.65146494, 0.65655583, 0.65966034, 0.6608476, 0.66136605, 0.66143376, 0.66145015])
        phiList = packingList
        phiJ = 0.6614521
        markerShape = "D"
    omegastar = np.zeros((len(packingList), 6))
    for n in range(len(packingList)):
        dirPacking = str(packingList[n])
        omegas = np.loadtxt(pathName + dirName + str(packingList[n]) + "/omegas.dat")
        pr = 1/(np.loadtxt(pathName + dirName + str(packingList[n]) + "/ipr.dat")*numParticles)
        pr = pr[omegas<1]
        omegas = omegas[omegas<1]
        for j in range(6):
            for i in range(len(pr)):
                if(pr[-i] < prThList[j]):
                    omegastar[n,j] = omegas[-i]
            #if(omegastar[n,j] == 0 and (phiJ - phiList[n]) < 5e-06):
            #    omegastar[n,j] = np.min(omegas)

            #print("omegastar:", omegastar[n,j], "deltaPhi:", phiJ - phiList[n])
            #saveData[n,0] = phiJ - phiList[n]
            #saveData[n,1] = omegastar[n]

    #np.savetxt(dirName + "omegastar.dat", saveData)
    for j in range(6):
        ax.loglog(phiJ - phiList, omegastar[:,j], linewidth = 0, marker=markerShape, markersize=10, markerfacecolor=colorList[j], markeredgecolor="k", markeredgewidth=1, alpha = 0.8)

#plt.legend(["$\\omega \\propto \\Delta\\varphi^{1/2}$", "$\\mathrm{PR_c}$ = 0.2", "$\\mathrm{PR_c}$ = 0.18", "$\\mathrm{PR_c}$ = 0.15", "$\\mathrm{PR_c}$ = 0.12", "$\\mathrm{PR_c}$ = 0.1", "$\\mathrm{PR_c}$ = 0.08"], loc="lower right", fontsize=12)
#plt.title("$N=$" + str(numParticles), fontsize=15)
fig.tight_layout()
plt.savefig("/home/farceri/Pictures/paper/prCutoff.pdf", transparent=True, format = "pdf", pad_inches=0.05)
plt.show()
