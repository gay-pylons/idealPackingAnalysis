'''
Created by Francesco
19 June 2019
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from mpltools import annotation
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
import sys
import os
from os import path
import pyCudaPacking as pcp

def getPDF(data, bins):
    pdf, edges = np.histogram(data, bins, density=True)
    edges = (edges[:-1] + edges[1:])/2
    return pdf, edges

def getGaps(dirName, dirData):
    p = pcp.Packing()
    gap = []
    for indexPacking in os.listdir(dirName + os.sep + dirData):
        if(path.isdir(dirName + os.sep + dirData + os.sep + indexPacking)):
            p.load(dirName + os.sep + dirData + os.sep + indexPacking)
            p.calcNeighborsCut(np.quad("0.8"))
            gap.append(p.getNeighborGaps().data)
    gap = np.array(gap)
    gap = np.concatenate(gap.ravel())
    gap /= np.mean(gap)
    return gap.astype(np.float64)

def plotGapPDF(dirName):
    gap7 = getGaps(dirName, "DeltaPhi1.1e-07")
    gap6 = getGaps(dirName, "DeltaPhi1.3e-06")
    gap5 = getGaps(dirName, "DeltaPhi1.5e-05")
    gap5hard = getGaps(dirName, "DeltaPhi1e-05")
    gap4 = getGaps(dirName, "DeltaPhi2e-04")
    gap3 = getGaps(dirName, "DeltaPhi3e-03")
    gap2 = getGaps(dirName, "DeltaPhi2e-02")
    bins = np.geomspace(2e-08, 4.99999e-07,15)
    pdf7, edges7 = getPDF(gap7, np.concatenate((bins, np.geomspace(5e-07, np.max(gap7), 50))))
    bins = np.geomspace(2e-07, 2.99999e-06,15)
    pdf6, edges6 = getPDF(gap6, np.concatenate((bins, np.geomspace(3e-06, np.max(gap6), 50))))
    bins = np.geomspace(2e-06, 3.99999e-05,15)
    pdf5, edges5 = getPDF(gap5, np.concatenate((bins, np.geomspace(4e-05, np.max(gap5), 40))))
    bins = np.geomspace(1e-06, 3.99999e-04,15)
    pdf5hard, edges5hard = getPDF(gap5hard, np.concatenate((bins, np.geomspace(4e-04, np.max(gap5hard), 40))))
    bins = np.geomspace(1e-05, 0.99999e-03,15)
    pdf4, edges4 = getPDF(gap4, np.concatenate((bins, np.geomspace(1e-03, np.max(gap4), 30))))
    bins = np.geomspace(1e-04, 1.99999e-02,15)
    pdf3, edges3 = getPDF(gap3, np.concatenate((bins, np.geomspace(2e-02, np.max(gap3), 25))))
    bins = np.geomspace(1e-03, 0.99999e-01,15)
    pdf2, edges2 = getPDF(gap2, np.concatenate((bins, np.geomspace(1e-01, np.max(gap2), 15))))
    #gap plot "#1E90FF", "#4682B4", "#3CB371", "#3CB371", "#32CD32", "#228B22"
    fig, ax = plt.subplots(1, dpi = 200)
    ax.loglog(edges7, pdf7, "s", markersize=12, markerfacecolor="#191970", markeredgecolor="k", markeredgewidth = 0.5, alpha = 1)
    ax.loglog(edges7[pdf7>0], pdf7[pdf7>0], "k", linewidth=0.6)
    ax.loglog(edges6, pdf6, "s", markersize=12, markerfacecolor="#4169E1", markeredgecolor="k", markeredgewidth = 0.5, alpha = 1)
    ax.loglog(edges6[pdf6>0], pdf6[pdf6>0], "k", linewidth=0.6)
    ax.loglog(edges5hard, pdf5hard, "D", markersize=12, markerfacecolor=[0,100/255,0], markeredgecolor="k", markeredgewidth = 0.5, alpha = 1)#006400
    ax.loglog(edges5hard[pdf5hard>0], pdf5hard[pdf5hard>0], "k", linewidth=0.6)
    ax.loglog(edges5, pdf5, "s", markersize=12, markerfacecolor="#00BFFF", markeredgecolor="k", markeredgewidth = 0.5, alpha = 1)
    ax.loglog(edges5[pdf5>0], pdf5[pdf5>0], "k", linewidth=0.6)
    ax.loglog(edges4, pdf4, "D", markersize=12, markerfacecolor=[40/255,150/255,40/255], markeredgecolor="k", markeredgewidth = 0.5, alpha = 1)#228B22
    ax.loglog(edges4[pdf4>0], pdf4[pdf4>0], "k", linewidth=0.6)
    ax.loglog(edges3, pdf3, "D", markersize=12, markerfacecolor=[80/255,200/255,80/255], markeredgecolor="k", markeredgewidth = 0.5, alpha = 1)#32CD32
    ax.loglog(edges3[pdf3>0], pdf3[pdf3>0], "k", linewidth=0.6)
    ax.loglog(edges2, pdf2, "D", markersize=12, markerfacecolor=[120/255,250/255,120/255], markeredgecolor="k", markeredgewidth = 0.5, alpha = 1)#00FF00
    ax.loglog(edges2[pdf2>0], pdf2[pdf2>0], "k", linewidth=0.6)
    x = np.geomspace(5e-07,0.5)
    ax.plot(x, 0.04*x**-0.41296, linewidth=1.5, color='k')
    #ax.set_ylim(5e-04, 1e8)
    #ax.set_xlim(3e-08, 99)
    ax.set_ylabel("PDF($h$)", fontsize=15)
    ax.set_xlabel("$h$", fontsize=15)
    ax.set_xticks([1e-08, 1e-06, 1e-04, 1e-02, 1])
    #ax.set_xticks(["10^{-8}", "10^{-6}", "10^{-4}", "10^{-2}", "10^{0}"])
    ax.tick_params(axis='both', labelsize=15)
    fig.tight_layout()
    #print(plt.rcParams)
    plt.savefig("/home/farceri/Pictures/paper/gap_distribution_new.pdf", transparent=True, format = "pdf", pad_inches=0.05)
    plt.show()

def plotModesPDF(dirName):
    omega7 = np.loadtxt(dirName + os.sep + "DeltaPhi1.1e-07" + os.sep + "omegasSamples.dat")
    pr7 = 1/(np.loadtxt(dirName + os.sep + "DeltaPhi1.1e-07" + os.sep + "iprSamples.dat")*4096)
    omega7 /= np.mean(omega7)
    omega7 = omega7[pr7>0.12]
    omega6 = np.loadtxt(dirName + os.sep + "DeltaPhi1.3e-06" + os.sep + "omegasSamples.dat")
    pr6 = 1/(np.loadtxt(dirName + os.sep + "DeltaPhi1.3e-06" + os.sep + "iprSamples.dat")*4096)
    omega6 /= np.mean(omega6)
    omega6 = omega6[pr6>0.12]
    omega5 = np.loadtxt(dirName + os.sep + "DeltaPhi1.5e-05" + os.sep + "omegasSamples.dat")
    pr5 = 1/(np.loadtxt(dirName + os.sep + "DeltaPhi1.5e-05" + os.sep + "iprSamples.dat")*4096)
    omega5 /= np.mean(omega5)
    omega5 = omega5[pr5>0.12]
    bins7 = np.geomspace(1e-04,1e-02,30)
    bins7 = np.concatenate((bins7, np.geomspace(1e-02,1.5,80)))
    bins7 = np.concatenate((bins7, np.linspace(1.5,3,100)))
    pdf7, edges7 = getPDF(omega7, bins7)
    bins6 = np.geomspace(1e-03,2e-02,50)
    bins6 = np.concatenate((bins6, np.geomspace(2e-02,1.5,80)))
    bins6 = np.concatenate((bins6, np.linspace(1.5,3,100)))
    pdf6, edges6 = getPDF(omega6, bins6)
    bins5 = np.geomspace(5e-03,3e-02,50)
    bins5 = np.concatenate((bins5, np.geomspace(3e-02,1.5,80)))
    bins5 = np.concatenate((bins5, np.linspace(1.5,3,100)))
    pdf5, edges5 = getPDF(omega5, bins5)
    #modes plot
    fig, ax = plt.subplots(1, dpi = 200)
    ax.loglog(edges7[pdf7>0], pdf7[pdf7>0], "o", markersize=7, markerfacecolor="#0000CD", markeredgecolor="k", alpha = 0.7)
    ax.loglog(edges7[pdf7>0], pdf7[pdf7>0], "k", linewidth=0.6)
    ax.loglog(edges6[pdf6>0], pdf6[pdf6>0], "s", markersize=7, markerfacecolor="#4169E1", markeredgecolor="k", alpha = 0.7)
    ax.loglog(edges6[pdf6>0], pdf6[pdf6>0], "k", linewidth=0.6)
    ax.loglog(edges5[pdf5>0], pdf5[pdf5>0], "^", markersize=7, markerfacecolor="#00BFFF", markeredgecolor="k", alpha = 0.7)
    ax.loglog(edges5[pdf5>0], pdf5[pdf5>0], "k", linewidth=0.6)
    ax.set_ylabel("D($\\omega/\\langle\\omega\\rangle$)", fontsize=15)
    ax.set_xlabel("$\\omega/\\langle\\omega\\rangle$", fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_xlim(5e-05, 3.5)
    ax.set_ylim(2e-03, 1.2)
    fig.tight_layout()
    plt.show()

def comparePDF(dirName):
    omega7 = np.loadtxt(dirName + os.sep + "DeltaPhi1.1e-07" + os.sep + "omegasSamples.dat")
    omega7 /= np.mean(omega7)
    omega = np.loadtxt(dirName + os.sep + "harmonic" + os.sep + "omegasSamples.dat")
    omega /= np.mean(omega)
    omegaStarDir = "/home/farceri/Documents/Data/omegaStar"
    omegaStarFile = np.array(["1024", "2048", "4096", "8192"])
    #make PDFs
    bins7 = np.geomspace(1e-04,1e-02,30)
    bins7 = np.concatenate((bins7, np.geomspace(1e-02,1.5,80)))
    bins7 = np.concatenate((bins7, np.linspace(1.5,3,100)))
    pdf7, edges7 = getPDF(omega7, bins7)
    bins = np.geomspace(5e-05,5e-03,30)
    bins = np.concatenate((bins, np.geomspace(5e-03,1.5,80)))
    bins = np.concatenate((bins, np.linspace(1.5,3,200)))
    pdf, edges = getPDF(omega, bins)
    fig, ax1 = plt.subplots(1, dpi = 200)
    #main plot
    ax1.loglog(edges[pdf>0], pdf[pdf>0], "+", color="#DC143C", markersize=7)
    ax1.loglog(edges[pdf>0], pdf[pdf>0], "k", linewidth=0.6)
    ax1.loglog(edges7[pdf7>0], pdf7[pdf7>0], "x", color="#0000CD", markersize=7)
    ax1.loglog(edges7[pdf7>0], pdf7[pdf7>0], "k", linewidth=0.6)
    ax1.set_ylabel("D($\\omega/\\langle\\omega\\rangle$)", fontsize=15)
    ax1.set_xlabel("$\\omega/\\langle\\omega\\rangle$", fontsize=15)
    ax1.tick_params(axis='both', labelsize=15)
    ax1.set_xlim(5e-05, 3.5)
    ax1.set_ylim(2e-03, 1.2)
    fig.tight_layout()
    #inset for omega star
    ax2 = plt.axes([0, 0, 1, 1])
    ip = InsetPosition(ax1, [0.3, 0.15, 0.58, 0.58])
    ax2.set_axes_locator(ip)
    omegaStarData = np.loadtxt(omegaStarDir + os.sep + omegaStarFile[0] + ".dat")
    ax2.loglog(omegaStarData[:,0], omegaStarData[:,1], 'o', color = "#000080", markersize = 7, markeredgewidth = 0.6, markeredgecolor = "k", alpha = 0.5)
    omegaStarData = np.loadtxt(omegaStarDir + os.sep + omegaStarFile[1] + ".dat")
    ax2.loglog(omegaStarData[:,0], omegaStarData[:,1], 'd', color = "#000080", markersize = 7, markeredgewidth = 0.6, markeredgecolor = "k", alpha = 0.5)
    omegaStarData = np.loadtxt(omegaStarDir + os.sep + omegaStarFile[2] + ".dat")
    ax2.loglog(omegaStarData[:,0], omegaStarData[:,1], 'v', color = "#000080", markersize = 7, markeredgewidth = 0.6, markeredgecolor = "k", alpha = 0.5)
    omegaStarData = np.loadtxt(omegaStarDir + os.sep + omegaStarFile[3] + ".dat")
    ax2.loglog(omegaStarData[:,0], omegaStarData[:,1], '^', color = "#000080", markersize = 7, markeredgewidth = 0.6, markeredgecolor = "k", alpha = 0.5)
    ax2.loglog(omegaStarData[6:,0], omegaStarData[6:,0]**0.5, linewidth=1.5, color = "k")
    ax2.set_ylabel("$\\omega^*$", fontsize=13)
    ax2.set_xlabel("$\Delta \\varphi$", fontsize=13)
    ax2.tick_params(axis='both', labelsize=13)
    ax2.legend(("$N=1024$", "$N=2048$", "$N=4096$", "$N=8192$"), fontsize=10, loc = "lower right")
    print(plt.rcParams.get('figure.figzise'))
    plt.savefig("/home/farceri/Pictures/paper/hardsoft_compare.pdf", transparent=True, format = "pdf", pad_inches=0.05)
    #annotation.slope_marker((1.5e-04, 6e-03), (1/2, 1), ax=ax2)
    plt.show()

def omega4(dirName):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize = (6.6, 8.85), dpi = 200)
    axs[0].set(ylabel = "P($\\omega/\\langle\\omega\\rangle$)")
    axs[1].set(ylabel = "D($\\omega/\\langle\\omega\\rangle$)")
    axs[1].set(xlabel = "$\\omega/\\langle\\omega\\rangle$")
    colorList = np.array(["#00008B", "#0000FF", "#4682B4", "#20B2AA", "#3CB371", "#228B22"])#008B8B
    for i in range(6):
        omegaloc = np.loadtxt(dirName + os.sep + "omega3D" + os.sep + str(i) + ".dat")
        prloc = np.loadtxt(dirName + os.sep + "pr3D" + os.sep + str(i) + ".dat")
        omega = np.loadtxt(dirName + os.sep + "omega3D" + os.sep + str(i) + ".dat")
        pr = np.loadtxt(dirName + os.sep + "pr3D" + os.sep + str(i) + ".dat")
        pr = pr[np.argsort(omega)]
        omega = np.sort(omega)
        if(i == 0):
            omega = np.concatenate((omega[:200], omega[200:8000], omega[8000:-1:20]))
            pr = np.concatenate((pr[:200], pr[200:8000], pr[8000:-1:20]))
        elif(i == 2):
            omega = np.concatenate((omega[:200:10], omega[200:8000:2], omega[8000:-1:20]))
            pr = np.concatenate((pr[:200:10], pr[200:8000:2], pr[8000:-1:20]))
        else:
            omega = np.concatenate((omega[:200:5], omega[200:8000], omega[8000:-1:20]))
            pr = np.concatenate((pr[:200:5], pr[200:8000], pr[8000:-1:20]))
        #pr plot
        axs[0].loglog(omega,pr,'o', markersize=7, markerfacecolor="None", markeredgecolor=colorList[i], markeredgewidth=1, alpha = 1)
        #modes plot
        pdf, edges = getPDF(omegaloc, np.geomspace(np.min(omegaloc), 1.85, 150))
        axs[1].loglog(edges[3:], pdf[3:], "^", markersize=14, markerfacecolor=colorList[i], markeredgecolor="k", markeredgewidth=0, alpha = 1)
    #inset
    #axin = plt.axes([0, 0, 1, 1])
    #ipaxin = InsetPosition(axs[0], [0.52, 0, 0.48, 0.48])
    #axin.set_axes_locator(ipaxin)
    axin = inset_axes(axs[0], '48%', '48%', 4)
    axis = inset_axes(axs[1], '52%', '52%', 3)
    colorInset = np.array(["#191970", "#0000FF", "#20B2AA", "#4682B4", "#228B22"])
    for i in range(0,5,2):
        omega = np.loadtxt(dirName + os.sep + "omega2D" + os.sep + str(4-i) + ".dat")
        omegaloc = np.loadtxt(dirName + os.sep + "omega2D" + os.sep + str(4-i) + ".dat")
        pr = np.loadtxt(dirName + os.sep + "pr2D" + os.sep + str(4-i) + ".dat")
        if(i!=0):
            omega = omega[:-1:3]
            pr = pr[:-1:3]
        axin.loglog(omega,pr,'o', markersize=5, markerfacecolor="None", markeredgecolor=colorInset[i], markeredgewidth=1, alpha = 1)
        pdf, edges = getPDF(omegaloc, np.geomspace(np.min(omegaloc), 1.85, 100))
        if(i==0):
            bins = np.linspace(np.min(omegaloc), 4e-03, 10)
            bins = np.concatenate((bins, np.geomspace(4.00001e-03, 2e-02, 20)))
            pdf, edges = getPDF(omegaloc, np.concatenate((bins, np.geomspace(2.000001e-02, 1.85, 50))))
        axis.loglog(edges, pdf, "^", markersize=10, markerfacecolor=colorInset[i], markeredgecolor="k", markeredgewidth=0, alpha = 1)
    #plot slope
    y = np.geomspace(0.13, 0.35)
    axis.plot(y, 2.5*y**1.8, linewidth=1.5, color = "k")
    axin.set_xlim(4e-04, 1.6)
    axin.set_ylim(1.1e-04, 1.8)
    axin.tick_params(axis='x', which='both', direction='in', pad=-16, labelsize=12)
    axin.tick_params(axis='y', which='both', direction='in', pad=-32, labelsize=12)
    axin.patch.set_alpha(1)
    axin.patch.set_facecolor("#FFFFFF")
    axis.set_xlim(1.02e-03, 1.6)
    axis.set_ylim(2.3e-02, 0.95)
    axis.tick_params(axis='x', which='both', direction='in', pad=-16, labelsize=12)
    axis.tick_params(axis='y', which='both', direction='in', pad=-32, labelsize=12)
    axis.patch.set_alpha(1)
    axis.patch.set_facecolor("#FFFFFF")
    #plot slope
    x = np.geomspace(0.28, 0.45)
    axs[1].plot(x, 10*x**4, linewidth=1.5, color = "k")
    #y = np.geomspace(4.7e-03, 8e-03)
    #axs[1].plot(y, 1e04*y**2, linewidth=1.5, color = "r")
    axs[1].set_xlim(4.5e-03, 1.8)
    axs[1].set_ylim(2.2e-02,)
    axs[0].set_ylim(3e-05,)
    axs[0].set_ylabel("PR($\\omega/\\langle\\omega\\rangle$)", fontsize=15)
    axs[1].set_ylabel("D($\\omega/\\langle\\omega\\rangle$)", fontsize=15)
    axs[1].set_xlabel("$\\omega/\\langle\\omega\\rangle$", fontsize=15)
    axs[0].tick_params(axis='both', labelsize=15)
    axs[1].tick_params(axis='both', labelsize=15)
    fig.patch.set_alpha(0)
    axs[0].patch.set_alpha(0)
    axs[1].patch.set_alpha(0)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.savefig("/home/farceri/Pictures/paper/omega4.pdf", transparent=False, format = "pdf", pad_inches=0.05)
    plt.show()

def plotpdf(dirName, numParticles):
    fig, ax = plt.subplots(dpi = 200)
    omega = np.loadtxt(dirName + os.sep + "omegasSamples.dat")
    pr = 1/(np.loadtxt(dirName + os.sep + "iprSamples.dat")*numParticles)
    pr = pr[np.argsort(omega)]
    omega = np.sort(omega)
    ax.loglog(omega, pr, 'o', markersize=4, markeredgecolor="k", markerfacecolor=[220/255,220/255,220/255], markeredgewidth = 0.5, alpha = 0.5)#228B22
    ax.set_ylabel("PR($\\omega/\\langle\\omega\\rangle$)", fontsize=18)
    ax.set_xlabel("$\\omega/\\langle\\omega\\rangle$", fontsize=18)
    ax.tick_params(axis='both', labelsize=15)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    fig.tight_layout()
    plt.savefig("/home/farceri/Pictures/paper/pr.png", transparent=True, format = "png", pad_inches=0.05)
    plt.show()

def prpdf(dirName, numParticles):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize = (3.3, 4.43), dpi = 200)
    omega = np.loadtxt(dirName + os.sep + "omegasSamples.dat")
    pr = 1/(np.loadtxt(dirName + os.sep + "iprSamples.dat")*numParticles)
    bins = np.linspace(np.min(omega), 0.2789999, 50)
    pdf, edges = getPDF(omega, np.concatenate((bins, np.geomspace(0.279, np.max(omega), 100))))
    #pdf, edges = getPDF(omega, np.geomspace(np.min(omega), np.max(omega), 100))
    #modes plot
    axs[1].loglog(edges, pdf, 'D', markersize=2, markerfacecolor="None", markeredgecolor = "k", markeredgewidth=0.5, alpha = 1)#228B22
    #pr plot
    pr = pr[np.argsort(omega)]
    omega = np.sort(omega)
    axs[0].loglog(omega, pr, 'o', markersize=1.5, markeredgecolor="k", markerfacecolor=[220/255,220/255,220/255], markeredgewidth = 0.5, alpha = 0.5)#228B22
    #plot slope
    #x = np.geomspace(0.25, 0.4)
    #axs[1].plot(x, 20*x**4, linewidth=1, color = "k")
    #x = np.geomspace(0.28, 0.45)
    #axs[1].plot(x, 10*x**4, linewidth=1.5, color = "k")
    #axs[1].set_xlim(3e-02, 6)
    axs[1].set_ylim(3e-04, 2)
    axs[0].set_ylabel("PR($\\omega/\\langle\\omega\\rangle$)", fontsize=8)
    axs[1].set_ylabel("D($\\omega/\\langle\\omega\\rangle$)", fontsize=8)
    axs[1].set_xlabel("$\\omega/\\langle\\omega\\rangle$", fontsize=8)
    axs[0].tick_params(axis='both', labelsize=8)
    axs[1].tick_params(axis='both', labelsize=8)
    fig.patch.set_alpha(0)
    axs[0].patch.set_alpha(0)
    axs[1].patch.set_alpha(0)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.savefig("/home/farceri/Pictures/paper/prpdf.png", transparent=True, format = "png", pad_inches=0.05)
    plt.show()

def prpdf3d2d(dirName):
    fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize = (6.4, 6), dpi = 150)
    #colorList = np.array(["#20B2AA", "#66CDAA", "#228B22"])#3CB371
    colorList = np.array(["#008000", "#32CD32", "#00FF00"])#32CD32
    colorList = np.array([[0,100/255,0], [40/255,170/255,40/255], [100/255,255/255,100/255]])
    omega7 = np.loadtxt(dirName + os.sep + "DeltaPhi1.1e-07/omegasSamples.dat")
    pr7 = 1/(np.loadtxt(dirName + os.sep + "DeltaPhi1.1e-07/iprSamples.dat")*4096)
    bins = np.geomspace(np.min(omega7), 0.5, 70)
    pdf, edges = getPDF(omega7, np.concatenate((bins, np.geomspace(0.50000000001, np.max(omega7), 100))))
    #pdf, edges = getPDF(omega7, np.geomspace(np.min(omega7), np.max(omega7), 100))
    axs[1,1].loglog(edges, pdf, "s", markersize=5, markerfacecolor="None", markeredgecolor="#191970", markeredgewidth=1, alpha = 1)
    axs[0,1].loglog(omega7, pr7,'o', markersize=3, markerfacecolor="None", markeredgecolor="#191970", markeredgewidth=1, alpha = 0.5)
    for i in range(3):
        omega2d = np.loadtxt(dirName + os.sep + "omega2D" + os.sep + str(i) + ".dat")
        pr2d = np.loadtxt(dirName + os.sep + "pr2D" + os.sep + str(i) + ".dat")
        pr2d = pr2d[np.argsort(omega2d)]
        omega2d = np.sort(omega2d)
        omega3d = np.loadtxt(dirName + os.sep + "omega3D" + os.sep + str(i) + ".dat")
        pr3d = np.loadtxt(dirName + os.sep + "pr3D" + os.sep + str(i) + ".dat")
        pr3d = pr3d[np.argsort(omega3d)]
        omega3d = np.sort(omega3d)
        pdf, edges = getPDF(omega2d, np.geomspace(np.min(omega2d), np.max(omega2d), 100))
        axs[1,0].loglog(edges, pdf, "D", markersize=5, markerfacecolor="None", markeredgecolor=colorList[i], markeredgewidth=1, alpha = 1)
        pdf, edges = getPDF(omega3d, np.geomspace(np.min(omega3d), np.max(omega3d), 100))
        axs[1,1].loglog(edges, pdf, "D", markersize=5, markerfacecolor="None", markeredgecolor=colorList[i], markeredgewidth=1, alpha = 1)
        if(i == 1):
            omega3d = omega3d[:-1:5]
            pr3d = pr3d[:-1:5]
        elif(i == 2):
            omega3d = omega3d[:-1:2]
            pr3d = pr3d[:-1:2]
        #pr plot
        axs[0,0].loglog(omega2d, pr2d,'o', markersize=3, markerfacecolor="None", markeredgecolor=colorList[i], markeredgewidth=1, alpha = 0.5)
        axs[0,1].loglog(omega3d, pr3d,'o', markersize=3, markerfacecolor="None", markeredgecolor=colorList[i], markeredgewidth=1, alpha = 0.5)
    #plot slope
    x = np.geomspace(0.1, 0.4)
    axs[1,0].plot(x, 1.5*x**1.8, linewidth=1.5, color = "k")
    axs[1,0].set_xlim(7e-04, 6)
    axs[1,0].set_ylim(2e-04, 2)
    #axs[0,0].set_ylim(3e-05,)
    x = np.geomspace(0.25, 0.5)
    axs[1,1].plot(x, 5*x**4, linewidth=1.5, color = "k")
    #axs[0,1].set_xlim(4e-04, 1.6)
    #axs[0,1].set_ylim(1.1e-04, 1.8)
    axs[1,1].set_xlim(4e-04, 8)
    axs[1,1].set_ylim(2e-04, 2)
    axs[0,0].set_title("$d=2$", fontsize=15)
    axs[0,0].set_ylabel("PR($\\omega/\\langle\\omega\\rangle$)", fontsize=15)
    axs[1,0].set_ylabel("D($\\omega/\\langle\\omega\\rangle$)", fontsize=15)
    axs[1,0].set_xlabel("$\\omega/\\langle\\omega\\rangle$", fontsize=15)
    axs[0,0].tick_params(axis='both', labelsize=13)
    axs[1,0].tick_params(axis='both', labelsize=13)
    #axs[0,1].set_ylabel("PR($\\omega/\\langle\\omega\\rangle$)", fontsize=15)
    #axs[1,1].set_ylabel("D($\\omega/\\langle\\omega\\rangle$)", fontsize=15)
    axs[0,1].set_title("$d=3$", fontsize=15)
    axs[1,1].set_xlabel("$\\omega/\\langle\\omega\\rangle$", fontsize=15)
    axs[0,1].tick_params(axis='both', labelsize=13)
    axs[1,1].tick_params(axis='both', labelsize=13)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("/home/farceri/Pictures/paper/prpdf2d3dnew.pdf", transparent=True, format = "pdf", pad_inches=0.05)
    plt.show()

def prpdf3d(dirName):
    fig, axs = plt.subplots(2, 1, sharex='col', figsize = (6.4, 6), dpi = 150)
    #colorList = np.array(["#20B2AA", "#66CDAA", "#228B22"])#3CB371
    colorList = np.array(["#008000", "#32CD32", "#00FF00"])#32CD32
    colorList = np.array([[0,100/255,0], [0,170/255,0], [0,220/255,0]])
    omega7 = np.loadtxt("/home/farceri/Documents/Data/VDOS/4096/DeltaPhi1.1e-07/omegasSamples.dat")
    pr7 = 1/(np.loadtxt("/home/farceri/Documents/Data/VDOS/4096/DeltaPhi1.1e-07/iprSamples.dat")*4096)
    bins = np.geomspace(np.min(omega7), 0.5, 70)
    pdf, edges = getPDF(omega7, np.concatenate((bins, np.geomspace(0.50000000001, np.max(omega7), 100))))
    pdf, edges = getPDF(omega7, np.geomspace(np.min(omega7), np.max(omega7), 100))
    axs[1].loglog(edges, pdf, "s", markersize=5, markerfacecolor="None", markeredgecolor="#191970", markeredgewidth=1, alpha = 1)
    axs[0].loglog(omega7, pr7,'o', markersize=3, markerfacecolor="None", markeredgecolor="#191970", markeredgewidth=1, alpha = 0.5)
    for i in range(3):
        omega3d = np.loadtxt(dirName + os.sep + "omega3D" + os.sep + str(i) + ".dat")
        pr3d = np.loadtxt(dirName + os.sep + "pr3D" + os.sep + str(i) + ".dat")
        pr3d = pr3d[np.argsort(omega3d)]
        omega3d = np.sort(omega3d)
        pdf, edges = getPDF(omega3d, np.geomspace(np.min(omega3d), np.max(omega3d), 100))
        axs[1].loglog(edges, pdf, "D", markersize=5, markerfacecolor="None", markeredgecolor=colorList[i], markeredgewidth=1, alpha = 1)
        if(i == 1):
            omega3d = omega3d[:-1:5]
            pr3d = pr3d[:-1:5]
        elif(i == 2):
            omega3d = omega3d[:-1:2]
            pr3d = pr3d[:-1:2]
        #pr plot
        axs[0].loglog(omega3d, pr3d,'o', markersize=3, markerfacecolor="None", markeredgecolor=colorList[i], markeredgewidth=1, alpha = 0.5)
    x = np.geomspace(0.25, 0.5)
    axs[1].plot(x, 5*x**4, linewidth=1.5, color = "k")
    #axs[0,1].set_xlim(4e-04, 1.6)
    #axs[0,1].set_ylim(1.1e-04, 1.8)
    axs[1].set_xlim(4e-04, 8)
    axs[1].set_ylim(2e-04, 2)
    axs[0].set_ylabel("PR($\\omega/\\langle\\omega\\rangle$)", fontsize=15)
    axs[1].set_ylabel("D($\\omega/\\langle\\omega\\rangle$)", fontsize=15)
    axs[1].set_xlabel("$\\omega/\\langle\\omega\\rangle$", fontsize=15)
    axs[0].tick_params(axis='both', labelsize=13)
    axs[1].tick_params(axis='both', labelsize=13)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("/home/farceri/Pictures/paper/prpdf3d_4.png", transparent=True, format = "png", pad_inches=0.05)
    plt.show()

def omegastar(dirName):
    star = mpath.Path.unit_regular_star(5)
    circle = mpath.Path.unit_circle()
    verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
    codes = np.concatenate([circle.codes, star.codes])
    cut_star = mpath.Path(verts, codes)
    fig, ax = plt.subplots(1, dpi = 200)
    omegaList = ["1024", "2048", "4096", "8192", "4096fromIso"]
    markerList = ["o", "^", "v", "D", "s"]
    for i in range(5):
        omegastar = np.loadtxt(dirName + os.sep + omegaList[i] + ".dat", comments="#")
        #omegastar = omegastar[omegastar[:,1]>0]
        if(omegaList[i] == "8192" or "2048"):
            omegastar = omegastar[1:]
        if(omegaList[i] == "4096"):
            omegastar = omegastar[2:]
        if(omegaList[i] == "4096fromIso"):
            omegastar = omegastar[1:]
            colorList = np.ones((omegastar.shape[0],3))
            colorList[:,0] += np.flip(np.linspace(-1,-0.5,omegastar.shape[0]))
            colorList[:,1] += np.flip(np.linspace(-1,-0.3,omegastar.shape[0]))
        else:
            colorList = np.zeros((omegastar.shape[0],3))
            colorList[:,1] = 1 - np.linspace(0,0.5,omegastar.shape[0])
            colorList[:,2] = 0.5 - np.linspace(0,0.5,omegastar.shape[0])
            colorList[:,0] = 0.5 - np.linspace(0,0.5,omegastar.shape[0])
        if(omegaList[i] == "4096fromIso"):
            for j in range(omegastar.shape[0]):
                plt.loglog(omegastar[j,0], omegastar[j,1], marker=markerList[i], markersize=14, markerfacecolor=colorList[omegastar.shape[0]-1-j], markeredgecolor="k", markeredgewidth=1, alpha = 1)
                #plt.loglog(omegastar[j,0], omegastar[j,1], marker="*", markersize=14, markerfacecolor="w", markeredgecolor="k", markeredgewidth=1, alpha = 1)
        else:
            for j in range(omegastar.shape[0]):
                plt.loglog(omegastar[j,0], omegastar[j,1], marker=markerList[i], markersize=12, markerfacecolor=colorList[j], markeredgecolor="k", markeredgewidth=1, alpha = 1)
    x = np.geomspace(2e-06, 2e-03)
    plt.plot(x, x**0.5, linewidth=1.5, color='k')
    ax.set_ylabel("$\\omega^*$", fontsize=18)
    ax.set_xlabel("$\Delta \\varphi$", fontsize=18)
    ax.tick_params(axis='both', labelsize=15)
    fig.tight_layout()
    #print(plt.rcParams)
    plt.savefig("/home/farceri/Pictures/paper/omegastar.pdf", transparent=True, format = "pdf", pad_inches=0.05)
    plt.show()

dirName = sys.argv[1]
plot = sys.argv[2]

if(plot == "hardsoft"):
    comparePDF(dirName)
elif(plot == "gaps"):
    plotGapPDF(dirName)
elif(plot == "modes"):
    plotModesPDF(dirName)
elif(plot == "omega4"):
    omega4(dirName)
elif(plot == "prpdf"):
    numParticles = int(sys.argv[3])
    prpdf(dirName, numParticles)
elif(plot == "prpdf3d2d"):
    prpdf3d2d(dirName)
elif(plot == "prpdf3d"):
    prpdf3d(dirName)
elif(plot == "omegastar"):
    omegastar(dirName)
elif(plot == "plotpdf"):
    numParticles = int(sys.argv[3])
    plotpdf(dirName, numParticles)
