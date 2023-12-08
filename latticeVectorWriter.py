#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 14:58:01 2022
writes lattice vectors: best to run from dev env, havent implemented command line yet
@author: violalum
"""
import numpy as np
n=16384
#parameters from circlepack
ab =  -6.1580388298e+00 + -2.6391295874e-01j
bb = 2.6129490743e-01 + -6.1571727530e+00j
thingName="idealPack16384-1"
fileName=f"{thingName}-latVecs"
bParam=np.array([ab,bb])
vec1=bParam.real
vec2=bParam.imag
vex=np.array([vec1,vec2],dtype=np.quad)
if vex[0][0]<0:
	np.save(f"./{n}/latVecs/{fileName}",-vex)
else:
	np.save(f"./{n}/latVecs/{fileName}",vex)#np.transpose(vex))
