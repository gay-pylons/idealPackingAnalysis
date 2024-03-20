#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2023
sweeps directories of packings in order to produce collection of moduli and save to repository.
@author: violalum
"""
import os
import numpy as np
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("pyCudaPacking","/home/vlum/code/pyCudaPacking/pyCudaPacking/__init__.py")
pcp = importlib.util.module_from_spec(spec)
sys.modules["pyCudaPacking"] = pcp
spec.loader.exec_module(pcp)

def bashModulus(path):
	os.system(f"sbatch singleModulus.srun {path}")

if __name__ == '__main__':
	directory=str(sys.argv[1])
	try:
		n=int(sys.argv[2])
		multiplePackings=True
	except:
		multiplePackings=False
	shear=[]
	bulk=[]
	pressure=[]
	contacts=[]
	zExcess=[]
	for filename in os.scandir(directory):
		if(filename.is_dir()):
			s, b = bashModulus(f'{filename.path}')
