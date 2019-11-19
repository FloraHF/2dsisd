import os
import numpy as np
from math import sqrt

with open(os.path.dirname(__file__)+'/info_fastD.csv', 'r') as f:
	data = f.readlines()
	for line in data:
		if 'rt' in line:
			R = float(line.split(',')[-1])
		if 'rc' in line:
			r = float(line.split(',')[-1])

def line(x):
    return x[1]

def circle(x, R=R):
	return sqrt(x[0]**2 + x[1]**2) - R

def dominant_region(x, xi, xds, a):
	for i, xd in enumerate(xds):
		if i == 0:
			inDR = a*np.linalg.norm(x-xi) - (np.linalg.norm(x-xd) - r)
		else:
			inDR = max(inDR, a*np.linalg.norm(x-xi) - (np.linalg.norm(x-xd) - r))
	return inDR	