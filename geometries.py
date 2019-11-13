import numpy as np
from math import sqrt
from Config import Config

def line(x):
    return x[1]

def circle(x, R=Config.TAG_RANGE):
	return sqrt(x[0]**2 + x[1]**2) - R

def dominant_region(x, xi, xds, a):
	for i, xd in enumerate(xds):
		if i == 0:
			inDR = a*np.linalg.norm(x-xi) - (np.linalg.norm(x-xd) - Config.CAP_RANGE)
		else:
			inDR = max(inDR, a*np.linalg.norm(x-xi) - (np.linalg.norm(x-xd) - Config.CAP_RANGE))
	return inDR	