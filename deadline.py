from math import pi, sqrt, sin, cos, atan2, acos
import matplotlib.pyplot as plt

with open('config.csv', 'a') as f:
	lines = f.readlines()
	for line in lines:
		data = line.split(',')
		var = data[0]
		val = data[-1].rstrip()
		if 'vd' == var:
			vd = float(val)
		if 'vi' == var:
			vi = float(val)
w = vi/vd

def get_s(xi, xd):
	dx = xi - xd
	dx = np.concatenate((dx, [0]))
	return -pi/2 - atan2(dx[1], dx[0])

def Q(s):
	return sqrt(1 + w**2 -2*w*sin(s))

def get_phi(s):
	cphi = w*cos(s)/Q(s)
	sphi = (1 - w*sin(s))/Q(s)
	return atan2(sphi, cphi)

def get_psi(s):
	cpsi = cos(s)/Q(s)
	spsi = (w - sin(s))/Q(s)
	return atan2(spsi, cpsi)

def generate_deadline_traj():