import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
from math import pi, sin, cos, sqrt, asin, acos, atan2

from Config import Config
from coords import xy_to_s
from strategy import get_Q, get_phi, get_psi
from RK4 import rk4, rk4_fxt_interval
vd = Config.VD
vi = Config.VI
r = Config.CAP_RANGE
a = vd/vi
w = 1/a

def dt_ds(s, t):
	return -r*get_Q(s)/((1. - w**2)*vd)

def get_time(se, s0=-asin(1/w), dt=0.02):
	return rk4_fxt_interval(dt_ds, s0, 0, se, dt)

def envelope_v(s):
	# this is forward in time!!!!!
	Q = get_Q(s)
	vxd = -vd*cos(s)/Q
	vyd = -vd*(w + sin(s))/Q
	vxi = -vi*w*cos(s)/Q
	vyi = -vi*(1 + w*sin(s))/Q
	return vxd, vyd, vxi, vyi

def envelope_core(s):
	beta = asin(1/w)
	A = r*w/(w**2 - 1)
	B = r*w/sqrt(w**2 - 1)
	xd =   A*sin(s)/w 
	yd = - A*cos(s)/w + A*s 
	xi =   A*sin(s)*w 
	yi = - A*cos(s)*w + A*s

	return xd, yd, xi, yi

def envelope_analytic(s, t, s0=-asin(1/w)):

	xd_s0, yd_s0, xi_s0, yi_s0 = envelope_core(s0)
	xdc, ydc, xic, yic = envelope_core(s)

	xd = xdc - xd_s0 - r*sin(s0)
	yd = ydc - yd_s0 + r*cos(s0)
	xi = xic - xi_s0 + 0.
	yi = yic - yi_s0 + 0. 

	if t > 0:
		vxd, vyd, vxi, vyi = envelope_v(s)
		xd = xd - vxd*t
		yd = yd - vyd*t
		xi = xi - vxi*t
		yi = yi - vyi*t

	return xd, yd, xi, yi

def envelope_rotate(s, t, delta=0):
	tht = pi/2 + delta
	xd, yd, xi, yi = envelope_analytic(s, t)
	C = np.array([[cos(tht), -sin(tht)], 
				  [sin(tht), cos(tht)]])
	xyd = C.dot(np.array([xd, yd]))
	xyi = C.dot(np.array([xi, yi]))

	return xyd, xyi

def envelope_6d(s, t, gmm=acos(1/w), D=0, delta=0):
	# input t is the total time
	ub = gmm - acos(1/w)
	assert delta < ub
	tc = get_time(s)
	ts = t-tc
	if ts < 0:
		ts = 0
		t = tc
	if ub - delta < 1e-6:
		xyd1, xyi = envelope_rotate(s, ts, delta=delta+D)
	else:
		alpha = pi/2 + delta + D
		xi = vi*t*cos(alpha)
		yi = vi*t*sin(alpha)
		xyi = np.array([xi, yi])

		beta = pi/2 + D + gmm
		xd1 = (vd*t + r)*cos(beta)
		yd1 = (vd*t + r)*sin(beta)
		xyd1 = np.array([xd1, yd1])
	# print(t, tt)

	beta = pi/2 + D - gmm
	xd2 = (vd*t + r)*cos(beta)
	yd2 = (vd*t + r)*sin(beta)
	xyd2 = np.array([xd2, yd2])

	return np.concatenate((xyd1, xyi, xyd2))

def envelope_traj(S, T, gmm, D, delta, n=50):

	tc = get_time(S) # time on curved traj
	nc = max(int(n*tc/(T + tc)), 1) # n for on curved traj
	ns = n - nc # n for on straight traj

	xs = []
	for s in np.linspace(-asin(1/w), S, nc):
		x = envelope_6d(s, get_time(s), gmm=gmm, D=D, delta=delta)
		xs.append(x)
		# print(x)
	for t in np.linspace(0.1, T, ns):
		x = envelope_6d(s, tc+t, gmm=gmm, D=D, delta=delta)
		xs.append(x)
		# print(x)
	xs = np.asarray(xs)

	return xs[::-1] # forward in time!!!

def envelope_policy(xs):
	policy = []
	for x, x_ in zip(xs[:-2], xs[1:]):
		phi_1 = atan2(x_[1]-x[1], x_[0]-x[0])
		phi_2 = atan2(x_[5]-x[5], x_[4]-x[4])
		psi = atan2(x_[3]-x[3], x_[2]-x[2])
		policy.append([phi_1, psi, phi_2])
		print(policy[-1])
	policy.append(policy[-1])

	return np.asarray(policy)

# def envelope_for_value(s0):
# 	return envelope_analytic(-asin(1/w), s0=s0)[-1]

if __name__ == '__main__':

	xs = envelope_traj(0.3, 4., acos(1/w)+0.2, 0.2, 0.2)
	envelope_policy(xs)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(xs[:,0], xs[:,1])
	ax.plot(xs[:,2], xs[:,3])
	ax.plot(xs[:,4], xs[:,5])
	ax.axis('equal')
	ax.grid()
	plt.show()
