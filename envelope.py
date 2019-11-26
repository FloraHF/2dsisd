import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
from math import pi, sin, cos, tan, sqrt, asin, acos, atan2

from RK4 import rk4, rk4_fxt_interval

class Envelope(object):
	"""docstring for Envelope"""
	def __init__(self, vi, vd, r):
		self.vi = vi 
		self.vd = vd 
		self.r = r
		self.a = vd/vi
		self.w = 1/self.a
		self.s_lb = -asin(1/self.w)
		

	def get_Q(self, s):
		return sqrt(1 + self.w**2 + 2*self.w*sin(s))

	def get_phi(self, s):
		Q = self.get_Q(s)
		cphi = self.w*cos(s)/Q 
		sphi = -(1 + self.w*sin(s))/Q
		return atan2(sphi, cphi)

	def get_psi(self, s):
		Q = self.get_Q(s)
		cpsi = cos(s)/Q 
		spsi = -(self.w + sin(s))/Q
		return atan2(spsi, cpsi)

	def mirror(self, xy, k):
		return np.array([2*k*xy[1] + xy[0]*(1-k**2), 2*k*xy[0] - xy[1]*(1-k**2)])/(1+k**2)

	def dt_ds(self, s, t):
		return -self.r*self.get_Q(s)/((1. - self.w**2)*self.vd)

	def get_time(self, se, s0, dt=0.02):
		return rk4_fxt_interval(self.dt_ds, s0, 0, se, dt)

	def envelope_v(self, s):
		# this is forward in time!!!!!
		Q = self.get_Q(s)
		vxd = -self.vd*cos(s)/Q
		vyd = -self.vd*(self.w + sin(s))/Q
		vxi = -self.vi*self.w*cos(s)/Q
		vyi = -self.vi*(1 + self.w*sin(s))/Q
		return vxd, vyd, vxi, vyi

	def envelope_core(self, s):
		beta = asin(1/self.w)
		A = self.r*self.w/(self.w**2 - 1)
		B = self.r*self.w/sqrt(self.w**2 - 1)
		xd =   A*sin(s)/self.w 
		yd = - A*cos(s)/self.w + A*s 
		xi =   A*sin(s)*self.w 
		yi = - A*cos(s)*self.w + A*s
		return xd, yd, xi, yi

	def envelope_analytic(self, s, t, s0):

		xd_s0, yd_s0, xi_s0, yi_s0 = self.envelope_core(s0)
		xdc, ydc, xic, yic = self.envelope_core(s)

		xd = xdc - xd_s0 - self.r*sin(s0)
		yd = ydc - yd_s0 + self.r*cos(s0)
		xi = xic - xi_s0 + 0.
		yi = yic - yi_s0 + 0. 

		if t > 0:
			vxd, vyd, vxi, vyi = self.envelope_v(s)
			xd = xd - vxd*t
			yd = yd - vyd*t
			xi = xi - vxi*t
			yi = yi - vyi*t
		return xd, yd, xi, yi

	def envelope_rotate(self, s, t, delta=0):
		tht = pi/2 + delta
		xd, yd, xi, yi = self.envelope_analytic(s, t, self.s_lb)
		C = np.array([[cos(tht), -sin(tht)], 
					  [sin(tht), cos(tht)]])
		xyd = C.dot(np.array([xd, yd]))
		xyi = C.dot(np.array([xi, yi]))

		return xyd, xyi

	def envelope_6d(self, s, t, gmm, D=0, delta=0):
		# input t is the total time
		ub = gmm - acos(1/self.w)
		# print(delta, ub)
		assert delta <= ub
		tc = self.get_time(s, self.s_lb)
		ts = t-tc
		if ts < 0:
			ts = 0
			t = tc
		if ub - delta < 1e-6:
			xyd1, xyi = self.envelope_rotate(s, ts, delta=delta+D)
		else:
			alpha = pi/2 + delta + D
			xi = vi*t*cos(alpha)
			yi = vi*t*sin(alpha)
			xyi = np.array([xi, yi])

			beta = pi/2 + D + gmm
			xd1 = (self.vd*t + r)*cos(beta)
			yd1 = (self.vd*t + r)*sin(beta)
			xyd1 = np.array([xd1, yd1])
		# print(t, tt)

		beta = pi/2 + D - gmm
		xd2 = (self.vd*t + self.r)*cos(beta)
		yd2 = (self.vd*t + self.r)*sin(beta)
		xyd2 = np.array([xd2, yd2])

		return xyd1, xyi, xyd2

	def envelope_traj(self, S, T, gmm, D, delta, n=50):

		tc = self.get_time(S, self.s_lb) # time on curved traj
		nc = max(int(n*tc/(T + tc)), 1) # n for on curved traj
		ns = n - nc # n for on straight traj
		xs1 = []
		if delta > 0 or S > self.s_lb:
			xs2 = []
		k = tan(D + pi/2)
		for s in np.linspace(self.s_lb, S, nc):
			xyd1, xyi, xyd2 = self.envelope_6d(s, self.get_time(s, self.s_lb), gmm, D=D, delta=delta)
			xs1.append(np.concatenate((xyd1, xyi, xyd2)))
			if delta > 0 or S > self.s_lb:
				xs2.append(np.concatenate((self.mirror(xyd2, k), self.mirror(xyi, k), self.mirror(xyd1, k))))
			# print(x)
		for t in np.linspace(0.1, T, ns):
			xyd1, xyi, xyd2 = self.envelope_6d(s, tc+t, gmm, D=D, delta=delta)
			xs1.append(np.concatenate((xyd1, xyi, xyd2)))
			if delta > 0 or S > self.s_lb:
				xs2.append(np.concatenate((self.mirror(xyd2, k), self.mirror(xyi, k), self.mirror(xyd1, k))))
			# print(x)
		xs1 = np.asarray(xs1)
		if delta > 0 and S > self.s_lb:
			xs2 = np.asarray(xs2)
			return xs1[::-1], xs2[::-1] # forward in time!!!
		else:
			return xs1[::-1], None

	def envelope_policy(self, xs):
		policy = []
		for x, x_ in zip(xs[:-2], xs[1:]):
			phi_1 = atan2(x_[1]-x[1], x_[0]-x[0])
			phi_2 = atan2(x_[5]-x[5], x_[4]-x[4])
			psi = atan2(x_[3]-x[3], x_[2]-x[2])
			policy.append([phi_1, psi, phi_2])
			# print(policy[-1])
		policy.append(policy[-1])
		return np.asarray(policy)

