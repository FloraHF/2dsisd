import numpy as np
from math import asin, acos, sin, cos, pi, sqrt

from Config import Config
from plotter import Plotter
from strategy_fastD import deep_target_strategy

class FastDgame(object):
	"""docstring for TDSIgame"""
	def __init__(self):
		
		self.strategy = deep_target_strategy
		self.vd = Config.VD
		self.vi = Config.VI_SLOW
		self.r = Config.CAP_RANGE
		self.dt = 0.1

		self.plotter = Plotter()
		
		self.x0 = np.array([-6., 3., 1., 8., 6., 3.]) 

	def is_capture(self, x):
		d1 = sqrt((x[0] - x[2])**2 + (x[0] - x[3])**2)
		d2 = sqrt((x[4] - x[2])**2 + (x[5] - x[3])**2)
		# print(d1, d2)
		return d1 < self.r or d2 < self.r 

	def step(self, x):
		psi, phi = self.strategy
		xd1 = x[0] + self.vd*cos(phi_1)*self.dt
		yd1 = x[1] + self.vd*sin(phi_1)*self.dt
		xd2 = x[4] + self.vd*cos(phi_2)*self.dt
		yd2 = x[5] + self.vd*sin(phi_2)*self.dt
		xi = x[2] + self.vi*cos(psi)*self.dt
		yi = x[3] + self.vi*sin(psi)*self.dt
		return np.array([xd1, yd1, xi, yi, xd2, yd2])		