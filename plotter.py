import matplotlib.pyplot as plt
import numpy as np
from math import pi, sin, cos

from Config import Config

class Plotter(object):
	"""docstring for Plotter"""
	def __init__(self):

		self.fig, self.ax = plt.subplots()
		self.linestyles = {'play': (0, ()), 'ref':(0, (5, 5))}
		self.colors = {'D1': 'g', 'I': 'r', 'D2': 'b'}
		self.r = Config.CAP_RANGE

	def plot_capture_ring(self, player, situation, x, n=50):
		xs, ys = [], []
		for t in np.linspace(0, 2*pi, n):
			xs.append(x[0] + self.r*cos(t))
			ys.append(x[1] + self.r*sin(t))
		self.ax.plot(xs, ys, color=self.colors[player], linestyle=self.linestyles[situation])

	def plot_traj(self, player, situation, xs):
		self.ax.plot(xs[:,0], xs[:,1], color=self.colors[player], linestyle=self.linestyles[situation])

	def plot_connect(self, p1, p2, xs1, xs2, skip=20):
		n = xs1.shape[0]
		for i, (x1, x2) in enumerate(zip(xs1, xs2)):
			if i%skip == 0 or i==n-1:
				self.ax.plot([x1[0], x2[0]], [x1[1], x2[1]], 'b--')
				self.ax.plot(x1[0], x1[1], 'o', color=self.colors[p1])
				self.ax.plot(x2[0], x2[1], 'o', color=self.colors[p2])

	def show_plot(self):
		self.ax.axis('equal')
		self.ax.grid()
		plt.show()

	def plot(self, xs):
		for situ, x in xs.items():
			dim = x.shape[-1]
			_p = None
			for i, (p, c) in zip([2, 4, 6], self.colors.items()):
				if i <= dim:
					self.plot_traj(p, situ, x[:,i-2:i])
					if 'D' in p:
						self.plot_capture_ring(p, situ, x[-1,i-2:i])
					if i >= 4 and situ=='play':
						self.plot_connect(_p, p, x[:,i-4:i-2], x[:,i-2:i])
				_p = p


		self.show_plot()

	def reset(self):
		self.ax.clear()