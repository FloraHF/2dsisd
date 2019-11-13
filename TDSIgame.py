import matplotlib.pyplot as plt

import numpy as np
from math import asin, acos, sin, cos, tan, pi, atan2, sqrt

from tensorflow.keras.models import load_model

from Config import Config
from plotter import Plotter
from envelope import envelope_traj

class TDSIgame(object):
	"""docstring for TDSIgame"""
	def __init__(self):
		
		self.policy_D1 = load_model(Config.POLICY_DIR+'_D1')
		self.policy_D2 = load_model(Config.POLICY_DIR+'_D2')
		self.policy_I = load_model(Config.POLICY_DIR+'_I')
		self.barrier = load_model(Config.BARRIER_DIR)
		self.vd = Config.VD 
		self.vi = Config.VI
		self.r = Config.CAP_RANGE
		self.a = self.vd/self.vi
		self.s_lb = -asin(self.a)
		self.gmm_lb = acos(self.a)
		self.dt = 0.1

		self.plotter = Plotter()
		
		self.x0 = np.array([-6., 3., 1., 8., 6., 3.]) 

	def analytic_traj(self, S=.2, T=4., gmm=acos(Config.VD/Config.VI)+0.3, D=0, delta=0.299999, n=50):
		assert S >= self.s_lb
		assert gmm >= self.gmm_lb
		xs, _ = envelope_traj(S, T, gmm, D, delta, n=n)
		return xs

	def step(self, x):
		phi_1 = self.policy_D1.predict(x[None])[0]
		phi_2 = self.policy_D2.predict(x[None])[0]
		psi = self.policy_I.predict(x[None])[0]
		xd1 = x[0] + self.vd*cos(phi_1)*self.dt
		yd1 = x[1] + self.vd*sin(phi_1)*self.dt
		xd2 = x[4] + self.vd*cos(phi_2)*self.dt
		yd2 = x[5] + self.vd*sin(phi_2)*self.dt
		xi = x[2] + self.vi*cos(psi)*self.dt
		yi = x[3] + self.vi*sin(psi)*self.dt
		return np.array([xd1, yd1, xi, yi, xd2, yd2])

	def is_capture(self, x):
		d1 = sqrt((x[0] - x[2])**2 + (x[0] - x[3])**2)
		d2 = sqrt((x[4] - x[2])**2 + (x[5] - x[3])**2)
		# print(d1, d2)
		return d1 < self.r or d2 < self.r 

	def advance(self, x0, te):
		t = 0
		xs = [x0]
		while t < te:
			x = self.step(xs[-1])
			xs.append(x)
			t += self.dt 
			if self.is_capture(x):
				print('capture')
				break
		return np.asarray(xs)


if __name__ == '__main__':

	game = TDSIgame()
	xs_ref = game.analytic_traj()
	xs_play = game.advance(xs_ref[0], 6.)
	game.plotter.plot({'play':xs_play, 'ref':xs_ref})
	# xs_play = game.advance(game.x0, 6.)
	# game.plotter.plot({'play':xs_play})
	# fig, ax = plt.subplots()
	# ax.plot(xs[:,0], xs[:,1], 'b')
	# ax.plot(xs[:,2], xs[:,3], 'r')
	# ax.plot(xs[:,4], xs[:,5], 'g')
	# ax.axis('equal')
	# ax.grid()
	# plt.show()

