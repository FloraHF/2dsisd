import numpy as np
from copy import deepcopy
from math import asin, acos, sin, cos, pi, sqrt

from tensorflow.keras.models import load_model

from Config import Config
from geometries import line, circle
from plotter import Plotter
from envelope import envelope_traj
from strategies_fastD import deep_target_strategy


class Player(object):

	def __init__(self, env, role, x):

		self.env = env
		self.role = role
		self.dt = env.dt
		self.x = x
        
		if 'D' in role:
			self.v = env.vd
		elif 'I' in role:
			self.v = env.vi

	def step(self, action):
		self.x += self.dt * self.v * np.array([cos(action), sin(action)])
    
	def reset(self, x):
		self.x = deepcopy(x)

	def get_x(self):
		return deepcopy(self.x)


class BaseGame(object):
	"""docstring for BaseGame"""
	def __init__(self, target, type, ni=1, nd=2):

		self.target = target
		self.vd = Config.VD
		if type == 'fastD':
			self.vi = Config.VI_SLOW
		elif type == 'slowD':
			self.vi = Config.VI_FAST
		self.a = self.vd/self.vi
		self.r = Config.CAP_RANGE
		self.dt = 0.1
		self.xi_0 = Config.XI0
		self.xd_0 = Config.XD0
		self.ni = ni
		self.nd = nd

		self.players = dict()
		for i in range(ni):
			pid = 'I'+str(i)
			self.players[pid] = Player(self, pid, Config.XI0[i])
		for i in range(nd):
			pid = 'D'+str(i)
			self.players[pid] = Player(self, pid, Config.XD0[i])

		self.plotter = Plotter(target, self.a)

	def is_capture(self, xi, xds):
		cap = False
		for xd in xds:
			d = sqrt((xi[0] - xd[0])**2 + (xi[1] - xd[1])**2)
			cap = cap or d < self.r
		return cap

	def get_state(self):
		xis, xds = np.zeros((self.ni,2)), np.zeros((self.nd,2))
		for role, p in self.players.items():
			if 'I' in role:
				xis[int(role[-1]),:] = p.get_x()
			elif 'D' in role:
				xds[int(role[-1]),:] = p.get_x()
		return xis, xds

	def step(self, xis, xds):
		actions = self.strategy(xis, xds)
		for (rp, p), (ra, act) in zip(self.players.items(), actions.items()):
			p.step(act)
		return self.get_state()

	def advance(self, te):	
		t = 0
		xi0, xd0 = self.get_state()
		xis, xds = [xi0], [xd0]
		while t < te:
			xi, xd = self.step(xis[-1], xds[-1])
			xis.append(xi)
			xds.append(xd)
			t += self.dt 
			if self.is_capture(xi[0], xd):
				print('capture')
				break
		return xis, xds

	def convert_data(self, xs):
		n = xs[0].shape[0]
		xs_ = [[] for _ in range(n)]
		for x in xs:
			for i in range(n):
				xs_[i].append(x[i])
		return np.asarray(xs_)

	def reset(self, xis, xds):
		for role, p in self.players.items():
			if 'D' in role:
				p.reset(xds[int(role[-1])])
			elif 'I' in role:
				p.reset(xis[int(role[-1])])


class SlowDgame(BaseGame):
	"""docstring for SlowDgame"""
	def __init__(self, target, ni, nd):
		print('what')
		super(SlowDgame, self).__init__(target, type='slowD', ni=ni, nd=nd)
		# super(SlowDgame, self).__init__(self)

		self.policies = dict()
		for role, p in self.players.items():
			self.policies[role] = load_model(Config.POLICY_DIR+'_'+role)
		self.s_lb = -asin(self.a)
		self.gmm_lb = acos(self.a)

	def analytic_traj(self, S=.1, T=4., gmm=acos(Config.VD/Config.VI_FAST)+0.3, D=0, delta=0.299999, n=50):
		assert S >= self.s_lb
		assert gmm >= self.gmm_lb
		xs, _ = envelope_traj(S, T, gmm, D, delta, n=n)
		return xs

	def strategy(self, xis, xds):
		acts = dict()
		# print(xis.shape, type(xds))
		# print(xds[0], xis[0], xds[1])
		x = np.concatenate((xds[0], xis[0], xds[1]))
		for role, p in self.policies.items():
			if 'D' in role:
				acts[role] = p.predict(x[None])[0]
			elif 'I' in role:
				acts[role] = p.predict(x[None])[0]
		return acts

class FastDgame(BaseGame):
	"""docstring for FastDgame"""
	def __init__(self, target, ni, nd):
		super(FastDgame, self).__init__(target, type='fastD', ni=ni, nd=nd)

	def strategy(self, xis, xds):
		psis, phis = deep_target_strategy(xis[0], xds, self.target, self.a)
		acts = dict()
		for role, p in self.players.items():
			if 'D' in role:
				acts[role] = phis[int(role[-1])]
			elif 'I' in role:
				acts[role] = psis[int(role[-1])]
		return acts

if __name__ == '__main__':

	# game = SlowDgame(line, ni=1, nd=2)
	# xs_ref = game.analytic_traj()
	# game.reset([xs_ref[0,2:4]], [xs_ref[0,:2], xs_ref[0,4:]])
	# xis, xds = game.advance(7.)
	# xis, xds = game.convert_data(xis), game.convert_data(xds)
	# xs_play = np.concatenate((xds[0], xis[0], xds[1]),-1)
	# game.plotter.plot({'play':xs_play, 'ref':xs_ref})

	game = FastDgame(line, ni=1, nd=2)
	game.reset([np.array([-1., 9.])], [np.array([-8., 6.]), np.array([6., 5.])])
	xis, xds = game.advance(10.)
	xis, xds = game.convert_data(xis), game.convert_data(xds)
	xs_play = np.concatenate((xds[0], xis[0], xds[1]),-1)
	game.plotter.plot({'play':xs_play})			
		
		
						