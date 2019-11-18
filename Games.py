import os
import numpy as np
from copy import deepcopy
from math import asin, acos, sin, cos, pi, sqrt

from tensorflow.keras.models import load_model

from Config import Config
from geometries import line, circle
from experiment_replay import ReplayPool
from plotter import Plotter
from envelope import envelope_traj
from strategies_fastD import deep_target_strategy


class Player(object):

	def __init__(self, env, role, res_dir='res1/'):

		self.env = env
		self.role = role
		self.dt = env.dt
		self.exp = ReplayPool(self.role, res_dir=res_dir)
		self.x = np.array([self.exp.x(self.exp.t_start), self.exp.y(self.exp.t_start)])
        
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
	def __init__(self, target, type, res_dir='res1/', ni=1, nd=2):

		self._script_dir = os.path.dirname(__file__)
		self._res_dir = res_dir
		with open(res_dir+'info.csv', 'r') as f:
			data = f.readlines()
			for line in data:
				if 'vd' in line:
					self.vd = float(line.split(',')[-1])
				if 'vi' in line:
					self.vi = float(line.split(',')[-1])
				if 'rc' in line:
					self.r = float(line.split(',')[-1])
				if 'r_close' in line:
					self.r_close = float(line.split(',')[-1])*self.r
				if 'k_close' in line:
					self.k_close = float(line.split(',')[-1])		

		self.target = target
		self.a = self.vd/self.vi
		self.dt = 0.1
		self.ni = ni
		self.nd = nd

		self.players = dict()
		for i in range(ni):
			pid = 'I'+str(i)
			self.players[pid] = Player(self, pid, res_dir=res_dir)
		for i in range(nd):
			pid = 'D'+str(i)
			self.players[pid] = Player(self, pid, res_dir=res_dir)

		self.plotter = Plotter(target, self.a, self.r)

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

	def read_exp_state(self, t):
		xis, xds = np.zeros((self.ni,2)), np.zeros((self.nd,2))
		for role, p in self.players.items():
			if 'I' in role:
				xis[int(role[-1]),:] = np.array([p.exp.x(t), p.exp.y(t)])
			elif 'D' in role:
				xds[int(role[-1]),:] = np.array([p.exp.x(t), p.exp.y(t)])
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
		return self.convert_data(xis), self.convert_data(xds)

	def replay_exp(self):
		t_start, t_end = 100., -1.,
		for role, p in self.players.items():
			if p.exp.t_start < t_start:
				t_start = p.exp.t_start
			if p.exp.t_end > t_end:
				t_end = p.exp.t_end
		t = t_start
		xis, xds = [], []
		while t < t_end:
			xi, xd = self.read_exp_state(t)
			xis.append(xi)
			xds.append(xd)
			t += self.dt 
		return self.convert_data(xis), self.convert_data(xds)

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
	def __init__(self, target, res_dir, ni, nd):
		print('what')
		super(SlowDgame, self).__init__(target, type='slowD', res_dir=res_dir, ni=ni, nd=nd)
		# super(SlowDgame, self).__init__(self)

		self.policies = dict()
		for role, p in self.players.items():
			self.policies[role] = load_model(Config.POLICY_DIR+'_'+role)
		self.s_lb = -asin(self.a)
		self.gmm_lb = acos(self.a)

	def analytic_traj(self, S=.1, T=4., gmm=acos(Config.VD/Config.VI_FAST)+0.1, D=0, delta=0.099999, n=50):
		assert S >= self.s_lb
		assert gmm >= self.gmm_lb
		xs, _ = envelope_traj(S, T, gmm, D, delta, n=n)
		with open(self._res_dir+'analytic_traj_param.csv', 'a') as f:
			f.write('S,%.3f\n'%S)
			f.write('T,%.3f\n'%T)
			f.write('gmm,%.3f\n'%gmm)
			f.write('D,%.3f\n'%D)
			f.write('de;ta,%.3f\n'%delta)
		return xs*0.1

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
	def __init__(self, target, res_dir, ni, nd):
		super(FastDgame, self).__init__(target, type='fastD', res_dir=res_dir, ni=ni, nd=nd)

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

	game = SlowDgame(line, res_dir='res1/', ni=1, nd=2)
	xs_ref = game.analytic_traj()
	game.reset([xs_ref[0, 2:4]], [xs_ref[0, :2], xs_ref[0, 4:]])
	xs_ref = np.array([xs_ref[:,2:4]]), np.array([xs_ref[:,:2], xs_ref[:,4:]])
	xs_play = game.advance(7.)
	xs_exp = game.replay_exp()
	# xis, xds = game.convert_data(xis), game.convert_data(xds)
	# xs_play = np.concatenate((xds[0], xis[0], xds[1]),-1)
	game.plotter.plot({'play':xs_play, 'ref':xs_ref, 'exp':xs_exp})
	# game.plotter.plot({'exp':xs_exp})

	# game = FastDgame(line, ni=1, nd=2)
	# game.reset([np.array([-1., 9.])], [np.array([-8., 6.]), np.array([6., 5.])])
	# xs_play = game.advance(10.)
	# xs_exp  = game.replay_exp()
	# # xis, xds = game.convert_data(xis), game.convert_data(xds)
	# ## xs_play = np.concatenate((xds[0], xis[0], xds[1]),-1)
	# game.plotter.plot({'play':xs_play, 'exp':xs_exp})			
		
		
						