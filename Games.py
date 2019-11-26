import os
import numpy as np
from copy import deepcopy
from math import asin, acos, atan2, sin, cos, pi, sqrt

from tensorflow.keras.models import load_model

from geometries import LineTarget, DominantRegion
from experiment_replay import ReplayPool
from plotter import Plotter
from envelope import Envelope
# from strategies_fastD import deep_target_strategy


class Player(object):

	def __init__(self, env, role, res_dir=None):

		self.env = env
		self.role = role
		self.dt = env.dt
		if res_dir is not None:
			self.exp = ReplayPool(self.role, res_dir=res_dir)
			self.x = np.array([self.exp.x(self.exp.t_start), self.exp.y(self.exp.t_start)])
		self.v_curr = np.array([0., 0.])
		
		if 'D' in role:
			self.v_max = env.vd
		elif 'I' in role:
			self.v_max = env.vi

	def step(self, action):
		self.v_curr = self.v_max*np.array([cos(action), sin(action)])
		self.x = self.x + self.dt*self.v_curr
	
	def reset(self, x):
		self.x = deepcopy(x)

	def get_x(self):
		return deepcopy(self.x)

	def get_velocity(self):
		return deepcopy(self.v_curr)


class BaseGame(object):
	"""docstring for BaseGame"""
	def __init__(self, target, gtype, res_dir=None, ni=1, nd=2):

		if res_dir is not None:
			self.res_dir = os.path.dirname(__file__)+'/exp_results/'+res_dir
			with open(self.res_dir+'/info.csv', 'r') as f:
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
					if 'S' in line:
						self.exp_S = float(line.split(',')[-1])
					if 'T' in line:
						self.exp_T = float(line.split(',')[-1])
					if 'gmm' in line:
						self.exp_gmm = float(line.split(',')[-1])
						# print('ub:', self.exp_gmm - acos(self.vd/self.vi))
					if 'D' == line.split(',')[0]:
						self.exp_D = float(line.split(',')[-1])*self.r
					if 'delta' in line:
						delta = float(line.split(',')[-1])
						if delta > self.exp_gmm - acos(self.vd/self.vi):
							delta = self.exp_gmm - acos(self.vd/self.vi)					
						self.exp_delta = delta
		else:
			self.res_dir = os.path.dirname(__file__)+'/test/'
			if not os.path.exists(self.res_dir):
				os.mkdir(self.res_dir)
			with open(os.path.dirname(__file__)+'/config.csv', 'r') as f:
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
		for i in range(nd):
			pid = 'D'+str(i)
			self.players[pid] = Player(self, pid, res_dir=res_dir)
		for i in range(ni):
			pid = 'I'+str(i)
			self.players[pid] = Player(self, pid, res_dir=res_dir)

		self.plotter = Plotter(self, target, self.a, self.r)

	def is_capture(self, xi, xds):
		cap = False
		for xd in xds:
			d = sqrt((xi[0] - xd[0])**2 + (xi[1] - xd[1])**2)
			cap = cap or d < self.r
		return cap

	def get_state(self):
		xs = dict()
		for role, p in self.players.items():
			xs[role] = p.get_x()
		return xs

	def get_velocity(self):
		vs = dict()
		for role, p in self.players.items():
			vs[role] = p.get_velocity()
		return vs

	def step(self, xs, dstrategy, istrategy, close_adjust=True):
		if close_adjust:
			actions = self.c_strategy(xs, dstrategy=dstrategy, istrategy=istrategy)
		else:
			dact = dstrategy(xs)
			iact = istrategy(xs)
			actions = {'D0': dact['D0'], 'D1': dact['D1'], 'I0': iact['I0']}
		# actions = strategy(self, xis, xds)
		for (rp, p), (ra, act) in zip(self.players.items(), actions.items()):
			# print(rp, p, ra, act)
			p.step(act)
		return self.get_state()

	def advance(self, te, dstrategy, istrategy, close_adjust=True):	
		t = 0
		xs0 = self.get_state()
		xs = dict()
		for role, x in xs0.items():
			xs[role] = [x]
		ts = [0]
		xold = xs0
		while t<te:
			xnew = self.step(xold, dstrategy=dstrategy, istrategy=istrategy, close_adjust=close_adjust)
			t += self.dt
			for role, x in xs.items():
				x.append(xnew[role])
			ts.append(t)
			if self.is_capture(xnew['I0'], [xnew['D0'], xnew['D1']]):
				print('capture')
				break
			xold = xnew
		for role, x in xs.items():
			xs[role] = np.asarray(x)
		return np.asarray(ts), xs

	def replay_exp(self):
		t_start, t_end = 100., -1.,
		for role, p in self.players.items():
			if p.exp.t_start < t_start:
				t_start = p.exp.t_start
			if p.exp.t_end > t_end:
				t_end = p.exp.t_end

		t = t_start
		xs = {role:[] for role in self.players}
		ts = []
		while t < t_end:
			ts.append(t)
			for role, p in self.players.items():
				xs[role].append(np.array([p.exp.x(t), p.exp.y(t)]))
			t += self.dt 
		for role, data in xs.items():
			xs[role] = np.asarray(data)

		return np.asarray(ts), xs

	def reset(self, xs):
		for role, p in self.players.items():
			p.reset(xs[role])


class SlowDgame(BaseGame):
	"""docstring for SlowDgame"""
	def __init__(self, target, res_dir, policy_dir='PolicyFn', ni=1, nd=2):
		super(SlowDgame, self).__init__(target, gtype='slowD', res_dir=res_dir, ni=ni, nd=nd)

		self.policies = dict()
		for role, p in self.players.items():
			self.policies[role] = load_model('Policies/'+policy_dir+'_'+role+'.h5')
		self.s_lb = -asin(self.a)
		self.gmm_lb = acos(self.a)
		self.analytic_traj = Envelope(self.vi, self.vd, self.r)

	def generate_analytic_traj(self, S, T, gmm, D, delta, n=50, file='traj_param_100.csv'):
		assert S >= self.s_lb
		assert gmm >= self.gmm_lb
		xs, _ = self.analytic_traj.envelope_traj(S, T, gmm, D, delta, n=n)
		fname = 'params/'+file
		if not os.path.exists('params/'):
			os.mkdir('params/')
		if os.path.exists(fname):
			os.remove(fname)
		with open(fname, 'a') as f:
			f.write('vd,%.3f\n'%self.vd)
			f.write('vi,%.3f\n'%self.vi)
			f.write('rc,%.3f\n'%self.r)
			# f.write('rt,%.3f\n'%self.rt)
			f.write('r_close,%.3f\n'%(self.r_close/self.r))
			f.write('k_close,%.3f\n'%self.k_close)

			f.write('S,%.10f\n'%S)
			f.write('T,%.10f\n'%T)
			f.write('gmm,%.10f\n'%gmm)
			f.write('D,%.10f\n'%D)
			f.write('delta,%.10f\n'%delta)

			f.write('xD0,' + ','.join(list(map(str,xs[0,:2]))) + '\n')
			f.write('xD1,' + ','.join(list(map(str,xs[0,4:]))) + '\n')
			f.write('xI0,' + ','.join(list(map(str,xs[0,2:4]))) + '\n')
		
		return {'D0': xs[:,:2], 'D1': xs[:,4:], 'I0': xs[:,2:4]}

	def reproduce_analytic_traj(self, n=50):
		# from envelope import envelope_traj
		xs, _ = self.analytic_traj.envelope_traj(self.exp_S, self.exp_T, self.exp_gmm, self.exp_D, self.exp_delta, n=n)
		return {'D0': xs[:,:2], 'D1': xs[:,4:], 'I0': xs[:,2:4]}

	def nn_strategy(self, xs):
		acts = dict()
		x = np.concatenate((xs['D0'], xs['I0'], xs['D1']))
		for role, p in self.policies.items():
			acts[role] = p.predict(x[None])[0]
		return acts

	def get_vecs(self, xs):
		D1 = np.concatenate((xs['D0'], [0]))
		D2 = np.concatenate((xs['D1'], [0]))
		I = np.concatenate((xs['I0'], [0]))
		D1_I = I - D1
		D2_I = I - D2
		D1_D2 = D2 - D1
		return D1_I, D2_I, D1_D2
	
	def get_base(self, D1_I, D2_I, D1_D2):
		base_d1 = atan2(D1_I[1], D1_I[0])
		base_d2 = atan2(D2_I[1], D2_I[0])
		base_i = atan2(-D2_I[1], -D2_I[0])
		# print(base_d1*180/pi, base_d2*180/pi, base_i*180/pi)
		return {'D0': base_d1, 'D1': base_d2, 'I0': base_i}

	def get_xyz(self, D1_I, D2_I, D1_D2):
		z = np.linalg.norm(D1_D2)/2
		x = -np.cross(D1_D2, D1_I)[-1]/(2*z)
		y = np.dot(D1_D2, D1_I)/(2*z) - z
		return x, y, z

	def get_theta(self, D1_I, D2_I, D1_D2):
		k1 = atan2(np.cross(D1_D2, D1_I)[-1], np.dot(D1_D2, D1_I))  # angle between D1_D2 to D1_I
		k2 = atan2(np.cross(D1_D2, D2_I)[-1], np.dot(D1_D2, D2_I))  # angle between D1_D2 to D2_I
		tht = k2 - k1
		if k1 < 0:
			tht += 2 * pi
		return tht

	def get_d(self, D1_I, D2_I, D1_D2):
		d1 = max(np.linalg.norm(D1_I), self.r)
		d2 = max(np.linalg.norm(D2_I), self.r)
		return d1, d2

	def get_alpha(self, D1_I, D2_I, D1_D2):
		d1, d2 = self.get_d(D1_I, D2_I, D1_D2)
		a1 = asin(self.r/d1)
		a2 = asin(self.r/d2)
		return d1, d2, a1, a2

	def z_strategy(self, xs):
		D1_I, D2_I, D1_D2 = self.get_vecs(xs)
		base = self.get_base(D1_I, D2_I, D1_D2)
		d1, d2 = self.get_d(D1_I, D2_I, D1_D2)
		tht = self.get_theta(D1_I, D2_I, D1_D2)
		phi_1 = -pi/2
		phi_2 = pi/2
		cpsi = d2 * sin(tht)
		spsi = -(d1 - d2 * cos(tht))
		psi = atan2(spsi, cpsi)
		# print('%.2f, %.2f, %.2f'%(phi_1*180/pi, phi_2*180/pi, psi*180/pi))

		phi_1 += base['D0']
		phi_2 += base['D1']
		psi += base['I0']

		# print('%.2f, %.2f, %.2f'%(phi_1*180/pi, phi_2*180/pi, psi*180/pi))

		return {'D0': phi_1, 'D1': phi_2, 'I0': psi}

	def h_strategy(self, xs):
		D1_I, D2_I, D1_D2 = self.get_vecs(xs)
		base = self.get_base(D1_I, D2_I, D1_D2)
		x, y, z = self.get_xyz(D1_I, D2_I, D1_D2)

		x_ = {'D1': np.array([0, -z]),
			  'D2': np.array([0, z]),
			  'I': np.array([x, y])}

		Delta = sqrt(np.maximum(x ** 2 - (1 - 1/self.a**2)*(x**2 + y**2 - (z/self.a)**2), 0))
		if (x + Delta) / (1 - 1/self.a ** 2) - x > 0:
			xP = (x + Delta)/(1 - 1/self.a**2)
		else:
			xP = -(x + Delta)/(1 - 1/self.a**2)

		P = np.array([xP, 0, 0])
		D1_ = np.concatenate((x_['D1'], [0]))
		D2_ = np.concatenate((x_['D2'], [0]))
		I_ = np.concatenate((x_['I'], [0]))
		D1_P = P - D1_
		D2_P = P - D2_
		I_P = P - I_
		D1_I_ = I_ - D1_
		D2_I_ = I_ - D2_
		D1_D2_ = D2_ - D1_

		phi_1 = atan2(np.cross(D1_I_, D1_P)[-1], np.dot(D1_I_, D1_P))
		phi_2 = atan2(np.cross(D2_I_, D2_P)[-1], np.dot(D2_I_, D2_P))
		psi = atan2(np.cross(-D2_I_, I_P)[-1], np.dot(-D2_I_, I_P))
		# print(phi_1, phi_2, psi)
		phi_1 += base['D0']
		phi_2 += base['D1']
		psi += base['I0']

		return {'D0': phi_1, 'D1': phi_2, 'I0': psi}

	def i_strategy(self, xs):
		
		D1_I, D2_I, D1_D2 = self.get_vecs(xs)
		base = self.get_base(D1_I, D2_I, D1_D2)
		d1, d2, a1, a2 = self.get_alpha(D1_I, D2_I, D1_D2)
		tht = self.get_theta(D1_I, D2_I, D1_D2)
		
		LB = acos(self.a)
		
		phi_2 = pi/2 - a2 + 0.01
		psi = -(pi/2 - LB + a2)
		d = d2*(sin(phi_2)/sin(LB))
		l1 = sqrt(d1**2 + d**2 - 2*d1*d*cos(tht + psi))
		cA = (d**2 + l1**2 - d1**2)/(2*d*l1)
		sA = sin(tht + psi)*(d1/l1)
		A = atan2(sA, cA)
		phi_1 = -(pi - (tht + psi) - A) + base['D0']
		phi_2 = pi/2 - a2 + base['D1']
		psi = -(pi/2 - LB + a2) + base['I0']
		return {'D0': phi_1, 'D1': phi_2, 'I0': psi}

	def m_strategy(self, xs):
		D1_I, D2_I, D1_D2 = self.get_vecs(xs)
		x, y, z = self.get_xyz(D1_I, D2_I, D1_D2)
		if x < 0.1:
			return self.h_strategy(xs)
		else:
			return self.i_strategy(xs)

	def c_strategy(self, xs, dstrategy=m_strategy, istrategy=m_strategy):
		D1_I, D2_I, D1_D2 = self.get_vecs(xs)
		base = self.get_base(D1_I, D2_I, D1_D2)
		vs = self.get_velocity()
		
		#=============== both defenders are close ===============#
		if np.linalg.norm(D1_I) < self.r_close and np.linalg.norm(D2_I) < self.r_close:  # in both range
			vD1 = np.concatenate((vs['D0'], [0]))
			phi_1 = atan2(np.cross(D1_I, vD1)[-1], np.dot(D1_I, vD1))
			phi_1 = self.k_close*phi_1 + base['D0']

			vD2 = np.concatenate((vs['D1'], [0]))
			phi_2 = atan2(np.cross(D2_I, vD2)[-1], np.dot(D2_I, vD2))
			phi_2 = self.k_close*phi_2 + base['D1']		 
			
			psi = -self.get_theta(D1_I, D2_I, D1_D2)/2 + base['I0']

			return {'D0': phi_1, 'D1': phi_2, 'I0': psi}

		#=============== only D1 is close ===============#
		elif np.linalg.norm(D1_I) < self.r_close:  # in D1's range
			# print(vs['D0'])
			vD1 = np.concatenate((vs['D0'], [0]))
			phi_1 = atan2(np.cross(D1_I, vD1)[-1], np.dot(D1_I, vD1))
			phi_1 = self.k_close*phi_1 + base['D0']

			phi_2 = dstrategy(xs)['D1']

			psi = - abs(acos(self.vd*cos(phi_1)/self.vi))
			psi = pi - self.get_theta(D1_I, D2_I, D1_D2) + psi + base['I0']

			return {'D0': phi_1, 'D1': phi_2, 'I0': psi}

		#=============== only D2 is close ===============#
		elif np.linalg.norm(D2_I) < self.r_close:
			vD2 = np.concatenate((vs['D1'], [0]))
			phi_2 = atan2(np.cross(D2_I, vD2)[-1], np.dot(D2_I, vD2))
			phi_1 = dstrategy(xs)['D0']

			phi_2 = self.k_close * phi_2 + base['D1']

			psi = abs(acos(self.vd * cos(phi_2)/self.vi))
			psi = psi - pi + base['I0']

			return {'D0': phi_1, 'D1': phi_2, 'I0': psi}

		#============== no defender is close =============#
		else:
			dact = dstrategy(xs)
			iact = istrategy(xs)
			return {'D0': dact['D0'], 'D1': dact['D1'], 'I0': iact['I0']}

class FastDgame(BaseGame):
	"""docstring for FastDgame"""
	def __init__(self, target, res_dir, ni, nd):
		super(FastDgame, self).__init__(target, gtype='fastD', res_dir=res_dir, ni=ni, nd=nd)

	def deep_target_strategy(self, xi, xds):
		dr = DominantRegion(self.r, self.a, xi, xds)
		xt = self.target.deepest_point_in_dr(dr)

		IT = np.concatenate((xt - xi, np.zeros((1,))))
		DTs = []
		for xd in xds:
			DT = np.concatenate((xt - xd, np.zeros(1,)))
			DTs.append(DT)
		xaxis = np.array([1, 0, 0])

		psi = atan2(np.cross(xaxis, IT)[-1], np.dot(xaxis, IT))
		phis = []
		for DT in DTs:
			phi = atan2(np.cross(xaxis, DT)[-1], np.dot(xaxis, DT))
			phis.append(phi)

		return [psi], phis

	def f_strategy(self, xs):
		psis, phis = self.deep_target_strategy(xs['I0'], (xs['D0'], xs['D1']))
		# print(psis, phis)
		acts = dict()
		for role, p in self.players.items():
			if 'D' in role:
				acts[role] = phis[int(role[-1])]
			elif 'I' in role:
				acts[role] = psis[int(role[-1])]
		return acts
		
