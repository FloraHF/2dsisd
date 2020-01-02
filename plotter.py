import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

import numpy as np
from scipy.interpolate import interp1d
from math import pi, sin, cos

from geometries import DominantRegion

class Plotter(object):
	"""docstring for Plotter"""
	def __init__(self, game, target, a, r):

		self.fig, self.ax = plt.subplots()
		self.linestyles = {'play': (0, ()), 'ref':(0, (6, 3)), 'exp':(0, ())}
		self.colors = {'D0': 'g', 'I0': 'r', 'D1': 'b'}
		self.target_specs = {'line':(0, ()), 'color':'k'}
		self.dcontour_specs = {'line':(0, ()), 'color':'k'}
		self.barrier_specs = {'line':(0, ()), 'color':'r'}
		self.xlim = [-.7, .8]
		self.ylim = [-1.3, 1.]

		self.game = game
		self.r = game.r
		self.a = game.a
		self.target = game.target
		self.target_level_0 = self.get_target()
		# print(self.target.y0)

	def get_sim_barrier_data(self):
		xy = []
		with open('exp_results/sim_barrier.csv', 'r') as f:
			lines = f.readlines()[1:]
			for line in lines:
				data = line.split(',')
				xy.append(np.array([float(data[0]), float(data[1])]))
				xy.append(np.array([-float(data[0]), float(data[1])]))
		return np.asarray(sorted(xy, key=lambda xys: xys[0]))

	def get_exp_barrier_data_asarray(self):
		x, y, cap = [], [], []
		with open('exp_results/exp_barrier_counted.csv', 'r') as f:
			lines = f.readlines()[1:]
			for line in lines:
				data = line.split(',')
				x.append(float(data[0]))
				y.append(float(data[1]))
				cap.append(float(data[-1]))

		return x, y, cap

	def get_exp_barrier_data_approx(self):

		xy = []
		with open('exp_results/exp_barrier_comp.csv', 'r') as f:
			lines = f.readlines()
			for line in lines:
				data = line.split(',')
				xy.append(([float(data[0].rstrip()), float(data[1].rstrip())]))
		return np.asarray(xy)

	def plot_exp_barrier_scatter(self, rotate=False, size=80):
		xs, ys, caps = self.get_exp_barrier_data_asarray()

		mcap, ment = [], []
		xcap, ycap = [], []
		xent, yent = [], []

		for i, (x, y, cap) in enumerate(zip(xs, ys, caps)):
			if cap > 0.01:
				x_ = [0] + np.cos(np.linspace(0, 2*np.pi*cap, 15)).tolist()
				y_ = [0] + np.sin(np.linspace(0, 2*np.pi*cap, 15)).tolist()
				mcap.append(np.column_stack([x_, y_]))
				if rotate:
					xcap.append(-y)
					ycap.append(x)
				else:
					xcap.append(x)
					ycap.append(y)
			if cap < 0.99:	
				x_ = [0] + np.cos(np.linspace(2*np.pi*cap, 2*np.pi, 15)).tolist()
				y_ = [0] + np.sin(np.linspace(2*np.pi*cap, 2*np.pi, 15)).tolist()
				ment.append(np.column_stack([x_, y_]))
				if rotate:
					xent.append(-y)
					yent.append(x)
				else:
					xent.append(x)
					yent.append(y)

		for i, (m, x, y) in enumerate(zip(mcap, xcap, ycap)):
			self.ax.scatter(x, y, marker=m, s=size, color='g', alpha=0.8)
		for (m, x, y) in zip(ment, xent, yent):
			self.ax.scatter(x, y, marker=m, s=size, color='red', alpha=0.8)

		x_ = [0] + np.cos(np.linspace(0, 2*np.pi*0.4, 15)).tolist()
		y_ = [0] + np.sin(np.linspace(0, 2*np.pi*0.4, 15)).tolist()
		m_ = np.column_stack([x_, y_])
		icon_cap = self.ax.scatter(0, -1, marker=m_, s=size, color='g', alpha=0.8)
		x_ = [0] + np.cos(np.linspace(2*np.pi*0.4, 2*np.pi, 15)).tolist()
		y_ = [0] + np.sin(np.linspace(2*np.pi*0.4, 2*np.pi, 15)).tolist()
		m_ = np.column_stack([x_, y_])
		icon_ent = self.ax.scatter(0, -1, marker=m_, s=size, color='r', alpha=0.8)

		return icon_cap, icon_ent

	def plot_exp_barrier_line(self, rotate=False):
		if rotate:
			xy = self.game.rotate_to_exp_point(self.get_exp_barrier_data_approx())
		else:			
			xy = self.get_exp_barrier_data_approx()
		self.ax.plot(xy[:,0], xy[:,1], color='r', linestyle=(0, ()),label='barrier exp')

	def plot_sim_barrier_line(self, rotate=False):
		if rotate:
			xy = self.game.rotate_to_exp_point(self.get_sim_barrier_data())
		else:
			xy = self.get_sim_barrier_data()
		self.ax.plot(xy[:,0], xy[:,1], color='r', linestyle=(0, (6, 1)), label='barrier sim')

	def plot_barrier(self, rotate=False):
		icon = self.plot_exp_barrier_scatter(rotate=rotate, size=50)
		# self.plot_exp_barrier_line(rotate=rotate)
		self.plot_sim_barrier_line(rotate=rotate)
		for role, p in self.game.players.items():
			if 'D' in role:
				print(p.x)
				label = None
				if role == 'D0':
					label = 'defender'
				if rotate:
					self.plot_capture_ring(role, 'play', self.game.rotate_to_exp_point(p.x), buff=True, label=label)
				else:
					self.plot_capture_ring(role, 'play', p.x, buff=True, label=label)
		if rotate: # target
			self.ax.plot([0.5, 0.5], [-1.2, 1.2], 'k', label='target', zorder=100)
		else:
			self.ax.plot([-1.2, 1.2], [-.5, -.5], 'k', label='target', zorder=100)
		# self.ax.set_xlim([-.65, .1])
		# self.ax.set_ylim([.2, .8]) # slowD, small
		self.ax.set_xlim([-1.15, 1.15])
		self.ax.set_ylim([-.55, .8]) # slowD, large
		# self.ax.set_xlim([-.65, .05])
		# self.ax.set_ylim([.35, .5]) # fastD, small
		self.ax.set_aspect('equal')
		self.ax.grid()
		self.ax.set_axisbelow(True)
		self.ax.tick_params(axis='both', which='major', labelsize=12)
		plt.xlabel('x(m)', fontsize=12)
		plt.ylabel('y(m)', fontsize=12)
		handles, labels = self.ax.get_legend_handles_labels()
		handles.append(icon)
		labels.append('capture rate')

		plt.legend(handles, labels, prop={'size': 11}, ncol=2, loc='lower center')
		plt.show()

	def get_data(self, fn, midx=0, midy=0, kx=1., ky=1., n=80):
		x = np.linspace(midx+kx*self.xlim[0], midx+kx*self.xlim[1], n)
		y = np.linspace(midy+ky*self.ylim[0], midy+ky*self.ylim[1], n)
		X, Y = np.meshgrid(x, y)
		D = np.zeros(np.shape(X))
		for i, (xx, yy) in enumerate(zip(X, Y)):
			# print(i)
			for j, (xxx, yyy) in enumerate(zip(xx, yy)):
				D[i,j] = fn(np.array([xxx, yyy]))
		return {'X': X, 'Y': Y, 'data': D}

	def get_target(self, n=60):
		if self.target.type == 'line':
			kx, ky = 1.2, 1.
		else:
			kx, ky = 3., 3.
		data = self.get_data(self.target.level, midx=0., midy=-.5, kx=kx, ky=ky, n=n)
		data = plt.contour(data['X'], data['Y'], data['data'], [0]).allsegs[0][0]
		self.reset()
		return self.game.rotate_to_exp_point(data)

	def plot_Iwin(self, xi, xds):

		def get_Iwin(x, xds=xds):
			D1_I, D2_I, D1_D2 = self.game.get_vecs({'I0':x, 'D0':xds[0], 'D1':xds[1]})
			d1, d2, a1, a2 = self.game.get_alpha(D1_I, D2_I, D1_D2)
			tht = self.game.get_theta(D1_I, D2_I, D1_D2)
			return tht - (a1 + a2) - (pi - 2*self.game.analytic_traj.gmm)
		Iwin = self.get_data(get_Iwin, midx=xi[0], midy=xi[1], kx=10, ky=10)
		CT = self.ax.contour(Iwin['X'], Iwin['Y'], Iwin['data'], [0], linestyles=(self.target_specs['line'],))
		plt.contour(CT, levels = [0], colors=(self.target_specs['color'],), linestyles=(self.target_specs['line'],))

	def plot_target(self, n=50):
		self.ax.plot(self.target_level_0[:,0], self.target_level_0[:,1], color=self.target_specs['color'], linestyle=self.target_specs['line'])

	def plot_dr(self, xi, xds, ind=True):
		k = 4.
		nd = len(xds)

		dr_comb = DominantRegion(self.r, self.a, xi, xds)
		if ind and nd > 1:
			for p, c in self.colors.items():
				if 'D' in p:
					i = int(p[-1])
					if i < nd:
						xd = xds[i]
						# print(i, xd)
						dr_ind = DominantRegion(self.r, self.a, xi, [xd])
						dr = self.get_data(dr_ind.level, kx=k, ky=k)
						CD = self.ax.contour(dr['X'], dr['Y'], dr['data'], [0], linestyles='dashed')
						plt.contour(CD, levels = [0], colors=(self.colors[p],), linestyles=('dashed',))
		# dr unioned
		dr = self.get_data(dr_comb.level, midx=xi[0], midy=xi[1], kx=k, ky=k)
		CD = self.ax.contour(dr['X'], dr['Y'], dr['data'], [0], linestyles='solid')
		plt.contour(CD, levels = [0], colors=(self.colors['I0'],), linestyles=('solid',))
		# locations of players
		# self.ax.plot(xi[0], xi[1], 'x', color=self.colors['I0'], zorder=100)
		self.ax.plot(xi[0], xi[1], 'x', color='k', zorder=100)
		for p, c in self.colors.items():
			if 'D' in p:
				i = int(p[-1])
				if i < nd:
					# self.ax.plot(xds[i][0], xds[i][1], 'x', color=self.colors[p], zorder=100)
					self.ax.plot(xds[i][0], xds[i][1], 'x', color='k', zorder=100)

		xt = self.game.deepest_in_target({'I0':xi, 'D0':xds[0], 'D1':xds[1]})
		# print(self.game.target.level(xt))
		# self.ax.plot(xt[0], xt[1], '<', color='k', zorder=200)

	def plot_dcontour(self, xi, xds, levels=[0.]):

		def get_constd(x):
			xt = self.target.deepest_point_in_dr(DominantRegion(self.r, self.a, x, xds))
			return self.target.level(xt)
		# print(get_constd(xi))

		vctr = self.get_data(get_constd, midx=xi[0], midy=xi[1], kx=5., ky=5.)
		# print(vctr['data'])
		CC = self.ax.contour(vctr['X'], vctr['Y'], vctr['data'], [0], linestyles=(self.dcontour_specs['line'],))
		self.ax.clabel(CC, inline=True, fontsize=10)
		self.ax.contour(vctr['X'], vctr['Y'], vctr['data'], levels=levels, colors=(self.dcontour_specs['color'],), linestyles=(self.dcontour_specs['line'],))
		self.plot_capture_ring('D0', None, xds[0])
		self.plot_capture_ring('D1', None, xds[1])

	def plot_capture_ring(self, player, situation, x, buff=False, n=50, label=None):
		xs, ys = [], []
		for t in np.linspace(0, 2*pi, n):
			xs.append(x[0] + self.r*cos(t))
			ys.append(x[1] + self.r*sin(t))

		self.ax.plot(x[0], x[1], 'o', color=self.colors[player], label=label)

		if situation is None or situation=='ref':
			self.ax.plot(xs, ys, color=self.colors[player], linestyle=(0, (5, 5)))
			ring = Circle((x[0], x[1]), self.r, fc=self.colors[player], ec=self.colors[player], alpha=0.1, label=None)
			self.ax.add_patch(ring)
		else:
			self.ax.plot(xs, ys, color=self.colors[player], linestyle=self.linestyles[situation])
			ring = Circle((x[0], x[1]), self.r, fc=self.colors[player], ec=self.colors[player], alpha=0.6, label=None)
			self.ax.add_patch(ring)

		if buff:
			if self.game.vi >= self.game.vd:
				r = self.game.r_close
			else:
				r = self.r
			ring = Circle((x[0], x[1]), r, fc=self.colors[player], ec=self.colors[player], alpha=0.2, label=None)
			self.ax.add_patch(ring)

	def plot_traj(self, player, situation, xs, label=None):

		if label is not None:
			if situation == 'ref':
				label = player + ': ' + 'sim'
			else:
				label = player + ': ' + 'exp'
			if 'I' in label:
				label = ' '+label
		else:
			label = player
		self.ax.plot(xs[:,0], xs[:,1], color=self.colors[player], linestyle=self.linestyles[situation], label=None)
		self.ax.plot(xs[-1,0], xs[-1,1], 'o', color=self.colors[player], linestyle=self.linestyles[situation], label='_Hidden')

	def plot_connect(self, p1, p2, xs1, xs2, skip=20):
		n = xs1.shape[0]
		for i, (x1, x2) in enumerate(zip(xs1, xs2)):
			if i%skip == 0 or i==n-1:
				self.ax.plot([x1[0], x2[0]], [x1[1], x2[1]], 'b--')
				self.ax.plot(x1[0], x1[1], 'o', color=self.colors[p1])
				self.ax.plot(x2[0], x2[1], 'o', color=self.colors[p2])

	def show_plot(self, fname=None):
		# self.ax.axis('equal')
		self.ax.grid()
		self.ax.tick_params(axis='both', which='major', labelsize=12)
		plt.xlabel('x(m)', fontsize=12)
		plt.ylabel('y(m)', fontsize=12)
		
		plt.gca().legend(prop={'size': 11}, ncol=3, handletextpad=0.1)
		print('set legend')
		self.ax.set_aspect('equal')
		self.ax.set_xlim(self.xlim)
		self.ax.set_ylim(self.ylim)

		if fname is not None:
			self.fig.savefig(self.game.res_dir+fname, bbox_inches='tight')
		else:
			plt.show()
		self.reset()

	def process_policy_labels(self, ps):
		labels = dict()
		if ps is not None:
			for role, lbl in ps.items():
				labels[role] = lbl 
		else:
			for role in self.game.players:
				labels[role] = None
		return labels

	def plot(self, xs, geox='', ps=None, traj=True, dr=False, ndr=0, dcontour=False, barrier=False, fname=None):
		self.fig, self.ax = plt.subplots()

		for situ in xs:
			xs[situ] = self.game.rotate_to_exp(xs[situ])
		
		self.reset()
		self.plot_target()
		icon_D0, = self.ax.plot([], [], 'o', color=self.colors['D0'], label='D0')
		icon_D1, = self.ax.plot([], [], 'o', color=self.colors['D1'], label='D1')
		icon_I, = self.ax.plot([], [], 'o', color=self.colors['I0'], label='I')

		if traj:

			plot_geo = False
			ps = self.process_policy_labels(ps)

			for situ, x in xs.items():
				for pid, px in x.items():
					self.plot_traj(pid, situ, px, label=ps[pid])
					if 'D' in pid:
						self.plot_capture_ring(pid, situ, px[-1, :])
				if situ == geox:
					plot_geo = True
					# self.plot_connect('I0', 'D0', x['I0'], x['D0'])
					# self.plot_connect('I0', 'D1', x['I0'], x['D1'])
		else:
			xlim, ylim = None, None
			plot_geo = True

		if plot_geo:
			# print(ndr)
			xi0 = xs[geox]['I0'][ndr, :]
			xd0s = [xs[geox]['D0'][ndr, :], xs[geox]['D1'][ndr, :]]
			if dr:
				if self.game.a >= 1:
					self.plot_dr(xi0, xd0s, ind=True)
				else:
					self.plot_dr(xi0, xd0s, ind=True)
					# self.plot_Iwin(xi0, xd0s)
			if dcontour:
				self.plot_dcontour(xi0, xd0s)

		if barrier:
			self.plot_barrier_scatter()
			self.plot_barrier_line()

		self.show_plot(fname=fname)

	def animate(self, ts, xs, ps=None, xrs=None, linestyle=(0, ()), label='', alpha=0.5):
		xs = self.game.rotate_to_exp(xs)
		xrs = self.game.rotate_to_exp(xrs)
		# print(xs)
		ps = self.process_policy_labels(ps)

		# print(xs['D0'])
		n = xs['D0'].shape[0]
		if ts is None:
			ts = np.linspace(0, 5, n)
		ts = ts - np.amin(ts)
		tail = int(n/5)

		self.ax.set_xlim(self.xlim)
		self.ax.set_ylim((-1.3, 1.3))

		self.plot_target()
		if xrs is not None:
			for pid, px in xrs.items():
				# self.plot_traj(pid, 'ref', px, label=ps[pid])
				self.plot_traj(pid, 'ref', px, label=None)
				if 'D' in pid:
					self.plot_capture_ring(pid, None, px[-1, :])

		time_template = 'time = %.1fs'
		time_text = self.ax.text(0.05, .94, '', transform=self.ax.transAxes, fontsize=11)
		plots = dict()
		# plots['target'], = self.ax.plot([], [], color=self.target_specs['color'], linestyle=self.target_specs['line'], label='target')
		plots['D0'], = self.ax.plot([], [], 'o', color=self.colors['D0'], label='D0')
		plots['D1'], = self.ax.plot([], [], 'o', color=self.colors['D1'], label='D1')
		plots['I0'], = self.ax.plot([], [], 'o', color=self.colors['I0'], label='I')
		for role in self.game.players:
			plots[role+'tail'], = self.ax.plot([], [], linewidth=2, color=self.colors[role], linestyle=linestyle, label=None)
			# if ps[role] is None:
			# 	plots[role+'tail'], = self.ax.plot([], [], linewidth=2, color=self.colors[role], linestyle=linestyle, label=role)
			# else:
			# 	if 'I' in role:
			# 		plots[role+'tail'], = self.ax.plot([], [], linewidth=2, color=self.colors[role], linestyle=linestyle, label=' '+role+': '+ps[role])
			# 	else:
			# 		plots[role+'tail'], = self.ax.plot([], [], linewidth=2, color=self.colors[role], linestyle=linestyle, label=role+': '+ps[role])

		# plots['Dline'], = self.ax.plot([], [], '--', color='b', label=None)
		plots['D0cap'] = Circle((0, 0), self.r, fc=self.colors['D0'], ec=self.colors['D0'], alpha=alpha, label=None)
		plots['D1cap'] = Circle((0, 0), self.r, fc=self.colors['D1'], ec=self.colors['D1'], alpha=alpha, label=None)
		self.ax.add_patch(plots['D0cap'])
		self.ax.add_patch(plots['D1cap'])

		self.ax.set_aspect('equal')
		self.ax.grid()
		self.ax.tick_params(axis='both', which='major', labelsize=14)
		plt.xlabel('x(m)', fontsize=14)
		plt.ylabel('y(m)', fontsize=14)

		def init():
			time_text.set_text('')
			# plots['target'].set_data([], [])
			for role, x in xs.items():
				# print(role)
				plots[role].set_data([], [])
				plots[role+'tail'].set_data([], [])
				if 'D' in role:
					plots[role+'cap'].center = (x[0,0], x[0,1])
			# plots['Dline'].set_data([], [])	

			return plots['D0'], plots['D1'], \
					plots['D0cap'], plots['D1cap'], \
					plots['I0'], \
					plots['D0tail'], plots['D1tail'], plots['I0tail'], \
					time_text

		def animate_traj(i):
			i = i%n
			ii = np.clip(i-tail, 0, i)
			# plots['target'].set_data(self.target_level_0[:,0], self.target_level_0[:,1])
			time_text.set_text(time_template % (ts[i]))
			for role, x in xs.items():
				plots[role].set_data(x[i,0], x[i,1])
				plots[role+'tail'].set_data(x[ii:i+1,0], x[ii:i+1,1])
				if 'D' in role:
					plots[role+'cap'].center = (x[i,0], x[i,1])
			# plots['Dline'].set_data([xs['D0'][i,0], xs['D1'][i,0]], [xs['D0'][i,1], xs['D1'][i,1]])	
			# plt.gca().legend(prop={'size': 12}, ncol=2)
			return  plots['D0'], plots['D1'], \
					plots['D0cap'], plots['D1cap'], \
					plots['I0'], \
					plots['D0tail'], plots['D1tail'], plots['I0tail'], \
					time_text

		ani = animation.FuncAnimation(self.fig, animate_traj, init_func=init, interval=self.game.dt*1000)
		plt.gca().legend(prop={'size': 11}, ncol=3, loc='lower center', handletextpad=0.1)
		plt.gcf().set_size_inches(4,5)
		self.fig.tight_layout()
		ani.save(self.game.res_dir+'ani_traj.mp4')

	def plot_velocity(self):
		t_start, t_end = -1., 10000.,
		for role, p in self.game.players.items():
			if p.exp.t_start > t_start:
				t_start = p.exp.t_start
			if p.exp.t_end < t_end:
				t_end = p.exp.t_end

		n = 1000
		ts = np.linspace(t_start-1, t_end, n)
		vs = {'D0': np.zeros(n),'I0': np.zeros(n), 'D1': np.zeros(n)}
		dis = {'D0': np.zeros(n), 'D1': np.zeros(n), 'target': np.zeros(n)}
		dcolor = {'D0': self.colors['D0'], 'D1': self.colors['D1'], 'target': self.target_specs['color']}
		a_cal = np.zeros(n)
		a_exp = np.zeros(n)
		a_sim = np.ones(n)*(self.game.a)

		fig, axs = plt.subplots(2, 1)
		for j, t in enumerate(ts):
			xi = self.game.players['I0'].exp.x(t)
			yi = self.game.players['I0'].exp.y(t)
			xd0 = self.game.players['D0'].exp.x(t)
			yd0 = self.game.players['D0'].exp.y(t)
			xd1 = self.game.players['D1'].exp.x(t)
			yd1 = self.game.players['D1'].exp.y(t)
			dis['D0'][j] = np.sqrt((yi - yd0)**2 + (xi - xd0)**2)
			dis['D1'][j] = np.sqrt((yi - yd1)**2 + (xi - xd1)**2)
			dis['target'][j] = yi + 0.5
			# a_exp[j] = self.game.players['I0'].exp.a(t)
			for role, player in self.game.players.items():
				vx = player.exp.vx(t)
				vy = player.exp.vy(t)
				v = np.sqrt(vx**2 + vy**2)
				vs[role][j] = v
			a_cal[j] = (vs['D0'][j] + vs['D1'][j])/(2*vs['I0'][j])
		
		# axs[2].plot(ts, a_cal, color='k', label='exp')
		# axs[2].plot(ts, a_exp, color='b', label='exp')
		# axs[2].plot(ts, a_sim, color='r', label='des')
		for role, player in self.game.players.items():
			if 'I' in role:
				axs[0].plot(ts, vs[role], color=self.colors[role], label='I')
			if 'D' in role:
				axs[0].plot(ts, vs[role], color=self.colors[role], label=role)

		for r, data in dis.items():
			axs[1].plot(ts, data, color=dcolor[r], label=r)

		axs[0].set_ylabel('velocity(m/s)')
		axs[1].set_ylabel('distance(m)')
		axs[1].set_xlabel('time(s)')
		# axs[2].set_ylabel('a')
		axs[0].set_ylim(0.24, 0.28)
		# axs[1].set_ylim(0.245, 0.254)
		# axs[2].set_ylim(0.99, 1.1)
		# trange = (27.1, 30.1)
		# axs[0].set_xlim(trange)
		# axs[1].set_xlim(trange)
		# axs[2].set_xlim(trange)
		for ax in axs:
			ax.grid(True)
			ax.legend(ncol=2)
		fig.tight_layout()
		plt.show()

	# def plot_dist(self):
	# 	t_start, t_end = -1., 10000.,
	# 	for role, p in self.game.players.items():
	# 		if p.exp.t_start > t_start:
	# 			t_start = p.exp.t_start
	# 		if p.exp.t_end < t_end:
	# 			t_end = p.exp.t_end

	# 	n = 1000
	# 	ts = np.linspace(t_start, t_end, n)
	# 	dis = {'D0': np.zeros(n), 'D1': np.zeros(n), 'target': np.zeros(n)}

	# 	fig, axs = plt.subplots(3, 1)
	# 	for j, t in enumerate(ts):
	# 		# a_exp[j] = self.game.players['I0'].exp.a(t)
	# 		xi = self.game.players['I0'].exp.x(t)
	# 		yi = self.game.players['I0'].exp.y(t)
	# 		xd0 = self.game.players['D0'].exp.x(t)
	# 		yd0 = self.game.players['D0'].exp.y(t)
	# 		xd1 = self.game.players['D1'].exp.x(t)
	# 		yd1 = self.game.players['D1'].exp.y(t)
	# 		dis['D0'][j] = np.sqrt((yi - yd1)**2 + (xi - xd1)**2)
	# 		dis['D1'][j] = np.sqrt((yi - yd1)**2 + (xi - xd1)**2)
	# 		dis['target'][j] = yi
		
	# 	axs[2].plot(ts, a_cal, color='k', label='exp')
	# 	# axs[2].plot(ts, a_exp, color='b', label='exp')
	# 	axs[2].plot(ts, a_sim, color='r', label='des')
	# 	for role, player in self.game.players.items():
	# 		if 'I' in role:
	# 			axs[0].plot(ts, vs[role], color=self.colors[role], label=role)
	# 		if 'D' in role:
	# 			axs[1].plot(ts, vs[role], color=self.colors[role], label=role)

	# 	axs[0].set_ylabel('velocity(m/s)')
	# 	axs[1].set_ylabel('velocity(m/s)')
	# 	axs[2].set_xlabel('time(s)')
	# 	axs[2].set_ylabel('a')
	# 	axs[0].set_ylim(0.23, 0.25)
	# 	axs[1].set_ylim(0.245, 0.254)
	# 	axs[2].set_ylim(0.99, 1.1)
	# 	trange = (19.4, 22.9)
	# 	axs[0].set_xlim(trange)
	# 	axs[1].set_xlim(trange)
	# 	axs[2].set_xlim(trange)
	# 	for ax in axs:
	# 		ax.grid(True)
	# 		ax.legend(ncol=2)
	# 	fig.tight_layout()
	# 	plt.show()

	def reset(self):
		self.ax.clear()