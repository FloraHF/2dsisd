import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

import numpy as np
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
		self.xlim = [-r, r]
		self.ylim = [-r, r]

		self.game = game
		self.r = game.r
		self.a = game.a
		self.target = game.target

	def get_data(self, fn, midx=0, midy=0, kx=1., ky=1., n=50):
		x = np.linspace(midx+kx*self.xlim[0], midx+kx*self.xlim[1], n)
		y = np.linspace(midy+ky*self.ylim[0], midy+ky*self.ylim[1], n)
		X, Y = np.meshgrid(x, y)
		D = np.zeros(np.shape(X))
		for i, (xx, yy) in enumerate(zip(X, Y)):
			for j, (xxx, yyy) in enumerate(zip(xx, yy)):
				D[i,j] = fn(np.array([xxx, yyy]))
		return {'X': X, 'Y': Y, 'data': D}

	def plot_target(self, n=50):
		if self.target.type == 'line':
			kx, ky = 2.5, 1.
		else:
			kx, ky = 3., 3.
		tgt = self.get_data(self.target.level, kx=kx, ky=ky)
		CT = self.ax.contour(tgt['X'], tgt['Y'], tgt['data'], [0], linestyles=(self.target_specs['line'],))
		plt.contour(CT, levels = [0], colors=(self.target_specs['color'],), linestyles=(self.target_specs['line'],))

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

		xt = self.game.deepest_in_target(xi, xds)
		print(self.game.target.level(xt))
		self.ax.plot(xt[0], xt[1], '<', color='k', zorder=200)

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
	
	def plot_capture_ring(self, player, situation, x, n=50):
		xs, ys = [], []
		for t in np.linspace(0, 2*pi, n):
			xs.append(x[0] + self.r*cos(t))
			ys.append(x[1] + self.r*sin(t))
		self.ax.plot(xs, ys, color=self.colors[player], linestyle=self.linestyles[situation])

	def plot_traj(self, player, situation, xs, label=None):
		if label is not None:
			if situation == 'ref':
				label = player + ': ' + 'ref'
			else:
				label = player + ': ' + label
			if 'I' in label:
				label = ' '+label
		else:
			label = player
		self.ax.plot(xs[:,0], xs[:,1], color=self.colors[player], linestyle=self.linestyles[situation], label=label)
		self.ax.plot(xs[-1,0], xs[-1,1], 'o', color=self.colors[player], linestyle=self.linestyles[situation], label='_Hidden')

	def plot_connect(self, p1, p2, xs1, xs2, skip=20):
		n = xs1.shape[0]
		for i, (x1, x2) in enumerate(zip(xs1, xs2)):
			if i%skip == 0 or i==n-1:
				self.ax.plot([x1[0], x2[0]], [x1[1], x2[1]], 'b--')
				self.ax.plot(x1[0], x1[1], 'o', color=self.colors[p1])
				self.ax.plot(x2[0], x2[1], 'o', color=self.colors[p2])

	def show_plot(self, fname=None):
		self.ax.axis('equal')
		self.ax.grid()
		self.ax.tick_params(axis='both', which='major', labelsize=14)
		plt.xlabel('x(m)', fontsize=14)
		plt.ylabel('y(m)', fontsize=14)
		plt.gca().legend(prop={'size': 12}, ncol=2)
		if fname is not None:
			self.fig.savefig(self.game.res_dir+fname)
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

	def plot(self, xs, geox, ps=None, dr=False, ndr=0, dcontour=False, fname=None):
		# ps: policies dict: {'D0': , 'D1': , 'I': }
		ps = self.process_policy_labels(ps)

		self.plot_target()
		for situ, x in xs.items():
			for pid, px in x.items():
				self.plot_traj(pid, situ, px, label=ps[pid])
				if 'D' in pid:
					self.plot_capture_ring(pid, situ, px[-1, :])
			if situ == geox:
				self.plot_connect('I0', 'D0', x['I0'], x['D0'])
				self.plot_connect('I0', 'D1', x['I0'], x['D1'])

		xi0 = xs[geox]['I0'][ndr, :]
		xd0s = [xs[geox]['D0'][ndr, :], xs[geox]['D1'][ndr, :]]
		if dr:
			self.plot_dr(xi0, xd0s, ind=True)
		if dcontour:
			self.plot_dcontour(xi0, xd0s)

		self.show_plot(fname=fname)

	def animate(self, ts, xs, ps=None, xrs=None, linestyle=(0, ()), label='', alpha=0.5):
		# print(xs)
		ps = self.process_policy_labels(ps)

		n = xs['D0'].shape[0]
		if ts is None:
			ts = np.linspace(0, 5, n)
		ts = ts - np.amin(ts)
		tail = int(n/5)

		xmin = np.amin(np.array([x[:,0] for p, x in xs.items()]))
		xmax = np.amax(np.array([x[:,0] for p, x in xs.items()]))
		ymin = np.amin(np.array([x[:,1] for p, x in xs.items()]))
		ymax = np.amax(np.array([x[:,1] for p, x in xs.items()]))
		dx = (xmax - xmin)*0.2
		dy = (ymax - ymin)*0.3

		# fig = plt.figure()
		self.ax.set_xlim((xmin-dx, xmax+dx))
		self.ax.set_ylim((ymin-dy, ymax+dy))
		# ax = fig.add_subplot(111, autoscale_on=True, xlim=(xmin-dx, xmax+dx), ylim=(ymin-dy, ymax+dy))
		if xrs is not None:
			for pid, px in xrs.items():
				self.plot_traj(pid, 'ref', px, label=ps[pid])
				if 'D' in pid:
					self.plot_capture_ring(pid, 'ref', px[-1, :])

		time_template = 'time = %.1fs'
		time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes, fontsize=14)
		plots = dict()
		plots['D0'], = self.ax.plot([], [], 'o', color=self.colors['D0'], label=None)
		plots['D1'], = self.ax.plot([], [], 'o', color=self.colors['D1'], label=None)
		plots['I0'], = self.ax.plot([], [], 'o', color=self.colors['I0'], label=None)
		for role in self.game.players:
			if ps[role] is None:
				plots[role+'tail'], = self.ax.plot([], [], linewidth=2, color=self.colors[role], linestyle=linestyle, label=role)
			else:
				if 'I' in role:
					plots[role+'tail'], = self.ax.plot([], [], linewidth=2, color=self.colors[role], linestyle=linestyle, label=' '+role+': '+ps[role])
				else:
					plots[role+'tail'], = self.ax.plot([], [], linewidth=2, color=self.colors[role], linestyle=linestyle, label=role+': '+ps[role])

		plots['Dline'], = self.ax.plot([], [], '--', color='b', label=None)
		plots['D0cap'] = Circle((0, 0), self.r, fc='b', ec=self.colors['D0'], alpha=alpha, label=None)
		plots['D1cap'] = Circle((0, 0), self.r, fc='b', ec=self.colors['D1'], alpha=alpha, label=None)
		self.ax.add_patch(plots['D0cap'])
		self.ax.add_patch(plots['D1cap'])

		self.ax.set_aspect('equal')
		self.ax.grid()
		self.ax.tick_params(axis='both', which='major', labelsize=14)
		plt.xlabel('x(m)', fontsize=14)
		plt.ylabel('y(m)', fontsize=14)
		plt.gca().legend(prop={'size': 12}, ncol=2)

		def init():
			time_text.set_text('')
			for role, x in xs.items():
				# print(role)
				plots[role].set_data([], [])
				plots[role+'tail'].set_data([], [])
				if 'D' in role:
					plots[role+'cap'].center = (x[0,0], x[0,1])
			plots['Dline'].set_data([], [])	

			return plots['D0'], plots['D1'], \
					plots['D0cap'], plots['D1cap'], \
					plots['I0'], \
					plots['D0tail'], plots['D1tail'], plots['I0tail'], \
					plots['Dline'], \
					time_text

		def animate_traj(i):
			i = i%n
			ii = np.clip(i-tail, 0, i)
			time_text.set_text(time_template % (ts[i]))
			for role, x in xs.items():
				plots[role].set_data(x[i,0], x[i,1])
				plots[role+'tail'].set_data(x[ii:i+1,0], x[ii:i+1,1])
				if 'D' in role:
					plots[role+'cap'].center = (x[i,0], x[i,1])
			plots['Dline'].set_data([xs['D0'][i,0], xs['D1'][i,0]], [xs['D0'][i,1], xs['D1'][i,1]])	
			return  plots['D0'], plots['D1'], \
					plots['D0cap'], plots['D1cap'], \
					plots['I0'], \
					plots['D0tail'], plots['D1tail'], plots['I0tail'], \
					plots['Dline'], time_text

		ani = animation.FuncAnimation(self.fig, animate_traj, init_func=init)
		# ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
		#								 repeat_delay=1000)
		ani.save(self.game.res_dir+'ani_traj.gif')
		print(self.game.res_dir+'ani_traj.gif')
		plt.show()	


	def reset(self):
		self.ax.clear()