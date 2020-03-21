import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import matplotlib.ticker as ticker
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from math import sin, cos

from Games import SlowDgame, FastDgame
from geometries import LineTarget

def sim_barrier(r, vd):

	x0 = {'D0': np.array([-.85, .2]), 'I0': np.array([-.2, 1.2]), 'D1': np.array([.85, .2])}
	if 1. < r:
		game = SlowDgame(LineTarget())
		lb0, ub0 = .0, .5
	else:
		game = FastDgame(LineTarget())
		lb0, ub0 = -.1, .4

	vi = r*vd
	game.reset(x0)
	game.set_vi(r*vd)
	game.set_vd(vd)

	xbs, ybs = [], []

	if os.path.exists('sim_revision1/sim_barrier_%.2f.csv'%((vd/vi)*100)):	
		os.remove('sim_revision1/sim_barrier_%.2f.csv'%((vd/vi)*100))

	with open('sim_revision1/sim_barrier_%.2f.csv'%((vd/vi)*100), 'a') as f:			
		for xI in np.linspace(-.6, 0, 13):
		# for xI in [-.6]:
			lb, ub = lb0, ub0
			print(xI)
			while abs(ub - lb) > 0.0005:
				# print(ub, lb)
				yI = .5*(lb + ub)
				x0['I0'] = np.array([xI, yI])
				# print(x0)
				game.reset(x0)
				# print('!!!!!!!!!! reseted x0 !!!!!!!!!!!!!')
				_, xs, info = game.advance(5000.)
				print(yI, info)
				# game.plotter.reset()
				# game.plotter.plot({'play':xs}, 'play', game.pstrategy, fname=None)
				if info == 'captured':
					ub = yI
				elif info =='entered':
					lb = yI
			xbs.append(xI)
			ybs.append(.5*(lb + ub))
			f.write(','.join(map(str, [xI, .5*(lb + ub)]))+'\n')
	
	# game.plotter.plot_sim_barrier_line()

	return xbs, ybs

def sim_traj(rs, vd):

	x0 = {'D0': np.array([-.85, .2]), 'I0': np.array([-0.5, .75]), 'D1': np.array([.85, .2])}
	trajs = []
	# rs = []

	for r in rs:
		if 1. < r:
			game = SlowDgame(LineTarget())
		else:
			game = FastDgame(LineTarget())

		game.reset(x0)
		game.set_vi(r*vd)
		game.set_vd(vd)

		_, traj, _ = game.advance(8.)
		trajs.append(traj)
		# rs.append(vd/vi)

	game.plotter.plot_traj_compare(trajs, rs)

def plot_sim_barrier():

	ratios = [1/0.8, 1/.97, 1/1.2, 1/1.56]
	# ratios = [25/24]
	linestyles=[(0, ((ratio-.6)*8, (ratio-.6)*3)) for ratio in ratios[::-1]]
	linestyles[-1] = 'solid'
	alphas = [(ratio*.8)**1.7 for ratio in ratios[::-1]]

	game = SlowDgame(LineTarget())
	fig, ax = plt.subplots()
	colors = ['purple', 'magenta', 'red', 'orange']
	for i, ratio in enumerate(ratios):
	# for ratio in [1/1.2,]:
		print((ratio/1.26)**1.7)
		xy = []
		with open('sim_revision1/sim_barrier_%.2f.csv'%(ratio*100), 'r') as f:
			print('sim_revision1/sim_barrier_%.2f.csv'%(ratio*100))
			lines = f.readlines()[1:]
			for line in lines:
				data = line.split(',')
				xy.append(np.array([float(data[0]), float(data[1])]))
				xy.append(np.array([-float(data[0]), float(data[1])]))
				# xy.append(np.array([-float(data[1]), float(data[0])]))
				# xy.append(np.array([-float(data[1]), -float(data[0])]))
		xy = np.asarray(sorted(xy, key=lambda xys: xys[0]))
		# xy = game.rotate_to_exp_point(xy)
		# line, = self.ax.plot(px[:,0], px[:,1], color=self.colors[pid], label='a=%.1f'%ratio, alpha=(ratio/1.4)**1.7, linestyle=(0, ((ratio-.6)*8, (ratio-.6)*3)))
		ax.plot(xy[:,0], xy[:,1], color='r', label='a=%.1f'%ratio, alpha=alphas[i], linestyle=linestyles[i])
		print(xy)

	# ax.plot([-1.5, 1.5], [-.5, -.5], color='k', label='target')
	# ax.plot([-0.85], [0.2], 'bo')
	# ax.plot([0.85], [0.2], 'bo')
	ax.plot([-.5], [.75], 'rx')
	ax.plot([-.2], [.5], 'ro', )
	ax.plot([-1.5, 1.5], [-.5, -.5], color='k', label='target')
	ax.plot([-0.85], [.2], 'bo')
	ax.plot([0.85], [.2], 'bo')
	ring = Circle((0.85, 0.2), game.r, fc='b', ec='b', alpha=0.6, linewidth=2)
	ax.add_patch(ring)
	ring = Circle((-0.85, 0.2), game.r, fc='b', ec='b', alpha=0.6, linewidth=2)
	ax.add_patch(ring)

	ring = Circle((0.85, 0.2), game.r*1.2, fc='b', ec='b', alpha=0.2)
	ax.add_patch(ring)
	ring = Circle((-0.85, 0.2), game.r*1.2, fc='b', ec='b', alpha=0.2)
	ax.add_patch(ring)
	# plt.xticks([-2.0, -1.5, -1.0, -0.5, 0, 0.5], [2.0, 1.5, 1.0, 0.5, 0, -0.5])
	
	xrs, yrs = [], []
	for t in np.linspace(0, 2*3.1416, 50):
		xrs.append(.85 + game.r*cos(t))
		yrs.append(.2 + game.r*sin(t))
	ax.plot(xrs, yrs, 'b')
	xrs, yrs = [], []
	for t in np.linspace(0, 2*3.1416, 50):
		xrs.append(-.85 + game.r*cos(t))
		yrs.append(.2 + game.r*sin(t))
	ax.plot(xrs, yrs, 'b')	

	ax.grid()
	ax.tick_params(axis='both', which='major', labelsize=15)
	plt.xlabel('y(m)', fontsize=15)
	plt.ylabel('x(m)', fontsize=15)

	plt.gca().legend(prop={'size': 14}, ncol=5, handletextpad=0.1, columnspacing=0.4, loc='lower center', numpoints=1, handlelength=1.2)
	plt.subplots_adjust(bottom=0.13)
	# print('set legend')
	ax.set_aspect('equal')
	ax.set_xlim([-1.2, 1.2])
	ax.set_ylim([-.53, .92])

	plt.show()

# sim_barrier(0.8, 0.25)
# sim_barrier(0.97, 0.25)
# sim_barrier(1.2, 0.25)
# sim_barrier(1.56, 0.25)
# sim_barrier(0.24/0.25, 0.25)
plot_sim_barrier()
# sim_traj([.8, .97, 1.2, 1.56], 0.25)
# sim_traj([1.2, 1.2, 1.2, 1.2], 0.25)
