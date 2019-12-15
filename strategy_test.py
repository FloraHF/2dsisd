import numpy as np
from math import acos

from Games import FastDgame, SlowDgame
from geometries import LineTarget

with open('config.csv', 'r') as f:
	data = f.readlines()
	for line in data:
		if 'vd' in line:
			vd = float(line.split(',')[-1])
		if 'vi' in line:
			vi = float(line.split(',')[-1])
sim_dir = 'res5/'			
xplot = dict()

if vd >= vi:
	game = FastDgame(LineTarget(), sim_dir=sim_dir)
	x0 = {'D0': np.array([-.8, 0.2]), 'I0': np.array([-.1, .4]), 'D1': np.array([.8, 0.2])}
else:
	game = SlowDgame(LineTarget(), sim_dir=sim_dir)
	# rgame = SlowDgame(LineTarget(), sim_dir=sim_dir)
	# for role in rgame.pstrategy:
	# 	rgame.pstrategy[role] = 'nn'

	xref = game.generate_analytic_traj(.0, 5, acos(1/1.5)+0.2,0,0.1999999999, file='traj_param.csv')
	x0 = dict()
	for role in xref:
		if 'I' in role:
			x0[role] = xref[role][0] + np.array([0., .0])
		else:
			x0[role] = xref[role][0]

	xplot['ref'] = xref
	# rgame.reset(x0)
	# tref, xplot['ref'] = rgame.advance(8.)
	# x0 = {role: x[0] for role, x in xref.items()}
	
	# x0 = {'D0': np.array([-.6, 0.9]), 'I0': np.array([-.1, 1.2]), 'D1': np.array([.6, 0.9])}
 
game.reset(x0)
tplay, xplay = game.advance(8.)
xplot['play'] = xplay

fname = '_'.join([strategy for role, strategy in game.pstrategy.items()])
game.plotter.plot(xs=xplot, geox='play', ps=game.pstrategy, dcontour=False, traj=True, fname='traj_'+fname+'.png')
# game.plotter.animate(ts, xs_exp, pdict, xrs=xs_ref)

# game.plotter.plot_target()