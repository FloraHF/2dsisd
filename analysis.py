import numpy as np
import time
from math import acos
from Games import FastDgame, SlowDgame
from geometries import LineTarget, CircleTarget

def play_fastD_game(xd0, xi, xd1, ni=1, nd=2, param_file='traj_param_100.csv'):
	game = FastDgame(LineTarget())
	x0 = {'D0': xd0, 'I0': xi, 'D1': xd1}
	game.reset(x0)
	game.record_data(x0, file=param_file)

	ts_play, xs_play = game.advance(8.)
	fname = '_'.join([strategy for role, strategy in game.pstrategy.items()])
	figid = param_file.split('.')[0].split('_')[-1]
	game.plotter.plot(xs={'play': xs_play}, geox='play', ps=game.pstrategy, traj=True, dcontour=True, fname='traj_'+fname+'_'+figid+'.png')

def generate_data_for_exp(S, T, gmm, D, delta, offy=0.2, ni=1, nd=2, param_file='traj_param_100.csv'):
	game = SlowDgame(LineTarget())
	xs_ref = game.generate_analytic_traj(S, T, gmm, D, delta, offy=offy, file=param_file)
	game.reset({'D0': xs_ref['D0'][0,:], 'I0': xs_ref['I0'][0,:], 'D1': xs_ref['D1'][0,:]})
	ts, xs_play = game.advance(8.)

	# for role in xs_ref:
	# 	xs_ref[role] = game.rotate_to_exp(xs_ref[role])
	xplot = {'play': xs_play, 'ref': xs_ref}

	fname = '_'.join([strategy for role, strategy in game.pstrategy.items()])
	figid = param_file.split('.')[0].split('_')[-1]
	game.plotter.plot(xs=xplot, geox='play', ps=game.pstrategy, traj=True, fname='traj_'+fname+'_'+figid+'.png')

def replay_exp(res_dir='res1/', ni=1, nd=2):
	x0s = dict()
	pdict = dict()
	with open('exp_results/'+res_dir+'/info.csv', 'r') as f:
		data = f.readlines()
		for line in data:
			if 'vd' in line:
				vd = float(line.split(',')[-1])
			if 'vi' in line:
				vi = float(line.split(',')[-1])
			if 'x' in line:
				ldata = line.split(',')
				role = ldata[0][1:]
				x0s[role] = np.array([float(ldata[1]), float(ldata[2])])	

	if vd < vi:
		game = SlowDgame(LineTarget(), exp_dir=res_dir, ni=ni, nd=nd)
		xs_ref = game.reproduce_analytic_traj()
		ts_ref = None
		# ts_ref2, xs_ref2 = game.advance(8.)
	else:
		game = FastDgame(LineTarget(), exp_dir=res_dir, ni=ni, nd=nd)
		ts_ref, xs_ref = game.advance(8.)

	game.reset({role: x for role, x in x0s.items()})
	ts, xs_exp, ps_exp = game.replay_exp()
	pdict = game.pstrategy
	game.plotter.animate(ts, xs_exp, pdict, xrs=xs_ref)
	game.plotter.plot({'exp':xs_exp}, 'exp', pdict, dr=False, fname='replay_traj.png')
	# game.plotter.plot({'ref':xs_ref, 'exp':xs_exp}, 'exp', pdict, dr=False, fname='replay_traj.png')


if __name__ == '__main__':

	# t0 = time.clock()
	# generate_data_for_exp(-.98, 2.5, acos(25/27)+0.5, 0, 0.3, param_file='traj_param_sd0.csv')
	play_fastD_game(np.array([-.85, 0.2]), np.array([-0.2, 0.4]), np.array([.85, 0.2]), param_file='traj_param_tst0.csv')

	# replay_exp(res_dir='ressd01/')

	# t1 = time.clock()
	# print(t1 - t0)