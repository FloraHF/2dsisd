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
	game.plotter.plot(xs={'play': xs_play}, geox='play', ps=game.pstrategy, dcontour=True, traj=True, fname='traj_'+fname+'_'+figid+'.png')

def generate_data_for_exp(S, T, gmm, D, delta, ni=1, nd=2, param_file='traj_param_100.csv'):
	game = SlowDgame(LineTarget())
	# xs_ref = game.generate_analytic_traj(S, T, gmm, D, delta, file=param_file)
	# game.reset({'D0': xs_ref['D0'][0,:], 'I0': xs_ref['I0'][0,:], 'D1': xs_ref['D1'][0,:]})
	# ts, xs_play = game.advance(8.)

	# xplot = {'play': xs_play, 'ref': xs_ref}
	game.reset({'D0': np.array([-0.85, 0.001]), 'I0': np.array([-0.2, 0.57]), 'D1': np.array([0.85, 0.001])})
	ts, xs_play = game.advance(8.)
	xplot = {'play': xs_play}

	fname = '_'.join([strategy for role, strategy in game.pstrategy.items()])
	figid = param_file.split('.')[0].split('_')[-1]
	game.plotter.plot(xs=xplot, geox='play', ps=game.pstrategy, traj=True, dr=True, fname='traj_'+fname+'_'+figid+'.png')

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
		game = SlowDgame(LineTarget(), res_dir=res_dir, ni=ni, nd=nd)
		xs_ref = game.reproduce_analytic_traj()
		ts_ref = None
	else:
		game = FastDgame(LineTarget(), res_dir=res_dir, ni=ni, nd=nd)
		ts_ref, xs_ref = game.advance(6., game.f_strategy, game.f_strategy, close_adjust=False)

	game.reset({role: x for role, x in x0s.items()})
	ts, xs_exp, ps_exp = game.replay_exp()
	for role, policy in ps_exp.items():
		pnum = int(policy[0])
		for pname_, pnum_ in game.players[role].exp.pdict.items():
			if pnum_ == pnum:
				pdict[role] = pname_

	game.plotter.plot({'ref':xs_ref, 'exp':xs_exp}, 'exp', pdict, fname='replay_traj.png')
	game.plotter.animate(ts, xs_exp, pdict, xrs=xs_ref)


if __name__ == '__main__':

	t0 = time.clock()
	# generate_data_for_exp(-0.02, 5.3, acos(1/1.5)+0.2, 0, 0.189999999, param_file='traj_param_xx.csv')
	play_fastD_game(np.array([-.85, -0.3]), np.array([-0.2, 0.1]), np.array([.85, -0.3]), param_file='traj_param_0x.csv')
	# replay_exp(res_dir='res5/', ni=1, nd=2)

	t1 = time.clock()
	print(t1 - t0)