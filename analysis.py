import numpy as np
from Games import FastDgame, SlowDgame
from geometries import line


if __name__ == '__main__':

	game = SlowDgame(line, res_dir='res1/', policy_dir='PolicyFn', ni=1, nd=2, read_exp=False)
	xs_ref = game.analytic_traj(game.s_lb+0.6, 4., game.gmm_lb+0.1, 0, 0.099999)
	game.reset({role: x[0,:] for role, x in xs_ref.items()})
	xs_play = game.advance(5., game.nn_strategy, game.nn_strategy, close_adjust=True)
	game.plotter.plot({'play':xs_play, 'ref':xs_ref})

	# game = FastDgame(line, res_dir='res1/', ni=1, nd=2, read_exp=False)
	# game.reset({'D0': np.array([-.6, .2]), 'I0': np.array([.1, .4]), 'D1': np.array([.6, .2])})
	# xs_play = game.advance(2., game.opt_strategy, game.opt_strategy, close_adjust=False)
	# # print(xs_play)
	# # xs_exp  = game.replay_exp()
	# # xis, xds = game.convert_data(xis), game.convert_data(xds)
	# ## xs_play = np.concatenate((xds[0], xis[0], xds[1]),-1)
	# game.plotter.plot({'play':xs_play})	