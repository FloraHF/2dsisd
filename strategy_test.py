import numpy as np
from math import acos
from Games import FastDgame, SlowDgame
from geometries import LineTarget, DominantRegion

x0 = {'D0': np.array([-.3, 0.]), 'I0': np.array([-.03, .1]), 'D1': np.array([.3, 0.])}
game = FastDgame(LineTarget(), None, ni=1, nd=2)
game.reset(x0)
# dr = DominantRegion(game.r, game.a, x0['I0'], [x0['D0'], x0['D1']])
# xt = game.target.deepest_point_in_dr(dr)
# print(dr.level(np.array([0,0])))


ts_play, xs_play = game.advance(8., game.f_strategy, game.f_strategy, close_adjust=False)
game.plotter.plot({'play':xs_play}, 'play', fname='play_traj.png', dr=True, ndr=0)
# game.plotter.animate(ts_play, xs_play, xrs=xs_play)
