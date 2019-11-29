import matplotlib.pyplot as plt
import numpy as np
from math import acos
import cmath as cm
from Games import FastDgame, SlowDgame
from geometries import LineTarget, DominantRegion


x0 = {'D0': np.array([-.5, 0.]), 'I0': np.array([-.0, .4]), 'D1': np.array([.5, 0.])}
game = FastDgame(LineTarget(), None, ni=1, nd=2)
game.reset(x0)
# dr = DominantRegion(game.r, game.a, x0['I0'], [x0['D0'], x0['D1']])
# xt = game.target.deepest_point_in_dr(dr)
# print(dr.level(np.array([0,0])))


ts_play, xs_play = game.advance(8., game.f_strategy, game.f_strategy, close_adjust=False)
game.plotter.plot({'play':xs_play}, 'play', fname='play_traj.png', dr=True, ndr=0)


cubs_real = [[], [], []]
cubs_imag = [[], [], []]
for x in np.linspace(1.5, -0.4, 100):
	x0['I0'] = np.array([0., x])
	cub = game.p_strategy(x0)
	for c in cub:
		print(c)
	for i in range(3):
		cubs_real[i].append(cub[i].real)
		cubs_imag[i].append(cub[i].imag)
fig, ax = plt.subplots()
for i, (real, imag) in enumerate(zip(cubs_real, cubs_imag)):
	ax.plot(np.asarray(real), np.asarray(imag), '.', color='b')
plt.show()
# game.plotter.animate(ts_play, xs_play, xrs=xs_play)
