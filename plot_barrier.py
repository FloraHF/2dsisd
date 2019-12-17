import numpy as np
from scipy.interpolate import interp1d

from Games import FastDgame, SlowDgame
from geometries import LineTarget, CircleTarget


game = SlowDgame(LineTarget(), exp_dir='resfd00/')

def get_barrier_data():
	with open('exp_results/barrier_counted.csv', 'r') as f:
		lines = f.readlines()[1:]
		n = len(lines)
		xs = [{'x':0., 'y':[], 'cap':[]}]
		for line in lines:
			data = line.split(',')
			# print(data)
			newx = float(data[0])
			y = float(data[1])
			cap = float(data[-1])

			xexits = False
			for j, x in enumerate(xs):
				oldx = x['x']
				if oldx == newx:
					xexits = True
					x['y'].append(y)
					x['cap'].append(cap)

			if not xexits:
				xs.append({'x': newx, 'y':[y], 'cap':[cap]})

	for x in xs[:-1]:
		x['interp'] = interp1d(x['cap'], x['y'], fill_value='extrapolate')
		print(x['x'], x['interp'](0.5))

game.plotter.plot()
		