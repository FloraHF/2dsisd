import numpy as np
from Games import FastDgame, SlowDgame
from geometries import LineTarget, CircleTarget


game = SlowDgame(LineTarget(), exp_dir='resfd00/')

with open('exp_results/barrier_counted.csv', 'a') as f:
	lines = f.readlines()
	n = len(lines)
	xs = [{'x':0., 'y':[], 'cap':[]}]
	for line in lines:
		data = line.split(',')
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

for x in xs:
	print(x)			
			