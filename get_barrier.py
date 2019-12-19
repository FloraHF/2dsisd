import os
import numpy as np

from Games import SlowDgame, FastDgame
from geometries import LineTarget


def get_exp_barrier():

	if os.path.exists('exp_results/exp_barrier.csv'):	
		os.remove('exp_results/exp_barrier.csv')

	for i in range(5):
		for j in range(20):
			res_dir = 'exp_results/'+'resfd%s%s'%(i,j)
			if os.path.exists(res_dir):
				with open(res_dir+'/info.csv', 'r') as f:
					lines = f.readlines()
					for line in lines:
						data = line.split(',')
						if 'xI0' == data[0]:
							x = data[1]
							y = data[2].rstrip()
						if 'dstrategy' == data[0]:
							dstrategy = data[-1].rstrip()
						if 'termination' == data[0]:
							termination = data[-1].rstrip()
				with open('exp_results/exp_barrier.csv', 'a') as f:
					f.write(','.join(map(str, [x, y, dstrategy, termination]))+'\n')

	with open('exp_results/exp_barrier.csv', 'r') as f:
		lines = f.readlines()
		n = len(lines)
		xs = {'x':[], 'cap':np.zeros(n), 'enter':np.zeros(n)}
		for i, line in enumerate(lines):
			data = line.split(',')
			newx = np.array([float(data[0]), float(data[1])])
			termination = data[-1].rstrip()
			# print(newx, termination)
			# print(newx)
			# print(xs['x'])
			xexits = False
			for j, oldx in enumerate(xs['x']):
				# print(oldx, newx)
				if np.array_equal(oldx,newx):
					xexits = True
					if termination == 'captured':
						xs['cap'][j] += 1
					if termination == 'entered':
						xs['enter'][j] += 1

			if not xexits:
				xs['x'].append(newx)
				if termination == 'captured':
					xs['cap'][len(xs['x'])-1] = 1
				if termination == 'entered':
					xs['enter'][len(xs['x'])-1] = 1

			# print(xs)

	if os.path.exists('exp_results/exp_barrier_counted.csv'):	
		os.remove('exp_results/exp_barrier_counted.csv')

	with open('exp_results/exp_barrier_counted.csv', 'a') as f:
		data = []
		for x in xs:
			data.append(x)
		f.write(','.join(map(str, data))+'\n')

	with open('exp_results/exp_barrier_counted.csv', 'a') as f:
		for i in range(len(xs['x'])):
			data = []
			for x, d in xs.items():
				# print()
				if x == 'x':
					data.append(d[i][0])
					data.append(d[i][1])
				else:
					data.append(d[i])
			# print(data)
			data.append(float(data[-2])/(float(data[-1]) + float(data[-2])))
			f.write(','.join(map(str, data))+'\n')

def get_sim_barrier():

	if os.path.exists('exp_results/sim_barrier.csv'):	
		os.remove('exp_results/sim_barrier.csv')
	
	x0 = dict()
	with open('exp_results/resfd00'+'/info.csv', 'r') as f:
		data = f.readlines()
		for line in data:
			if 'x' in line:
				ldata = line.split(',')
				role = ldata[0][1:]
				x0[role] = np.array([float(ldata[1]), float(ldata[2])])	
			if 'vd' in line:
				vd = float(line.split(',')[-1])
			if 'vi' in line:
				vi = float(line.split(',')[-1])

	if vd < vi:
		game = SlowDgame(LineTarget(), exp_dir='resfd00/')
		lb0, ub0 = .1, .8
	else:
		game = FastDgame(LineTarget(), exp_dir='resfd00/')
		lb0, ub0 = -.1, .4


	xbs, ybs = [], []
	with open('exp_results/sim_barrier.csv', 'a') as f:			
		for xI in np.linspace(-.7, 0, 30):
		# for xI in [-.2]:
			lb, ub = lb0, ub0
			print(xI)
			while abs(ub - lb) > 0.001:
				# print(ub, lb)
				yI = .5*(lb + ub)
				x0['I0'] = np.array([xI, yI])
				# print(x0)
				game.reset(x0)
				_, xs, info = game.advance(20.)
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

	return xbs, ybs

get_exp_barrier()
get_sim_barrier()
