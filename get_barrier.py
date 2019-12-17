import os
import numpy as np

if os.path.exists('exp_results/barrier.csv'):	
	os.remove('exp_results/barrier.csv')

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
			with open('exp_results/barrier.csv', 'a') as f:
				f.write(','.join(map(str, [x, y, dstrategy, termination]))+'\n')

with open('exp_results/barrier.csv', 'r') as f:
	lines = f.readlines()
	n = len(lines)
	xs = {'x':[], 'cap':np.zeros(n), 'enter':np.zeros(n)}
	for i, line in enumerate(lines):
		data = line.split(',')
		newx = np.array([float(data[0]), float(data[1])])
		termination = data[-1].rstrip()
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
				xs['cap'][j+1] = 1
			if termination == 'entered':
				xs['enter'][j+1] = 1

if os.path.exists('exp_results/barrier_counted.csv'):	
	os.remove('exp_results/barrier_counted.csv')

with open('exp_results/barrier_counted.csv', 'a') as f:
	data = []
	for x in xs:
		data.append(x)
	f.write(','.join(map(str, data))+'\n')

with open('exp_results/barrier_counted.csv', 'a') as f:
	for i in range(len(xs['x'])):
		data = []
		for x, d in xs.items():
			# print(len(d))
			if x == 'x':
				data.append(d[i][0])
				data.append(d[i][1])
			else:
				data.append(d[i])
		data.append(float(data[-2])/(float(data[-1]) + float(data[-2])))
		f.write(','.join(map(str, data))+'\n')