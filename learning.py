import matplotlib.pyplot as plt
import numpy as np
from math import pi, asin, acos, ceil
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

from envelope import envelope_traj, envelope_policy, w

BARRIER_DIR = 'BarrierFn'
POLICY_DIR = 'PolicyFn'

def network(n_in, ns, act_fns):
	model = Sequential()
	for i, (n, fn) in enumerate(zip(ns, act_fns)):
		if i == 0:
			model.add(Dense(n, input_shape=(n_in,), activation=fn))
		else:
			model.add(Dense(n, activation=fn))
	model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
	return model

def sample_traj(n_gmm=4, T=7.):
	lb = acos(1/w)
	step = (pi/2 - lb)/(n_gmm - 1)
	xys, yis = [], []
	xis = []
	n_samples = 0
	for gmm in np.linspace(lb, pi/2, n_gmm):
		print('sampling gmm=%.1f/[%.1f, %.1f]'%(gmm*180/pi, lb*180/pi, 90))
		Dmax = pi/2 - gmm
		if 2*Dmax < step:
			Ds = [0]
		elif 2*Dmax <= 2*step:
			Ds = [-Dmax, Dmax]
		else:
			Ds = np.linspace(-Dmax, Dmax, ceil(2*Dmax/step))
		dmax = gmm - lb
		if dmax < step:
			ds = [0]
		elif Dmax <= 2*step:
			ds = [0, dmax]
		else:
			ds = np.linspace(0, dmax, ceil(dmax/step))
		for D in Ds:
			# print('  sampling D=%.1f/[%.1f, %.1f]'%(D*180/pi, -Dmax*180/pi, Dmax*180/pi))
			for delta in ds:
				for S in np.linspace(-asin(1/w), 0.49*pi, n_gmm):
					xs1, xs2 = envelope_traj(S, T, gmm, D, delta, n=50)
					for x1 in xs1:
						xys.append(x1[[0, 1, 2, 4, 5]])
						yis.append(x1[3])
						xis.append(x1[4:])
						n_samples += 1
					# if xs2 is not None:
					# 	for x2 in xs2:
					# 		xys.append(x2[[0, 1, 2, 4, 5]])
					# 		yis.append(x2[3])
					# 		xis.append(x2[4:])
					# 		n_samples += 1		
	xis = np.asarray(xis)
	fig, ax = plt.subplots()
	ax.plot(xis[:,0], xis[:,1], 'o')
	ax.axis('equal')
	ax.grid()
	plt.show()					
	return np.asarray(xys), np.asarray(yis), n_samples

def sample_policy(n_gmm=4, T=7.):
	lb = acos(1/w)
	step = (pi/2 - lb)/(n_gmm - 1)
	xys, ps = [], []
	n_samples = 0
	xis = []
	for gmm in np.linspace(lb, pi/2, n_gmm):
		print('sampling gmm=%.1f/[%.1f, %.1f]'%(gmm*180/pi, lb*180/pi, 90))
		Dmax = pi/2 - gmm
		if 2*Dmax < step:
			Ds = [0]
		elif 2*Dmax <= 2*step:
			Ds = [-Dmax, Dmax]
		else:
			Ds = np.linspace(-Dmax, Dmax, ceil(2*Dmax/step))
		dmax = gmm - lb
		if dmax < step:
			ds = [0]
		elif Dmax <= 2*step:
			ds = [0, dmax]
		else:
			ds = np.linspace(0, dmax, ceil(dmax/step))
		for D in Ds:
			# print('  sampling D=%.1f/[%.1f, %.1f]'%(D*180/pi, -Dmax*180/pi, Dmax*180/pi))
			for delta in ds:
				# print('  sampling delta=%.1f/[%.1f, %.1f]'%(delta*180/pi, 0, dmax*180/pi))
				for S in np.linspace(-asin(1/w), 0.49*pi, n_gmm):
					# print('  sampling S=%.1f/[%.1f, %.1f]'%(S*180/pi, -asin(1/w)*180/pi, 90))
					xs1, xs2 = envelope_traj(S, T, gmm, D, delta, n=50)
					# fig, ax = plt.subplots()
					for x1, p1 in zip(xs1, envelope_policy(xs1)):
						xys.append(x1)
						ps.append(p1)
						n_samples += 1
						# xis.append(x1[4:])					
					if xs2 is not None:
						for x2, p2 in zip(xs2, envelope_policy(xs2)):
							xys.append(x2)
							ps.append(p2)
							# xis.append(x2[4:])
							n_samples += 1
	# xis = np.asarray(xis)
	# fig, ax = plt.subplots()
	# ax.plot(xis[:,0], xis[:,1], 'o')
	# ax.axis('equal')
	# ax.grid()
	# plt.show()	

	return np.asarray(xys), np.asarray(ps), n_samples


def learn_barrier(n_gmm,
				  ns=[20, 20, 1], 
				  act_fns=[tf.nn.relu, tf.nn.relu, None],
				  batch_size=1000,
				  epochs=2000, 
				  save_dir=BARRIER_DIR):

	model = network(5, ns, act_fns)
	xys, yis, n = sample_traj(n_gmm)
	model.fit(xys, yis, batch_size=batch_size, epochs=epochs)
	model.save(save_dir)

def get_barrier_xi(xis, xd1=-3., yd1=3., xd2=-3., yd2=3., dir=BARRIER_DIR):
	barrier = load_model(dir)
	n = len(xis)
	xis = np.linspace(-5., 5., n)
	yis = []
	for xi in xis:
		xy = np.array([xd1, yd1, xi, xd2, yd2])
		yi = barrier.predict(xy[None])[0]
		yis.append(np.squeeze(yi))
	return np.asarray(yis)

def learn_policy(n_gmm,
				  ns=[20, 20, 1], 
				  act_fns=[tf.nn.relu, tf.nn.relu, None],
				  batch_size=3000,
				  epochs=1500,
				  save_dir=POLICY_DIR):

	model = network(6, ns, act_fns)
	xys, ps, n = sample_policy(n_gmm)
	model.fit(xys, ps[:,0], batch_size=batch_size, epochs=epochs)
	model.save(save_dir+'_D0.h5')
	model.fit(xys, ps[:,1], batch_size=batch_size, epochs=epochs)
	model.save(save_dir+'_I0.h5')
	model.fit(xys, ps[:,2], batch_size=batch_size, epochs=epochs)
	model.save(save_dir+'_D1.h5')


if __name__ == '__main__':
	
	learn_policy(6)

