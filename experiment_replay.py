import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os
import numpy as np
from math import sin, cos, acos, atan2, sqrt, pi
from scipy.interpolate import interp1d

class ReplayPool(object):

	def __init__(self, role, res_dir='res1/'):

		self._role = role
		self.pdict = {'nn': 6,
					  'w': 5,
					  'pt': 4,
					  'pp': 3,
					  'h': 2,
					  'f': 1,
					  'D0 close': 0,
					  'D1 close': 0,
					  'both close': -1,}
		# self._exp_role = role[0] + str(int(role[1:])+1)

		self._script_dir = os.path.dirname(__file__)
		self._res_dir ='exp_results/' + res_dir
		with open(self._res_dir+'info.csv', 'r') as f:
			data = f.readlines()
			for line in data:
				if self._role == line.split(',')[0]:
					self._frame = line.split(',')[-1][:-1]

		self._tm = 1.
		self.t_start, self.t_end, self.t_close, self.fp = self._read_policy()
		self.t_start -= 1.
		self.t_end += 1.
		self.x, self.y = self._read_xy(self.t_start, self.t_end, file='location.csv')
		self.vx, self.vy = self._read_xy(self.t_start, self.t_end, file='cmdVtemp.csv')
		# self.a = self._read_a(self.t_start, self.t_end)

	# read policy
	def _read_policy(self):
		t, p, t_close = [], [], None
		data_dir = os.path.join(self._script_dir, self._res_dir + self._frame + '/data/')
		# print(self._frame)
		with open(data_dir + 'policy.csv') as f:
		    data = f.readlines()
		    for line in data:
		        datastr = line.split(',')
		        time = float(datastr[0])
		        t.append(time)
		        policy = datastr[1]
		        # print(policy)
		        # print(policy)
		        for pname, pnum in self.pdict.items():
		        	if pname in policy:
		        		policy_id = pnum
		        	if 'close' in policy:
		        		if t_close is None:
		        			t_close = time
		        p.append(policy_id)
		# print(len(t), len(p))
		f_policy = interp1d(t, p, fill_value='extrapolate')

		return t[0], t[-1], t_close, f_policy

	# read location
	def _read_xy(self, t_start, t_end, file='location.csv'):
		t, x, y = [], [], []
		data_dir = os.path.join(self._script_dir, self._res_dir + self._frame + '/data/')
		with open(data_dir + file) as f:
		    data = f.readlines()
		    for line in data:
		        datastr = line.split(',')
		        time = float(datastr[0])
		        if t_start-self._tm < time < t_end+self._tm:
		        	t.append(time)
		        	x.append(float(datastr[1]))
		        	y.append(float(datastr[2]))
		t = np.asarray(t)
		x = np.asarray(x)
		y = np.asarray(y)

		return interp1d(t, x, fill_value='extrapolate'), interp1d(t, y, fill_value='extrapolate')

	# velocity
	def _read_a(self, t_start, t_end):
		t, a = [], []
		# data_dir = os.path.join(self._script_dir, self._res_dir + self._frame + '/data/')
		data_dir = os.path.join(self._script_dir, self._res_dir)
		with open(data_dir + 'a.csv') as f:
		    data = f.readlines()
		    for line in data:
		        datastr = line.split(',')
		        time = float(datastr[0])
		        if time > t_start-0.1 and time < t_end+0.1:
		        	t.append(time)
		        	a.append(float(datastr[1]))       
		t = np.asarray(t)
		a = np.asarray(a)

		return interp1d(t, a, fill_value='extrapolate')

	# ########################## coords ###########################
	# def _get_vecs(self, xd1, yd1, xd2, yd2, xi, yi):
	#     D1 = np.array([xd1, yd1, 0])
	#     D2 = np.array([xd2, yd2, 0])
	#     I = np.array([xi, yi, 0])
	#     D1_I = I - D1
	#     D2_I = I - D2
	#     D1_D2 = D2 - D1
	#     return D1_I, D2_I, D1_D2

	# def _get_xyz(self, D1_I, D2_I, D1_D2):
	#     z = np.linalg.norm(D1_D2/2)
	#     x = -np.cross(D1_D2, D1_I)[-1]/(2*z)
	#     y = np.dot(D1_D2, D1_I)/(2*z) - z
	#     return x, y, z

	# def _get_theta(self, D1_I, D2_I, D1_D2):
	#     k1 = atan2(np.cross(D1_D2, D1_I)[-1], np.dot(D1_D2, D1_I))
	#     k2 = atan2(np.cross(D1_D2, D2_I)[-1], np.dot(D1_D2, D2_I))  # angle between D1_D2 to D2_I
	#     tht = k2 - k1
	#     if k1 < 0:
	#         tht += 2*pi
	#     return tht

	# def _get_d(self, D1_I, D2_I, D1_D2):
	#     d1 = max(np.linalg.norm(D1_I), r)
	#     d2 = max(np.linalg.norm(D2_I), r)
	#     return d1, d2

	# def _get_alpha(self, D1_I, D2_I, D1_D2):
	#     d1, d2 = self._get_d(D1_I, D2_I, D1_D2)
	#     a1 = asin(r/d1)
	#     a2 = asin(r/d2)
	#     return d1, d2, a1, a2

	# ########################## policies ###########################
	# def _get_physical_heading(self, phi_1, phi_2, psi, D1_I, D2_I, D1_D2):
	# 	# print('relative headings', phi_1*180/pi, phi_2*180/pi, psi*180/pi)
	# 	dphi_1 = atan2(D1_I[1], D1_I[0])
	# 	dphi_2 = atan2(D2_I[1], D2_I[0])
	# 	dpsi = atan2(-D2_I[1], -D2_I[0])
	# 	# print(dphi_1*180/pi, dphi_2*180/pi, dpsi*180/pi)
	# 	phi_1 += dphi_1
	# 	phi_2 += dphi_2
	# 	psi += dpsi
	# 	# print('physical headings', phi_1*180/pi, phi_2*180/pi, psi*180/pi)
	# 	return phi_1, phi_2, psi

	# def _h_strategy(self, xd1, yd1, xd2, yd2, xi, yi, a=.1/.15):

	# 	D1_I, D2_I, D1_D2 = self._get_vecs(xd1, yd1, xd2, yd2, xi, yi)
	# 	x, y, z = self._get_xyz(D1_I, D2_I, D1_D2)
	# 	print('xyz', x, y, z)

	# 	xd1_, yd1_ = 0, -z
	# 	xd2_, yd2_ = 0,  z
	# 	xi_, yi_   = x, y

	# 	# print(xd1_, yd1_, xd2_, yd2_, xi_, yi_)
	# 	Delta = sqrt(np.maximum(x**2 - (1 - 1/a**2)*(x**2 + y**2 - (z/a)**2), 0))
	# 	if (x + Delta)/(1 - 1/a**2) - x > 0:
	# 	    xP = (x + Delta)/(1 - 1/a**2)
	# 	else:
	# 	    xP = -(x + Delta)/(1 - 1/a**2)

	# 	P = np.array([xP, 0, 0])
	# 	D1_P = P - np.array([xd1_, yd1_, 0])
	# 	D2_P = P - np.array([xd2_, yd2_, 0])
	# 	I_P = P - np.array([xi_, yi_, 0])
	# 	D1_I_, D2_I_, D1_D2_ = self._get_vecs(xd1_, yd1_, xd2_, yd2_, xi_, yi_)

	# 	phi_1 = atan2(np.cross(D1_I_, D1_P)[-1], np.dot(D1_I_, D1_P))
	# 	phi_2 = atan2(np.cross(D2_I_, D2_P)[-1], np.dot(D2_I_, D2_P))
	# 	psi = atan2(np.cross(-D2_I_, I_P)[-1], np.dot(-D2_I_, I_P))

	# 	# print(phi_1*180/pi, phi_2*180/pi, psi*180/pi)

	# 	return phi_1, phi_2, psi

	# def _adjust_strategy(self, phi_1, phi_2, psi, D1_I, D2_I, D1_D2):
	# 	tht = self._get_theta(D1_I, D2_I, D1_D2)
	# 	if np.linalg.norm(D1_I) < self._r_close and np.linalg.norm(D2_I) < self._r_close:  # in both range
	# 		vD1 = np.array([vxd1, vyd1, 0])
	# 		vD2 = np.array([vxd2, vyd2, 0])
	# 		phi_1 = atan2(np.cross(D1_I, vD1)[-1], np.dot(D1_I, vD1))
	# 		phi_2 = atan2(np.cross(D2_I, vD2)[-1], np.dot(D2_I, vD2))
	# 		phi_1 = self._k_close*phi_1
	# 		phi_2 = self._k_close*phi_2
	# 		psi = -tht / 2
	# 	elif np.linalg.norm(D1_I) < self._r_close:
	# 		vD1 = np.array([vxd1, vyd1, 0])
	# 		phi_1 = atan2(np.cross(D1_I, vD1)[-1], np.dot(D1_I, vD1))
	# 		phi_1 = self._k_close*phi_1
	# 		if np.linalg.norm(vI) > np.linalg.norm(vD1):
	# 		    psi = - abs(acos(np.linalg.norm(vD1)*cos(phi_1)/np.linalg.norm(vI)))
	# 		else:
	# 		    psi = - abs(phi_1)
	# 		psi = pi - tht + psi
	# 	elif np.linalg.norm(D2_I) < self._r_close:
	# 		vD2 = np.array([vxd2, vyd2, 0])
	# 		phi_2 = atan2(np.cross(D2_I, vD2)[-1], np.dot(D2_I, vD2))
	# 		psi = self._k_close * phi_2
	# 		if np.linalg.norm(vI) > np.linalg.norm(vD2):
	# 		    psi = abs(acos(np.linalg.norm(vD2)*cos(phi_2)/np.linalg.norm(vI)))
	# 		else:
	# 		    psi = abs(phi_2)
	# 		psi = psi - pi

	# 	return phi_1, phi_2, psi

