import matplotlib.pyplot as plt
import numpy as np
from math import pi, sin, cos, atan2, sqrt
from coords import xy_to_s
from scipy.optimize import minimize, root_scalar

from Config import Config
vd = Config.VD
vi = Config.VI
r = Config.CAP_RANGE
a = vd/vi
w = 1/a

def get_Q(s):
	return sqrt(1 + w**2 + 2*w*sin(s))

def get_phi(s):
	Q = get_Q(s)
	cphi = w*cos(s)/Q 
	sphi = -(1 + w*sin(s))/Q
	return atan2(sphi, cphi)

def get_psi(s):
	Q = get_Q(s)
	cpsi = cos(s)/Q 
	spsi = -(w + sin(s))/Q
	return atan2(spsi, cpsi)


def get_headings(xd, yd, xi, yi):

	def get_s(s, xd=xd, yd=yd, xi=xi, yi=yi):

		phi = get_phi(s) - (pi/2 - s)
		psi = get_psi(s) - (pi/2 - s)
		# print(phi, psi)

		def err_d(t, xd=xd, yd=yd, xi=xi, yi=yi):	
			xd_ = xd + vd*cos(phi)*t 
			yd_ = yd + vd*sin(phi)*t 
			xi_ = xi + vi*cos(psi)*t 
			yi_ = yi + vi*sin(psi)*t 
			d = sqrt((xd_ - xi_)**2 + (yd_ - yi_)**2)
			return (d - r)**2

		sol = root_scalar(err_d, x0=0, x1=5)
		t = sol.root
		# print(phi, psi,err_d(t))
		errs = []
		for t in np.linspace(0, 10, 20):
			errs.append(err_d(t))
		fig, ax = plt.subplots()
		ax.plot(np.linspace(0, 10, 20), errs)
		plt.title('%.2f, %.2f'%(phi*180/pi, psi*180/pi))
		plt.show()


		xd_ = xd + vd*cos(phi)*t 
		yd_ = yd + vd*sin(phi)*t 
		xi_ = xi + vi*cos(psi)*t 
		yi_ = yi + vi*sin(psi)*t 

		s_ = xy_to_s(np.array([xd_, yd_, xi_, yi_]))

		return (s - s_)**2

	sol = minimize(get_s, 0, options={'disp': False})
	s = sol.x
	print(s, get_s(s))

	phi = get_phi(s) - (pi/2 - s)
	psi = get_psi(s) - (pi/2 - s)

	return phi, psi

if __name__ == "__main__":
	get_headings(0., 5., 5., 3.)