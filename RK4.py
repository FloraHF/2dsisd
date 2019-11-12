import numpy as np

def rk4(f, x0, dt):
	# one-step rk4 integration 
	k1 = f(x0)
	k2 = f(x0 + 0.5*k1*dt)
	k3 = f(x0 + 0.5*k2*dt)
	k4 = f(x0 + 1.0*k3*dt)
	k = (k1 + k2 + k2 + k3 + k3 + k4)/6
	return x0 + k*dt

def rk4_fxt(f, t0, x0, dt):
	k1 = f(t0, x0)
	k2 = f(t0 + 0.5*dt, x0 + 0.5*k1*dt)
	k3 = f(t0 + 0.5*dt, x0 + 0.5*k2*dt)
	k4 = f(t0 + 1.0*dt, x0 + 1.0*k3*dt)
	k = (k1 + k2 + k2 + k3 + k3 + k4)/6
	return x0 + k*dt

def rk4_fxt_interval(f, t0, x0, te, dt):
	n = int((te - t0)/dt)
	ts = np.linspace(t0, te, n)
	dt = (te - t0)/(n - 1)
	
	x = x0
	for t in ts[:-2]:
		x = rk4_fxt(f, t, x, dt)
	return x

