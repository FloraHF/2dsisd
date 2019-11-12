from math import atan2
from Config import Config
r = Config.CAP_RANGE
def xy_to_s(x):
	cos_s = (x[1] - x[3])/r
	sin_s = (x[2] - x[0])/r
	return atan2(sin_s, cos_s)