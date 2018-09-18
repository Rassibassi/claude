import numpy as np
import claude.utils as cu

def qammod(M):
	r = np.arange(np.array(np.sqrt(M)))
	r = 2*(r-np.mean(r))
	r = np.meshgrid(r,r)
	constellation = np.expand_dims(np.reshape(r[0]+1j*r[1],[-1]),axis=0)
	norm = np.sqrt(np.mean(np.abs(constellation)**2))
	return constellation/norm