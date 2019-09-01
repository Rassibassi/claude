import numpy as np
import claude.utils as cu

def qammod(M):
	r = np.arange(np.array(np.sqrt(M)))
	r = 2*(r-np.mean(r))
	r = np.meshgrid(r,r)
	constellation = np.expand_dims(np.reshape(r[0]+1j*r[1],[-1]),axis=0)
	norm = np.sqrt(np.mean(np.abs(constellation)**2))
	return constellation/norm

def generateSymbols(constellation,size):
	M=constellation.shape[0]
	constellation = constellation[0,np.random.randint(M, size=size)]
	return constellation

def rrcos(beta, sps, span):
    delay = span * sps / 2
    t     = np.arange(-delay, delay+1) / sps
    pi    = np.pi
    
    # Initialization
    h = np.zeros(t.shape)

    # find t=0
    idx1 = np.where(t == 0)
    if len(idx1):
        h[idx1] = (1 + beta * (4 / pi - 1)) / sps

    # find where denominator is zero
    idx2 = np.where(np.abs(np.abs(4*beta*t) - 1.0) < np.sqrt(np.finfo(float).eps))
    if len(idx2):
        pi4beta = pi / (4 * beta)
        h[idx2] = beta / np.sqrt(2) * ( (1 + 2/pi) * np.sin(pi4beta) + (1 - 2/pi) * np.cos(pi4beta) ) / sps

    # main equation   
    ind    = np.arange(0,len(t))
    ind    = np.delete(ind, np.union1d(idx1, idx2))
    nind   = t[ind]
    h[ind] = ( np.sin(pi * nind * (1-beta)) + 4 * beta * nind * np.cos(pi * nind * (1+beta)) ) / ( pi * nind * ( 1 - (4 * beta * nind)**2 ) ) / sps
    
    # Normalize filter energy
    h = h / np.sqrt(np.sum(h**2));
    return h

def fftShiftZeroPad(h, signalLength):
	zeroPadding = (signalLength-h.shape[0])
	h = np.concatenate([h[int(h.shape[0]/2):],np.zeros((zeroPadding,)),h[:int(h.shape[0]/2)]])
	return h