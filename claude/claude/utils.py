import numpy as np
from scipy.optimize import fminbound

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ''
        for key in self.__dict__.keys():
            s = s + key + ':' + 1*'\t' + str( self.__dict__[key] ) + '\n'

        return s

def unitlessAxis(samplesPerSymbol,filterSpan):
    axis = np.linspace(-(filterSpan/2), (filterSpan/2), samplesPerSymbol*filterSpan+1)
    return axis[:-1]

def freqAxis(N,Fs):
    if N/2%1==0:
        f = np.concatenate( [np.arange(0,N/2), np.arange(-N/2,0)] )/(N/Fs)
    else:
        f = np.concatenate( [np.arange(0,N/2), np.arange(-N/2+.5,0)] )/(N/Fs)
    return f
    
def omegaAxis(N,Fs):
    return 2*np.pi*freqAxis(N,Fs)
    
def hotOnes(size,tanspose,M,seed=None):
    if seed!=None:
        np.random.seed(seed)
    x_seed = np.eye(M, dtype=int)
    idx = np.random.randint(M,size=size)
    x = np.transpose(x_seed[:,idx], tanspose)
    return x,idx,x_seed
    
def lin2dB(lin,dBtype='dBm'):
    if dBtype == 'db' or dBtype == 'dB':
        fact = 0
    elif dBtype == 'dbm' or dBtype == 'dBm':
        fact = -30
    elif dBtype == 'dbu' or dBtype == 'dBu':
        fact = -60
    else:
        raise ValueError('dBtype can only be dB, dBm or dBu.')

    return 10*np.log10(lin)-fact

def dB2lin(dB,dBtype='dBm'):
    if dBtype == 'db' or dBtype == 'dB':
        fact = 0
    elif dBtype == 'dbm' or dBtype == 'dBm':
        fact = -30
    elif dBtype == 'dbu' or dBtype == 'dBu':
        fact = -60
    else:
        raise ValueError('dBtype can only be dB, dBm or dBu.')

    return 10**( (dB+fact)/10 )

def SNRtoMI(N,effSNR,constellation):
    N = int(N)

    SNRlin = 10**(effSNR/10)
    constellation = constellation/np.sqrt(np.mean(np.abs(constellation)**2))
    M = constellation.size

    ## Simulation
    x_id = np.random.randint(0,M,(N,))
    x = constellation[:,x_id]

    z = 1/np.sqrt(2)*( np.random.normal(size=x.shape) + 1j*np.random.normal(size=x.shape) );
    y = x + z*np.sqrt(1/SNRlin);

    return calcMI_MC(x,y,constellation)

def calcMI_MC(x,y,constellation):
    """
        Transcribed from Dr. Tobias Fehenberger MATLAB code.
        See: https://www.fehenberger.de/#sourcecode
    """
    if y.shape[0] != 1:
        y = y.T
    if x.shape[0] != 1:
        x = x.T
    if constellation.shape[0] == 1:
        constellation = constellation.T

    M = constellation.size
    N = x.size
    P_X = np.zeros( (M,1) )
    
    x = x / np.sqrt( np.mean( np.abs( x )**2 ) ) # normalize such that var(X)=1
    y = y / np.sqrt( np.mean( np.abs( y )**2 ) ) # normalize such that var(Y)=1

    ## Get X in Integer Representation
    xint = np.argmin( np.abs( x - constellation )**2, axis=0)

    fun = lambda h: np.dot( h*x-y, np.conj( h*x-y ).T )
    h = fminbound( fun, 0,2)
    N0 = np.real( (1-h**2)/h**2 )
    y = y / h

    ## Find constellation and empirical input distribution
    for s in np.arange(M):
        P_X[s] = np.sum( xint==s ) / N
        
    ## Monte Carlo estimation of (a lower bound to) the mutual information I(XY)
    qYonX = 1 / ( np.pi*N0 ) * np.exp( ( -(np.real(y)-np.real(x))**2 -(np.imag(y)-np.imag(x))**2 ) / N0 )
    
    qY = 0
    for ii in np.arange(M):
        qY = qY + P_X[ii] * (1/(np.pi*N0)*np.exp((-(np.real(y)-np.real(constellation[ii,0]))**2-(np.imag(y)-np.imag(constellation[ii,0]))**2)/N0))
    
    realmin = np.finfo(float).tiny
    MI=1/N*np.sum(np.log2(np.maximum(qYonX,realmin)/np.maximum(qY,realmin)))

    return MI

def generateBitVectors(N, M):
    # Generates N bit vectors with M bits
    w = int(np.log2(M))
    d = np.zeros((N,w))
    r = np.random.randint(low=0, high=M, size=(N,) )
    for ii in range(N):
        d[ii,:] = np.array( [ float(x) for x in np.binary_repr(r[ii],width=w) ] )
    return d

def generateUniqueBitVectors(M):
    # Generates log2(M) unique bit vectors with M bits
    w = int(np.log2(M))
    d = np.zeros((M,w))
    for ii in range(M):
        d[ii,:] = np.array( [ float(x) for x in np.binary_repr(ii,width=w) ] )
    return d