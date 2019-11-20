import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import claude.tx as tx
import claude.utils as cu

def i_fft(h, signal, transform_filter=True):
    if transform_filter:
        return tf.signal.ifft( tf.signal.fft(h) * tf.signal.fft(signal) )
    else:
        return tf.signal.ifft( h * tf.signal.fft(signal) )

def QAMencoder(X, constellation, realOutput=True):    
    Xcpx = tf.cast(X, constellation.dtype)
    comp = tf.squeeze(constellation @ tf.linalg.matrix_transpose(Xcpx), -2)

    if realOutput:
        return tf.stack([tf.math.real(comp),tf.math.imag(comp)],axis=-1)
    else:
        return comp

def upsample(symbols, sps, N):
    """
        symbols: (?, ?, ..., N), Tensor where most inner dimension is being upsampled
        sps: Samples per symbol, or upsampling factor
        N: Number of symbols/samples
    """

    zeroShape = tf.concat((tf.shape(symbols),        tf.constant([sps-1], tf.int32)), axis=0)
    newShape  = tf.concat((tf.shape(symbols)[0:-1], tf.constant([N*sps], tf.int32)), axis=0)

    symbols_stacked = tf.concat([tf.expand_dims(symbols, -1), tf.zeros(zeroShape, dtype=symbols.dtype)], axis=-1)

    return tf.reshape(symbols_stacked, newShape)

def downsample(signal, sps, N):
    """
        symbols: (?, ?, ..., N*sps), Tensor where most inner dimension is upsampled
        sps: Samples per symbol, or upsampling factor
        N: Number of symbols/samples
    """

    outerShape = tf.shape(signal)[0:-1]
    innerShape = tf.constant([N, sps], tf.int32)
    shape = tf.concat((outerShape, innerShape), axis=0)

    return tf.reshape(tf.expand_dims(signal, -1), shape)[..., 0]

def pulseshaper(symbols_up, rollOff, sps, span, N):
    p = tx.fftShiftZeroPad(tx.rrcos(rollOff, sps, span), N*sps)
    h = tf.expand_dims(tf.constant(p, dtype=symbols_up.dtype), axis=0)
    return i_fft(h, symbols_up)

def truncate(nSamples, *signals):
    signals = [signal[..., nSamples:-nSamples] for signal in signals]
    return signals
    
def dispersion_compensation(signal, beta2, distance, N, Fs):
    if signal.dtype == tf.complex64:
        realType = tf.float32
    else:
        realType = tf.float64
    
    omega       = tf.constant( cu.omegaAxis(N, Fs), realType )
    two         = tf.constant( 2, realType )

    zeroOneCpx  = tf.constant( 0+1j, signal.dtype )
    distanceCpx = tf.cast( distance, signal.dtype )    
    
    dispersion_compensation = tf.exp( zeroOneCpx * distanceCpx * tf.cast( beta2/two * tf.square(omega), signal.dtype ) )
    dispersion_compensation = tf.expand_dims( dispersion_compensation, 0 )
    signal = i_fft( dispersion_compensation, signal, transform_filter=False )
    return signal

def staticPhaseRotationCompensation(symbols, nPilots=None):
    '''
        see:
        J. Diniz et. al.
        "Low-complexity carrier phase recovery based on principal component analysis for square-QAM modulation formats"
        Optics Express, Vol. 27, Issue 11, pp. 15617-15626 (2019)
        https://orbit.dtu.dk/en/publications/lowcomplexity-carrier-phase-recovery-based-on-principal-component-analysis-for-squareqam-modulation-formats(c1132726-ce90-433a-8dc7-8ccb3e5bc180).html
    '''
    if symbols.dtype == tf.complex64:
        realType = tf.float32
    else:
        realType = tf.float64    
    
    HALF         = tf.constant( 0.5, realType)
    FOUR         = tf.constant( 4, realType)
    PI           = tf.constant( np.pi, realType)
    zeroOneCpx   = tf.constant( 0+1j, symbols.dtype)

    if nPilots is not None:
        pilot_symbols = symbols[..., 0:nPilots]
    else:
        pilot_symbols = symbols
    
    symbols2 = tf.square( pilot_symbols )
    symbols2vec = tf.stack([tf.math.real(symbols2), tf.math.imag(symbols2)], axis=-2)
    covarianceMatrix = tf.matmul(symbols2vec, tf.linalg.matrix_transpose(symbols2vec))
    eig = tf.linalg.eigh(covarianceMatrix)
    eigenVec = eig[-1][..., -1] # pick largest eigenvalue
    phi = HALF * tf.math.atan(eigenVec[..., 1] / eigenVec[..., 0]) - PI/FOUR
    
    phiCpx = tf.expand_dims( tf.cast(phi, symbols.dtype), -1 )
    
    symbols = symbols * tf.exp( -zeroOneCpx * phiCpx )
    
    return symbols

def tfarg(fn, x):
    """
        tf.argmin and tf.argmax only handle Tensors of < 6 dimensions. This fixes it.
        
        tfarg finds 'fn' (tf.argmin or tf.argmax) of the inner-most dimension of 'x', hence equivalent to tf.argmin(x, -1) or tf.argmax(x. -1).
    """
    return tf.reshape( fn( tf.reshape( x, [-1, tf.shape(x)[-1]] ), -1 ), tf.shape(x)[0:-1] )

def testPhases(constellation, txSymbols, rxSymbols, nDims, M, nTestPhases=4, nPilots=None):
    PI = tf.constant(np.pi, rxSymbols.dtype)
    zeroTwoCpx = tf.constant( 0+2j, rxSymbols.dtype)

    allRxSymbols = rxSymbols

    if nPilots is not None:
        rxSymbols = rxSymbols[..., 0:nPilots]
        txSymbols = txSymbols[..., 0:nPilots]
    
    tile_multiples = [1] * (nDims+1)
    tile_multiples[-1] = nTestPhases
    phi4rot = tf.cast( tf.range(0, 1, 1/nTestPhases), rxSymbols.dtype)
    
    rxSymbols4rot = tf.tile( tf.expand_dims(rxSymbols, -1), tile_multiples ) * tf.exp( -zeroTwoCpx * PI * phi4rot )
    
    tile_multiples = [1] * (nDims+2)
    tile_multiples[-1] = M
    rxSymbols4rot_tiled = tf.tile( tf.expand_dims( rxSymbols4rot, -1 ), tile_multiples )
    
    tile_multiples = [1] * (nDims+1)
    tile_multiples[-1] = M
    txSymbols_tiled = tf.tile( tf.expand_dims( txSymbols, -1 ), tile_multiples )
    txIdx = tf.argmin( tf.abs( txSymbols_tiled - constellation ), -1 )    
    
    rxIdx4rot = tfarg(tf.argmin, tf.abs( rxSymbols4rot_tiled - constellation ))
    errors4rot = tf.reduce_sum( tf.cast( tf.not_equal( tf.expand_dims(txIdx, -1), rxIdx4rot ), tf.int32 ), -2)
    
    rotIdx = tf.argmin( errors4rot, -1 )
    rotByThis = tf.expand_dims( tf.gather(phi4rot, rotIdx), -1 )

    allRxSymbols = allRxSymbols * tf.exp( -zeroTwoCpx * PI * rotByThis )
    
    return allRxSymbols

def real2complex(x):
    return tf.complex(x[...,0], x[...,1])

def complex2real(x, axis=-1):
    return tf.stack((tf.math.real(x), tf.math.imag(x)), axis=axis)

from tensorflow.python.framework import function

def norm_grad(x, dy):
    return tf.expand_dims(dy, axis=-1)*(x/(tf.norm(x, keepdims=True, axis=-1)+1.0e-19))
    
@function.Defun(tf.float64, tf.float64)
def norm_grad64(x, dy):
    return norm_grad(x, dy)

@function.Defun(tf.float32, tf.float32)
def norm_grad32(x, dy):
    return norm_grad(x, dy)

@function.Defun(tf.float64, grad_func=norm_grad64)
def norm64(x):
    return tf.norm(x, axis=-1)

@function.Defun(tf.float32, grad_func=norm_grad32)
def norm32(x):
    return tf.norm(x, axis=-1)

def norm(x):
    if x.dtype == tf.float32:
        return norm32(x)
    elif x.dtype == tf.float64:
        return norm64(x)

def norm_factor(constellation, epsilon=1e-12):
    if any([ constellation.dtype == x for x in [tf.complex64,tf.complex128] ]):
        castTo = constellation.dtype
        constellation = complex2real(constellation)
    else:
        castTo = False
    
    rmean = tf.reduce_mean( tf.square( norm(constellation) ) )
    normFactor = tf.math.rsqrt( tf.maximum(rmean, epsilon) )
    
    if castTo:
        return tf.cast(normFactor, castTo)
    else:
        return normFactor

def logBase(x,base):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator

def log10(x):
    return logBase(x,10)

def log2(x):
    return logBase(x,2)

def softmaxMI(softmax, X, Px):
    MI = tf.reduce_mean( log2( tf.reduce_sum( softmax*X, axis=-1) / Px) )
    return MI

def symbolErrorrate(constellation, txSymbols, rxSymbols, nDims, M, reduce_axis):
    tile_multiples = [1] * (nDims+1)
    tile_multiples[-1] = M
    
    rxSymbols_tiled = tf.tile( tf.expand_dims( rxSymbols, -1 ), tile_multiples )
    rxIdx = tf.argmin( tf.abs( rxSymbols_tiled - constellation ), axis=-1 )
    
    txSymbols_tiled = tf.tile( tf.expand_dims( txSymbols, -1 ), tile_multiples )
    txIdx = tf.argmin( tf.abs( txSymbols_tiled - constellation ), axis=-1 )
    
    errors = tf.reduce_sum( tf.cast( tf.not_equal( txIdx, rxIdx ), tf.int32 ), reduce_axis )
    errorrate = tf.reduce_mean( tf.cast( tf.not_equal( txIdx, rxIdx ), tf.float32 ), reduce_axis )
    
    return errorrate

def effectiveSNR(txSymbols, rxSymbols, signalPower, reduce_axis):
    estNoisePower = tf.reduce_mean( tf.square( tf.abs( txSymbols - rxSymbols ) ), reduce_axis )
    effSNR = lin2dB( signalPower / estNoisePower, 'dB' )

    return effSNR

def gaussianMI(x, y, constellation, M, dtype=tf.float64):
    """
        Computes mutual information with Gaussian auxiliary channel assumption and constellation with uniform porbability distribution

        x: (1, N), N normalized complex samples at the transmitter, where N is the batchSize/sampleSize
        y: (1, N), N normalized complex observations at the receiver, where N is the batchSize/sampleSize
        constellation: (1, M), normalized complex constellation of order M
        
        Transcribed from Dr. Tobias Fehenberger MATLAB code.
        See: https://www.fehenberger.de/#sourcecode
    """
    if len(constellation.shape) == 1:
        constellation = tf.expand_dims(constellation, axis=0)
    if len(y.shape) == 1:
        y = tf.expand_dims(y, axis=0)
    if len(x.shape) == 1:
        x = tf.expand_dims(x, axis=0)
    if y.shape[0] != 1:
        y = tf.linalg.matrix_transpose(y)
    if x.shape[0] != 1:
        x = tf.linalg.matrix_transpose(x)
    if constellation.shape[0] == 1:
        constellation = tf.linalg.matrix_transpose(constellation)

    N = tf.cast( tf.shape(x)[1], dtype )

    PI = tf.constant( np.pi, dtype=dtype )
    REALMIN = tf.constant( np.finfo(float).tiny, dtype=dtype )

    xint = tf.math.argmin(tf.square(tf.abs(x - constellation)), axis=0, output_type=tf.int32)
    x_count = tf.math.bincount(xint)
    x_count = tf.ensure_shape(x_count, (M,))
    P_X = tf.cast(x_count, dtype) / N
        
    N0 = tf.reduce_mean( tf.square( tf.abs(x-y) ) )
    
    qYonX = 1 / ( PI*N0 ) * tf.exp( ( -tf.square(tf.math.real(y)-tf.math.real(x)) -tf.square(tf.math.imag(y)-tf.math.imag(x)) ) / N0 )
    
    qY = []
    for ii in np.arange(M):
        temp = P_X[ii] * (1 / (PI * N0) * tf.exp( ( -tf.square(tf.math.real(y)-tf.math.real(constellation[ii,0])) -tf.square(tf.math.imag(y)-tf.math.imag(constellation[ii,0])) ) / N0) )
        qY.append(temp)
    qY = tf.reduce_sum( tf.concat(qY, axis=0), axis=0)
            
    MI = 1 / N * tf.reduce_sum( log2( tf.math.maximum(qYonX, REALMIN) / tf.math.maximum(qY, REALMIN) ) )
    
    return MI

def gaussianLLR( constellation, constellation_bits, Y, SNR_lin, M ):
    """
        Computes log likelihood ratio with Gaussian auxiliary channel assumption
        
        constellation: (1, M), where M is the constellation order
        constellation_bits: (m, M), where m=log2(M) the number of bits per symbol and M is the constellation order
        Y: (1, N), N normalized complex observations at the receiver, where N is the batchSize/sampleSize
        SNR_lin: SNR (linear) of Gaussian auxiliary channel
        M: Constellation order
    """
    
    M = int(M) # Order of constellations
    m = int(np.log2(M)) # Number of bits per symbol

    expAbs = lambda x,y,SNR_lin: tf.exp( -SNR_lin * tf.square( tf.abs( y - x ) ) )

    constellation_zero = tf.stack( [tf.boolean_mask( constellation, item ) for item in tf.split( tf.equal( constellation_bits, 0 ), num_or_size_splits=m, axis=0 )], axis=1 )
    constellation_zero.set_shape((int(M/2),m))
    constellation_one = tf.stack( [tf.boolean_mask( constellation, item ) for item in tf.split( tf.equal( constellation_bits, 1 ), num_or_size_splits=m, axis=0 )], axis=1 )
    constellation_one.set_shape((int(M/2),m))

    constellation_zero_flat = tf.reshape(constellation_zero,[-1])
    constellation_one_flat = tf.reshape(constellation_one,[-1])

    sum_zeros = tf.reduce_sum( tf.reshape( expAbs( tf.expand_dims(constellation_zero_flat, axis=1), Y, SNR_lin ), [int(M/2),m,-1] ), axis=0 )
    sum_ones = tf.reduce_sum( tf.reshape( expAbs( tf.expand_dims(constellation_one_flat, axis=1), Y, SNR_lin ), [int(M/2),m,-1] ), axis=0 )

    LLRs = tf.math.log( sum_zeros / sum_ones )

    return LLRs

def GMI( bits_reshaped, LLRs ):
    """
        Computes GMI from LLR and bits
        
        bits_reshaped: (m, N), where m the number of bits per symbol and N is the batchSize/sampleSize
        LLRs: (m, N), where m the number of bits per symbol and N is the batchSize/sampleSize
    """
    
    realType = LLRs.dtype
    bits_reshaped = tf.cast( bits_reshaped, realType)

    one = tf.constant( 1, realType )
    two = tf.constant( 2, realType )

    MI_per_bit = one - tf.reduce_mean( log2( one + tf.exp( ( two*bits_reshaped - one ) * LLRs ) ), axis=1)
    GMI = tf.reduce_sum( MI_per_bit )

    return GMI

def lin2dB(lin,dBtype):
    if dBtype == 'db' or dBtype == 'dB':
        fact = 0
    elif dBtype == 'dbm' or dBtype == 'dBm':
        fact = -30
    elif dBtype == 'dbu' or dBtype == 'dBu':
        fact = -60
    else:
        raise ValueError('dBtype can only be dB, dBm or dBu.')

    fact = tf.constant(fact,lin.dtype)
    ten = tf.constant(10,lin.dtype)

    return ten*log10(lin)-fact

def dB2lin(dB,dBtype):
    if dBtype == 'db' or dBtype == 'dB':
        fact = 0
    elif dBtype == 'dbm' or dBtype == 'dBm':
        fact = -30
    elif dBtype == 'dbu' or dBtype == 'dBu':
        fact = -60
    else:
        raise ValueError('dBtype can only be dB, dBm or dBu.')

    fact = tf.constant(fact,dB.dtype)
    ten = tf.constant(10,dB.dtype)

    return ten**( (dB+fact)/ten )