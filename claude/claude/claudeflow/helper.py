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

def IQ_abs(x):
    return tf.sqrt(tf.square(x[:,0])+tf.square(x[:,1]))

def IQ_norm(x,epsilon=1e-12):
    rmean = tf.reduce_mean( tf.square( IQ_abs(x) ) )
    rsqrt = tf.rsqrt(tf.maximum(rmean, epsilon))    
    return x*rsqrt

def logBase(x,base):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator

def log10(x):
    return logBase(x,10)

def log2(x):
    return logBase(x,2)

def softmaxMI(softmax, X, Px):
    MI = tf.reduce_mean( logBase( tf.reduce_sum( softmax*X, axis=-1) / Px, 2) )
    return MI

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
    
    # TODO
    # use following line if P_X is not uniform or empirical
    # xint = tf.math.argmin( tf.square( tf.abs( x - constellation ) ), axis=0)
    # ...
    # else:
    P_X = tf.constant( 1 / M, dtype=dtype)
    N0 = tf.reduce_mean( tf.square( tf.abs(x-y) ) )
    
    qYonX = 1 / ( PI*N0 ) * tf.exp( ( -tf.square(tf.math.real(y)-tf.math.real(x)) -tf.square(tf.math.imag(y)-tf.math.imag(x)) ) / N0 )
    
    qY = []
    for ii in np.arange(M):
        temp = P_X * (1 / (PI * N0) * tf.exp( ( -tf.square(tf.math.real(y)-tf.math.real(constellation[ii,0])) -tf.square(tf.math.imag(y)-tf.math.imag(constellation[ii,0])) ) / N0) )
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

def create_reset_metric(metric, scope='reset_metrics', *metric_args, **metric_kwargs):
    """
        see https://github.com/tensorflow/tensorflow/issues/4814#issuecomment-314801758
    """
    with tf.compat.v1.variable_scope(scope) as scope:
        metric_op, update_op = metric(*metric_args, **metric_kwargs)
        vars = tf.contrib.framework.get_variables(scope, collection=tf.compat.v1.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.compat.v1.variables_initializer(vars)
    return metric_op, update_op, reset_op