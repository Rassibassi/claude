import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def QAMencoder(x,c):
    comp = tf.transpose(tf.matmul(c,tf.cast(tf.transpose(x,(1,0)),c.dtype)),(1,0))
    return tf.concat([tf.real(comp),tf.imag(comp)],axis=1)

def IQ_abs(x):
    return tf.sqrt(tf.square(x[:,0])+tf.square(x[:,1]))

def IQ_norm(x,epsilon=1e-12):
    rmean = tf.reduce_mean( tf.square( IQ_abs(x) ) )
    rsqrt = tf.rsqrt(tf.maximum(rmean, epsilon))    
    return x*rsqrt

def logBase(x,base):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator

def log10(x):
    return logBase(x,10)

def log2(x):
    return logBase(x,2)

def softmax_MI(softmax, X, Px):
    MI = tf.reduce_mean( logBase( tf.reduce_sum( softmax*X, axis=-1) / Px, 2) )
    return MI

def gaussian_MI(x, y, constellation, M, dtype=tf.float64):
    """
    Computes mutual information with Gaussian auxiliary channel assumption and constellation with uniform porbability distribution
    x: (1, N), N normalized samples at the transmitter
    y: (1, N), N normalized observations at the receiver
    constellation: (1, M), normalized constellation of order M
    
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
        y = tf.linalg.transpose(y)
    if x.shape[0] != 1:
        x = tf.linalg.transpose(x)
    if constellation.shape[0] == 1:
        constellation = tf.linalg.transpose(constellation)

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
    
    qYonX = 1 / ( PI*N0 ) * tf.exp( ( -tf.square(tf.real(y)-tf.real(x)) -tf.square(tf.imag(y)-tf.imag(x)) ) / N0 )
    
    qY = []
    for ii in np.arange(M):
        temp = P_X * (1 / (PI * N0) * tf.exp( ( -tf.square(tf.real(y)-tf.real(constellation[ii,0])) -tf.square(tf.imag(y)-tf.imag(constellation[ii,0])) ) / N0) )
        qY.append(temp)
    qY = tf.reduce_sum( tf.concat(qY, axis=0), axis=0)
            
    MI = 1 / N * tf.reduce_sum( log2( tf.math.maximum(qYonX, REALMIN) / tf.math.maximum(qY, REALMIN) ) )
    
    return MI

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
    '''
        see https://github.com/tensorflow/tensorflow/issues/4814#issuecomment-314801758
    '''
    with tf.variable_scope(scope) as scope:
        metric_op, update_op = metric(*metric_args, **metric_kwargs)
        vars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(vars)
    return metric_op, update_op, reset_op