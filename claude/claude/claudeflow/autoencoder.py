import tensorflow as tf
import numpy as np
from tensorflow.compat.v1.layers import Dense, Dropout

import claude.utils as cu
import claude.claudeflow.helper as cfh

def _layer_summary(layer_name, dtype=tf.float32):
    with tf.compat.v1.variable_scope(layer_name, reuse=tf.compat.v1.AUTO_REUSE):
        tf.compat.v1.summary.histogram('weights', tf.compat.v1.get_variable("kernel", dtype=dtype))
        tf.compat.v1.summary.histogram('bias', tf.compat.v1.get_variable("bias", dtype=dtype))

def encoder(x, aeParam, bits=False, toComplex=False, summaries=False, dropout=False, dropout_fun=Dropout, keep_prob=1., name='encoder'):
    if bits:
        xSeed = tf.constant(cu.generateUniqueBitVectors(aeParam.constellationOrder), x.dtype)
    else:
        xSeed = tf.linalg.eye(aeParam.constellationOrder, dtype=x.dtype)
    symbols = _encoder(x, aeParam.nHidden, aeParam.nLayers, aeParam.activation, nOutput=aeParam.constellationDim, summaries=summaries, dropout=dropout, dropout_fun=dropout_fun, keep_prob=keep_prob, name=name)
    constellation = _encoder(xSeed, aeParam.nHidden, aeParam.nLayers, aeParam.activation, nOutput=aeParam.constellationDim, summaries=summaries, dropout=dropout, dropout_fun=dropout_fun, keep_prob=keep_prob, name=name)

    norm_factor = cfh.norm_factor(constellation)

    symbols = norm_factor * symbols
    constellation = norm_factor * constellation

    if toComplex:
        symbols = cfh.real2complex(symbols)
        constellation = cfh.real2complex(constellation)

    return symbols, constellation

def _encoder(layer, nHidden, nLayers, activation, nOutput=2, summaries=False, kernel_initializer='glorot_uniform', dropout=False, dropout_fun=Dropout, keep_prob=1., name='encoder'):
    for i in range(nLayers):
        layer_name = name+str(i)
        layer = Dense(nHidden, activation=activation, kernel_initializer=kernel_initializer, _reuse=tf.compat.v1.AUTO_REUSE, name=layer_name)(layer)
        
        if summaries:
            _layer_summary(layer_name, layer.dtype)

        if dropout:
            layer = dropout_fun(keep_prob)(layer)

    layer_name = name+'_out'
    layer = Dense(nOutput, _reuse=tf.compat.v1.AUTO_REUSE, name=layer_name)(layer)

    if summaries:
        _layer_summary(layer_name, layer.dtype)

    return layer

def decoder(x, aeParam, bits=False, fromComplex=False, summaries=False, dropout=False, dropout_fun=Dropout, keep_prob=1., name='decoder'):
    if fromComplex:
        x = cfh.complex2real(x)

    if bits:
        outDim = np.log2(aeParam.constellationOrder)
    else:
        outDim = aeParam.constellationOrder
        
    return _decoder(x, aeParam.nHidden, aeParam.nLayers, aeParam.activation, outDim, summaries=summaries, dropout=dropout, dropout_fun=dropout_fun, keep_prob=keep_prob, name=name)

def _decoder(layer, nHidden, nLayers, activation, M, summaries=False, dropout=False, dropout_fun=Dropout, keep_prob=1., name='decoder'):
    for i in range(nLayers):
        layer_name = name+str(i)
        layer = Dense(nHidden, activation=activation, _reuse=tf.compat.v1.AUTO_REUSE, name=layer_name)(layer)

        if summaries:
            _layer_summary(layer_name, layer.dtype)

        if dropout:
            layer = dropout_fun(layer, keep_prob)

    layer_name = name+'_out'
    layer = Dense(M, _reuse=tf.compat.v1.AUTO_REUSE, name=layer_name)(layer)

    if summaries:
        _layer_summary(layer_name, layer.dtype)

    return layer