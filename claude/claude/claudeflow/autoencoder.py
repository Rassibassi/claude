import tensorflow as tf
from tensorflow.compat.v1.layers import Dense, Dropout

import claude.claudeflow.helper as cfh

def _layer_summary(layer_name, dtype=tf.float32):
    with tf.variable_scope(layer_name, reuse=tf.compat.v1.AUTO_REUSE):
        tf.summary.histogram('weights', tf.get_variable("kernel", dtype=dtype))
        tf.summary.histogram('bias', tf.get_variable("bias", dtype=dtype))

def encoder(x, aeParam, toComplex=False, summaries=False, dropout=False, dropout_fun=Dropout, keep_prob=1., name='encoder'):
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

def _encoder(layer, nHidden, nLayers, activation, nOutput=2, summaries=False, dropout=False, dropout_fun=Dropout, keep_prob=1., name='encoder'):
    for i in range(nLayers):
        layer_name = name+str(i)
        layer = Dense(nHidden, activation=activation, _reuse=tf.compat.v1.AUTO_REUSE, name=layer_name)(layer)
        
        if summaries:
            _layer_summary(layer_name, layer.dtype)

        if dropout:
            layer = dropout_fun(keep_prob)(layer)

    layer_name = name+'_out'
    layer = Dense(nOutput, _reuse=tf.compat.v1.AUTO_REUSE, name=layer_name)(layer)

    if summaries:
        _layer_summary(layer_name, layer.dtype)

    return layer

def decoder(x, aeParam, fromComplex=False, summaries=False, dropout=False, dropout_fun=Dropout, keep_prob=1., name='decoder'):
    if fromComplex:
        x = cfh.complex2real(x)

    return _decoder(x, aeParam.nHidden, aeParam.nLayers, aeParam.activation, aeParam.constellationOrder, summaries=summaries, dropout=dropout, dropout_fun=dropout_fun, keep_prob=keep_prob, name=name)

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