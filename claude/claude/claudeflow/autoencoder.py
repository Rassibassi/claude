import tensorflow as tf

def _layer_summary(layer_name, dtype=tf.float32):
    with tf.variable_scope(layer_name, reuse=True):
        tf.summary.histogram('weights', tf.get_variable("kernel", dtype=dtype))
        tf.summary.histogram('bias', tf.get_variable("bias", dtype=dtype))

def encoder(layer,nHidden,nLayers,activation,dropout=False,dropout_fun=tf.nn.dropout,keep_prob=1.,nOutput=2,summaries=False,name='encoder'):
    for i in range(nLayers):
        layer_name = name+str(i)
        layer = tf.layers.dense(layer, nHidden, activation=activation, reuse=tf.AUTO_REUSE, name=layer_name)
        
        if summaries:
            _layer_summary(layer_name, layer.dtype)

        if dropout:
            layer = dropout_fun(layer, keep_prob)

    layer_name = name+'_out'
    layer = tf.layers.dense(layer, nOutput, reuse=tf.AUTO_REUSE,name=layer_name)

    if summaries:
        _layer_summary(layer_name, layer.dtype)

    return layer

def decoder(layer,nHidden,nLayers,activation,M,dropout=False,dropout_fun=tf.nn.dropout,keep_prob=1.,summaries=False,name='decoder'):
    for i in range(nLayers):
        layer_name = name+str(i)
        layer = tf.layers.dense(layer, nHidden, activation=activation, reuse=tf.AUTO_REUSE, name=layer_name)

        if summaries:
            _layer_summary(layer_name, layer.dtype)

        if dropout:
            layer = dropout_fun(layer, keep_prob)

    layer_name = name+'_out'
    layer = tf.layers.dense(layer, M, reuse=tf.AUTO_REUSE, name=layer_name)

    if summaries:
        _layer_summary(layer_name, layer.dtype)

    return layer