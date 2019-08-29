import tensorflow as tf
import numpy as np

import claude.claudeflow.helper as cfh
import claude.utils as cu
import claude.tx as tx

def defaultParameters(precision='double'):
	param = cu.AttrDict()
	param.nPol       = 2

	param.sps        = 16
	param.nSamples   = 1024
	param.rollOff    = 0.05
	param.filterSpan = 128
	param.optimizeP  = False

	param.PdBm       = 1
	param.Rs         = 32e9
	param.channels   = np.array( [-100., -50., 0., 50., 100.] )
	param.nChannels  = len(param.channels)

	if precision=='single':
		param.realType    = tf.float32
		param.complexType = tf.complex64
	else:
		param.realType    = tf.float64
		param.complexType = tf.complex128

	return param

def wdmTransmitter(symbols, param, frequencyShift=True):
	sps        = param.sps
	filterSpan = param.filterSpan
	nChannels  = param.nChannels
	nPol       = param.nPol
	nSamples   = param.nSamples
	rollOff    = param.rollOff
	PdBm       = param.PdBm
	Rs         = param.Rs 
	channels   = param.channels

	realType    = param.realType
	complexType = param.complexType

	Fs          = sps * Rs
	t           = np.arange( nSamples * sps )
	frequencies = channels * 1e9

	symbols = cfh.upsample(symbols, sps, nSamples)
	signal = cfh.pulseshaper(symbols, rollOff, sps, filterSpan, nSamples)

	# power, signal normalization
	if param.optimizeP:
		P0 = tf.constant( cu.dB2lin( PdBm, 'dBm'), dtype=realType )
		P0 = tf.contrib.distributions.softplus_inverse(P0)
		P0 = tf.nn.softplus( tf.Variable( P0 ), name='Power' )
	else:
		P0 = tf.constant( cu.dB2lin( PdBm, 'dBm'), dtype=realType, name='Power' )

	norm = tf.math.rsqrt( tf.reduce_mean( tf.square( tf.abs( signal ) ), keepdims=True, axis=-1 ) )
	signal = tf.sqrt( tf.cast( P0, dtype=signal.dtype ) ) * tf.cast( norm, dtype=signal.dtype ) * signal

	if frequencyShift:
		# frequency shift
		np_fShift = np.stack( [np.exp( 2j * np.pi * f/Fs * t ) for f in frequencies] )
		txFreqShift = tf.expand_dims( tf.constant( np_fShift, dtype=signal.dtype ), axis=1 )
		signal = signal * txFreqShift

		# combine channels
		signal = tf.reduce_sum( signal, axis=1 )

	return signal

def wdmReceiver(signal, param, frequencyShift=True):
	sps        = param.sps
	filterSpan = param.filterSpan
	nChannels  = param.nChannels
	nPol       = param.nPol
	nSamples   = param.nSamples
	rollOff    = param.rollOff
	Rs         = param.Rs 
	channels   = param.channels

	Fs          = sps * Rs
	t           = np.arange( nSamples * sps )
	frequencies = channels * 1e9

	if frequencyShift:
		# frequency shift    
		np_fShift = np.stack( [ np.exp( 2j * np.pi * f/Fs * t ) for f in frequencies[::-1] ] )
		rxFreqShift = tf.expand_dims( tf.constant( np_fShift, dtype=signal.dtype ), axis=1 )
		signal = tf.expand_dims( signal, axis=1 )
		signal = tf.tile( signal, [1, nChannels, 1, 1] ) * rxFreqShift

	# matched filter
	signal = cfh.pulseshaper(signal, rollOff, sps, filterSpan, nSamples)
	symbols = cfh.downsample(signal, sps, nSamples)

	# normalization
	norm = tf.math.rsqrt( tf.reduce_mean( tf.square( tf.abs( symbols ) ), keepdims=True, axis=-1 ) )
	symbols = symbols * tf.cast( norm, dtype=symbols.dtype )

	return symbols