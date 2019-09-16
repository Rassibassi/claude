import tensorflow as tf
import numpy as np

import claude.claudeflow.helper as cfh
import claude.utils as cu
import claude.tx as tx

c = 299792458

def defaultParameters(D=16.4640, Fc=1.9341e+14, precision='double'):

	lambda_ = c / Fc
	beta2 = D*1e-6*lambda_**2/(2*np.pi*c)

	param = cu.AttrDict()
	param.M       	     = 4
	param.nPol       	 = 2
	 
	param.sps        	 = 16
	param.nSamples   	 = 1024
	param.rollOff    	 = 0.05
	param.filterSpan 	 = 128
	param.optimizeP  	 = False
	 
	param.PdBm       	 = 1
	param.Rs         	 = 32e9
	param.channels   	 = np.array( [-100., -50., 0., 50., 100.] )
	param.nChannels  	 = len(param.channels)
	param.frequencyShift = True

	param.dispersionCompensation = False
	param.beta2 				 = beta2 	 # with D = 16.464
	param.dz     				 = 1000000.0 # 10 * 100 * 1e3
	param.Fs 					 = 5.1200e+11
	param.N 					 = param.sps * param.nSamples

	if precision=='single':
		param.realType    = tf.float32
		param.complexType = tf.complex64
	else:
		param.realType    = tf.float64
		param.complexType = tf.complex128

	return param

def wdmTransmitter(symbols, param):
	sps        	   = param.sps
	filterSpan 	   = param.filterSpan
	nChannels  	   = param.nChannels
	nPol       	   = param.nPol
	nSamples   	   = param.nSamples
	rollOff    	   = param.rollOff
	PdBm       	   = param.PdBm
	Rs         	   = param.Rs 
	channels   	   = param.channels
	frequencyShift = param.frequencyShift

	realType    = param.realType
	complexType = param.complexType

	Fs          = sps * Rs
	t           = np.arange( nSamples * sps )
	frequencies = channels * 1e9

	symbols = cfh.upsample(symbols, sps, nSamples)
	signal  = cfh.pulseshaper(symbols, rollOff, sps, filterSpan, nSamples)
	signal  = signal * tf.sqrt( tf.constant( sps, dtype=complexType ) )

	# power, signal normalization
	P0 = tf.constant( cu.dB2lin( PdBm, 'dBm'), dtype=realType )

	if param.optimizeP:
		P0 = tf.contrib.distributions.softplus_inverse(P0)
		P0 = tf.nn.softplus( tf.Variable( P0 ) )

	tf.identity( P0, name='P0' )
	tf.identity( cfh.lin2dB( P0, 'dBm' ), name='PdBm' )

	normP0 = tf.identity( P0 / nPol, name='normP0' )

	signal 	= tf.cast( tf.sqrt( normP0 ), dtype=complexType) * signal

	if frequencyShift:
		# frequency shift
		np_fShift 	= np.stack( [np.exp( 2j * np.pi * f/Fs * t ) for f in frequencies] )
		txFreqShift = tf.expand_dims( tf.constant( np_fShift, dtype=complexType ), axis=1 )
		signal 		= signal * txFreqShift

		# combine channels
		signal = tf.reduce_sum( signal, axis=1 )

	return signal

def wdmReceiver(signal, param):
	sps        	   = param.sps
	filterSpan 	   = param.filterSpan
	nChannels  	   = param.nChannels
	nPol       	   = param.nPol
	nSamples   	   = param.nSamples
	rollOff    	   = param.rollOff
	Rs         	   = param.Rs 
	channels   	   = param.channels
	frequencyShift = param.frequencyShift

	realType    = param.realType
	complexType = param.complexType

	dispersionCompensation = param.dispersionCompensation
	beta2 				   = param.beta2
	dz 					   = param.dz
	Fs 					   = param.Fs

	Fs          = sps * Rs
	t           = np.arange( nSamples * sps )
	frequencies = channels * 1e9	
	
	if dispersionCompensation:
		beta2 	= tf.constant( beta2, realType )
		dz 		= tf.constant( dz, realType )
		signal 	= cfh.dispersion_compensation(signal, beta2, dz, nSamples*sps, Fs)

	if frequencyShift:
		# frequency shift    
		np_fShift 	= np.stack( [ np.exp( 2j * np.pi * f/Fs * t ) for f in frequencies[::-1] ] )
		rxFreqShift = tf.expand_dims( tf.constant( np_fShift, dtype=signal.dtype ), axis=1 )
		signal 		= tf.expand_dims( signal, axis=1 )
		signal 		= tf.tile( signal, [1, nChannels, 1, 1] ) * rxFreqShift

	# matched filter
	signal  = cfh.pulseshaper(signal, rollOff, sps, filterSpan, nSamples)
	signal  = signal * tf.math.rsqrt( tf.constant( sps, dtype=complexType ) )
	symbols = cfh.downsample(signal, sps, nSamples)

	return symbols