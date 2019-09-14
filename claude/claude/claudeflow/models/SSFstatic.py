import tensorflow as tf
import numpy as np
import numpy.matlib as matlib

import claude.claudeflow.helper as cfh
import claude.utils as cu

c = 299792458

def logStepSizes(spanLength, alpha, nSteps):
	# Bosco, G., et al.
	# "Suppression of spurious tones induced by the split-step method in fiber systems simulation."
	# IEEE Photonics Technology Letters 12.5 (2000): 489-491.
	alphalin = alpha / ( 10 * np.log10( np.exp(1) ) )
	n = np.arange(1,nSteps+1)
	sigma = (1-np.exp(-2*alphalin*spanLength))/nSteps
	h = -1/(2*alphalin) * np.log( (1-n*sigma) / (1-(n-1)*sigma) )
	return h

def randomizeSteps(stepSizes, spanLength, nSpans, sigma=0.01):
	stepSizes = matlib.repmat(stepSizes, nSpans, 1)
	stepSizes = stepSizes + np.random.normal(0, sigma*stepSizes[0,0], size=stepSizes.shape)
	stepSizes[:,-1] = stepSizes[:,-1] - (np.sum( stepSizes, axis=1 ) - spanLength)
	return stepSizes

def defaultParameters(D=16.4640, Fc=1.9341e+14, precision='double'):

	lambda_ = c/Fc
	beta2 = D*1e-6*lambda_**2/(2*np.pi*c)

	param = cu.AttrDict()
	param.Fs = 5.1200e+11
	param.N = 16 * 1024
	param.nSteps = 1
	param.stepSize = 100
	param.ampScheme = 'EDFA'
	param.noiseEnabled = True
	param.manakovEnabled = True
	param.dispersionCompensationEnabled = False # inline dispersion compensation
	param.checkpointInverval = 2

	param.nPol = 2

	param.lambda_ = lambda_ # Wavelength	
	param.Fc = Fc

	param.D = D	
	param.alpha = 0.2
	param.beta2 = beta2
	param.gamma = 1.3
	param.nSpans = 10	
	param.spanLength = 100
	param.noiseFigure = 5

	param.intType = tf.int32
	if precision == 'single':
		param.realType    = tf.float32
		param.complexType = tf.complex64
	else:
		param.realType    = tf.float64
		param.complexType = tf.complex128

	param.stepSizeTemplate = logStepSizes(param.spanLength, param.alpha, param.nSteps)
	
	return param
	
def model(param, signal):
	# TODO CHECK VALUE mod( spanLength / stepSize ) == 0 etc

	# TF config
	intType = param.intType
	realType = param.realType
	complexType = param.complexType
	checkpointInverval = param.checkpointInverval

	# Real constants
	h     = tf.constant( 6.6261e-34, realType )
	c     = tf.constant( 299792458, realType )
	pi    = tf.constant( np.pi, realType )
	one   = tf.constant( 1, realType )
	two   = tf.constant( 2, realType )
	three = tf.constant( 3, realType )
	ten   = tf.constant( 10, realType )
	eight = tf.constant( 8, realType )
	nine  = tf.constant( 9, realType )
	e3    = tf.constant( 1e3, realType )
	e6    = tf.constant( 1e6, realType )
	em3   = tf.constant( 1e-3, realType )
	em6   = tf.constant( 1e-6, realType )

	# Complex constants
	minusCpx     = tf.constant( -1, complexType )
	zeroOneCpx   = tf.constant( 0+1j, complexType )
	twoZeroCpx   = tf.constant( 2+0*1j, complexType )

	N                  = param.N
	Fs                 = param.Fs
	nPol               = param.nPol
	stepSize 		   = param.stepSize
	nSpans             = param.nSpans
	nSteps 			   = param.nSteps

	# Fiber constants
	Fc                 = tf.constant( param.Fc, realType )
	D                  = tf.constant( param.D, realType )
	gamma              = tf.constant( param.gamma, realType ) * em3
	alpha              = tf.constant( param.alpha, realType ) * em3	
	spanLength         = tf.constant( param.spanLength, realType ) * e3
	noiseFigure        = tf.constant( param.noiseFigure, realType )
	alphalin           = alpha / ( ten * cfh.log10( tf.exp(one) ) )
	lambda_            = c / Fc # 'lambda' is part of python's syntax
	beta2              = D*em6 * tf.square(lambda_) / ( two * pi * c )

	EDFAGain           = spanLength * alpha

	ampScheme          = param.ampScheme
	noiseEnabled       = param.noiseEnabled
	manakovEnabled     = param.manakovEnabled
	dispersionCompensationEnabled = param.dispersionCompensationEnabled

	abs2 = lambda x: tf.square(tf.abs(x))
	hFilter = lambda H, s: tf.signal.ifft( H * tf.signal.fft( s ) )

	# step size
	if isinstance( stepSize, list ) or isinstance( stepSize, np.ndarray ):
		# step size is a list/ndarray
		# shape: 1 x nSteps
		# every span has the same step profile
		
		asymmetricSteps = True
		stepSizeIsTensor = False

		stepSize = tf.constant( np.array( stepSize ), realType ) * e3

		if np.sum( stepSize ) != spanLength:
			raise ValueError('Stepsizes must sum up to the span length.')

	elif isinstance( stepSize, tf.Tensor ):
		# step size is a tensor/placeholder
		# shape: nSpans x nSteps
		# each span has a different step profile
		
		asymmetricSteps = True
		stepSizeIsTensor = True

		stepSize = stepSize * e3
	else:
		# step size is a scalar value and always the same
		# shape: 1 x 1

		asymmetricSteps = False
		stepSizeIsTensor = False

		stepSize = tf.constant( param.stepSize, realType ) * e3
		nz       = tf.math.ceil( spanLength / stepSize )	# Number of steps to take
		dz = spanLength / nz 					# Distance per step

	## EDFA
	G = cfh.dB2lin( EDFAGain, 'dB' )
	noiseFigureLin = tf.pow( ten, noiseFigure/ten )
	nsp = ( G * noiseFigureLin - one ) / ( two * (G-one) ) # Agrawal, Fiber-optic communication, Edition 4, eq. 7.2.15
	Pn = tf.identity( (G-one) * nsp * h * c / lambda_, name='noiseDensity' ) * Fs

	## Attenuation filter
	attenuation = tf.exp( -alphalin / two );

	## Dispersion Filter
	omega = tf.constant( cu.omegaAxis(N,Fs), realType )
	dispersion = tf.exp( minusCpx * zeroOneCpx * tf.cast( beta2/two * tf.square(omega), complexType ) )

	fLinear = tf.cast( attenuation, complexType ) * dispersion
	fLinear = tf.expand_dims( fLinear, 0 )

	if dispersionCompensationEnabled:
		dispersionComp = tf.exp( zeroOneCpx * tf.cast( beta2/two * tf.square(omega) * spanLength, complexType ) );
		dispersionComp = tf.expand_dims( dispersionComp, 0 )

	def nonlinearPhaseRot( s ):
		if nPol == 2:
			if manakovEnabled:
				phaseRotX = eight/nine * ( abs2( s[:,0,:] ) + abs2( s[:,1,:] ) )
				# Polarization rotation symmetric with Manakov
				phaseRotY = phaseRotX            
			else:
				phaseRotX = two/three * abs2( s[:,1,:] ) + abs2( s[:,0,:] )
				phaseRotY = two/three * abs2( s[:,0,:] ) + abs2( s[:,1,:] )

			phaseRot = tf.stack( [ phaseRotX, phaseRotY ], axis=1 )
			
		else:
			phaseRot  = abs2(s)
			
		return gamma * phaseRot

	def step_body( s ):    
		# nonlinear operator (without distance)
		phaseRot  = nonlinearPhaseRot( s )
		
		# cast into complex domain
		phaseRotCpx = tf.cast( phaseRot, complexType )
		dzCpx = tf.cast( dz, complexType )
		
		# Apply half dispersion and half attenuation
		s = hFilter( fLinear**(dzCpx/twoZeroCpx), s )

		# Apply nonlinearities
		s = s * tf.exp( zeroOneCpx * phaseRotCpx * dzCpx )

		# Apply half dispersion and half attenuation
		s = hFilter( fLinear**(dzCpx/twoZeroCpx), s )
		
		return s

	def span_body( s ):		
		if dispersionCompensationEnabled:
			s = hFilter( dispersionComp, s )

		# The EDFA amplifies and add noise at the end of the span
		if ampScheme == 'EDFA':
			s = tf.cast( tf.sqrt(G), complexType ) * s
			if noiseEnabled:
				noise  = tf.complex( tf.random.normal( tf.shape( s ), mean=0.0, stddev=1.0, dtype=realType ),\
									 tf.random.normal( tf.shape( s ), mean=0.0, stddev=1.0, dtype=realType ) )
				tf.compat.v1.add_to_collection( "checkpoints", noise )
				s = s + tf.cast( tf.sqrt( Pn/two ), complexType ) * noise
		
		return s

	print('nSpans: {}, nSteps: {}'.format(nSpans,nSteps), flush=True)

	signal = tf.identity( signal, name="Span_{:03d}".format(0))

	for span in range(nSpans):
		tf.compat.v1.add_to_collection( "checkpoints", signal )
		for step in range(nSteps):

			if asymmetricSteps:
				if not stepSizeIsTensor:
					# stepSize: 1 x nSteps
					dz = stepSize[step]
				else:
					# stepSize: nSpans x nSteps
					dz = stepSize[span,step]

			signal = step_body( signal )

			if (step+1) % checkpointInverval == 0:
				tf.compat.v1.add_to_collection( "checkpoints", signal )

		signal = span_body( signal )
		signal = tf.identity( signal, name="Span_{:03d}".format(span+1))

	return signal
