import tensorflow as tf
import numpy as np

import claude.claudeflow.helper as cfh
import claude.utils as cu

c = 299792458
h = 6.6261e-34

def defaultParameters(D=16.4640, Fc=1.9341e+14, precision='double'):

    param = cu.AttrDict()
    param.nChannels       = 5
    param.nSpans          = 10
    param.nPol            = 2
    param.sps             = 16
    param.nSamples        = 2**10
    param.Rs              = 32e9
    param.noiseEnabled    = True
    param.noiseFigure     = 5


    param.chSpacing       = 50e9 # GHz
    param.spanLength      = 100  # km
    param.alpha           = 0.2  # db/km
    param.gamma           = 1.3  # W^-1/km
    param.D               = D    # ps/(km nm)
    param.simplified      = 1
    param.Fc              = Fc
    param.precision       = precision

    return param


# numpy functions
def alphalin(alpha):
    return alpha / ( 10 * np.log10( np.exp(1) ) ) * 1e-3; 

def dispF(beta2, length, omega):
    return np.exp( 1j * beta2 / 2 * omega**2 * length * 1e3 )

def phaseF(tau, omega):
    return np.exp( -1j * omega * tau )

def Hxpm(alpha, gamma, beta2, delta_f, spanLength, omega, simplified):
    """
    check https://ieeexplore.ieee.org/document/5709960
    see equations 14c and 15c, that is where the equations from this function are from.
    """
    C = ( 8 * gamma * 1e-3) / 9
    dBeta = -2 * np.pi * beta2 * delta_f
    alphalin_ = alphalin(alpha)
    if simplified == 1: # without pulse broadening dispersion
        nom     = ( np.exp( (-alphalin_ + 1j * dBeta * omega) * spanLength * 1e3 ) -1 );
        denom   = ( -alphalin_ + 1j * dBeta * omega );
    elif simplified == 2: # without pulse broadening dispersion and attenuation in nom
        nom     = 1;
        denom   = ( -alphalin_ + 1j * dBeta * omega );
    return C * nom / denom;

def model(param, TxSignal):
    nChannels      = param.nChannels
    nSpans         = param.nSpans
    sps            = param.sps
    nSamples       = param.nSamples
    Rs             = param.Rs
    noiseEnabled   = param.noiseEnabled

    # Nonlinear Fiber Channel:
    Fc              = param.Fc                    # Hz
    chSpacing       = param.chSpacing             # Hz
    spanLength      = param.spanLength            # km
    alpha           = param.alpha                 # db/km
    gamma           = param.gamma                 # W^-1/km
    D               = param.D                     # ps/(km nm)
    simplified      = param.simplified
    precision       = param.precision

    Fs        = sps * Rs
    L         = sps * nSamples
    middle    = nChannels // 2

    carrier_freqs   = chSpacing * (1 + np.arange(-nChannels // 2, nChannels // 2))
    lambda_         = c / (carrier_freqs + Fc)

    beta2 = -D * 1e-6 * ( (lambda_[middle])**2 ) / (2 * np.pi * c)

    # delta f (difference in frequency) from each channel to middle channel
    delta_f = np.zeros((nChannels, nChannels))
    for probe in range(nChannels):
        delta_f[probe, :] = (c / lambda_ - c / lambda_[probe])

    # delta beta
    delta_beta = -2 * np.pi * beta2 * delta_f[middle,:];
    # delta_T walkoff in time after spanLength propagation
    delta_T_per_span = delta_beta * spanLength * 1e3;

    # omega axis
    omega = cu.omegaAxis(L, Fs);

    if precision == 'single':
        complexType = tf.complex64
        realType    = tf.float32
    elif precision == 'double':
        complexType = tf.complex128
        realType    = tf.float64

    # tf constants
    one        = tf.constant( 1, realType )
    two        = tf.constant( 2, realType )
    ten        = tf.constant( 10, realType )
    zeroOneCpx = tf.constant( 1j, complexType )
    twoZeroCpx = tf.constant( 2+0*1j, complexType )

    ## EDFA
    tf_h          = tf.constant( h, realType )
    tf_c          = tf.constant( c, realType )
    EDFAGain      = tf.constant( spanLength * alpha, realType )
    noiseFigure   = tf.constant( param.noiseFigure, realType )
    tf_lambda_mid = tf.constant( lambda_[middle], realType )
    tf_Fs         = tf.constant( Fs, realType )

    G = cfh.dB2lin( EDFAGain, 'dB' )
    noiseFigureLin = tf.pow( ten, noiseFigure/ten )
    nsp = ( G * noiseFigureLin - one ) / ( two * (G-one) ) # Agrawal, Fiber-optic communication, Edition 4, eq. 7.2.15
    Pn = tf.identity( (G-one) * nsp * tf_h * tf_c / tf_lambda_mid, name='noiseDensity' ) * tf_Fs

    # tf functions
    def tf_abs2(x):
        return tf.square( tf.abs(x) )

    def xpolmStep(u_x, u_y):
        u_yx = zeroOneCpx * u_x * tf.math.conj(u_y)
        u_xy = zeroOneCpx * u_y * tf.math.conj(u_x)
        return u_yx, u_xy

    def xpmStep(u_x, u_y):
        u_xx  = tf.cast( two * tf_abs2(u_x) + tf_abs2(u_y), complexType)
        u_yy  = tf.cast( two * tf_abs2(u_y) + tf_abs2(u_x), complexType)
        return u_xx, u_yy

    def crosstalkXpol(phi_x, phi_y, w_yx, w_xy, u):
        rx = tf.exp( zeroOneCpx * (phi_x + phi_y) / twoZeroCpx) \
             * ( tf.exp( zeroOneCpx * (phi_x - phi_y) / twoZeroCpx) \
                 * tf.cast( tf.sqrt( one - tf_abs2(w_xy) ), complexType ) \
                 * u[:, 0, :] + w_yx * u[:, 1, :] )
        return rx

    def crosstalkYpol(phi_x, phi_y, w_yx, w_xy, u):
        ry = tf.exp( zeroOneCpx * (phi_x + phi_y) / twoZeroCpx) \
             * ( tf.exp( -zeroOneCpx * (phi_x - phi_y) / twoZeroCpx) \
                 * tf.cast( tf.sqrt( one - tf_abs2(w_yx) ), complexType ) \
                 * u[:, 1, :] + w_xy * u[:, 0, :] )
        return ry


    u = [ [ tf.zeros(tf.shape(TxSignal[0]), complexType) \
            for _ in range(nChannels) ] \
            for _ in range(nSpans+1) ]
    filters = [ [ tf.constant( Hxpm(alpha, gamma, beta2, delta_f[probe,mm], spanLength, omega, simplified), complexType) \
                  for mm in range(nChannels) ] \
                  for probe in range(nChannels) ]

    dispF_per_span = tf.constant( dispF(beta2, spanLength, omega), complexType )
    phaseF_per_span = [ tf.constant( phaseF(dT, omega), complexType ) for dT in delta_T_per_span ]

    for nn in range(nSpans):    
        # This is weird but works
        if nn==0:
            for probe in range(nChannels):        
                u[nn][probe] = u[nn][probe] + TxSignal[probe]
        # weird end

        for probe in range(nChannels):

            w_yx  = tf.zeros( (L,), complexType )
            w_xy  = tf.zeros( (L,), complexType )
            phi_x = tf.zeros( (L,), complexType )
            phi_y = tf.zeros( (L,), complexType )

            for mm in range(nChannels):
                if mm==probe:
                    continue;

                # Waveform at the beginning of the span
                u_tau_x = u[nn][mm][:, 0, :]
                u_tau_y = u[nn][mm][:, 1, :]
                # calculate nonlinearities
                u_yx, u_xy = xpolmStep( u_tau_x, u_tau_y )
                u_xx, u_yy = xpmStep( u_tau_x, u_tau_y )

                # xpm filters
                Hxpm_ = filters[probe][mm]

                u_yx_f = tf.signal.ifft( tf.signal.fft(u_yx) * Hxpm_ )
                u_xy_f = tf.signal.ifft( tf.signal.fft(u_xy) * Hxpm_ )
                u_xx_f = tf.signal.ifft( tf.signal.fft(u_xx) * Hxpm_ )
                u_yy_f = tf.signal.ifft( tf.signal.fft(u_yy) * Hxpm_ )

                w_yx = w_yx + u_yx_f
                w_xy = w_xy + u_xy_f
                phi_x = phi_x + u_xx_f
                phi_y = phi_y + u_yy_f

            rx = crosstalkXpol(phi_x, phi_y, w_yx, w_xy, u[nn][probe])
            ry = crosstalkYpol(phi_x, phi_y, w_yx, w_xy, u[nn][probe])

            rx_f = tf.signal.ifft( tf.signal.fft(rx) * dispF_per_span * phaseF_per_span[probe] )
            ry_f = tf.signal.ifft( tf.signal.fft(ry) * dispF_per_span * phaseF_per_span[probe] )

            u[nn + 1][probe] = u[nn + 1][probe] + tf.stack([rx_f, ry_f], axis=1)

            if noiseEnabled:
                noise_shape = tf.shape( u[nn + 1][probe] )
                noise  = tf.complex( tf.random.normal( noise_shape, mean=0.0, stddev=1.0, dtype=realType ),\
                                     tf.random.normal( noise_shape, mean=0.0, stddev=1.0, dtype=realType ) )
                tf.compat.v1.add_to_collection( "checkpoints", noise )
                u[nn + 1][probe] = u[nn + 1][probe] + tf.cast( tf.sqrt( Pn/two ), complexType ) * noise

    # dispersion compensation and constant phase rotation compensation
    dispF_comp = tf.constant( dispF(-beta2, nSpans * spanLength, omega), complexType )
    phaseF_comp = [tf.constant( phaseF(-nSpans * dT, omega), complexType ) for dT in delta_T_per_span]

    u_out = []
    for probe in range(nChannels):
        u_split = tf.split(u[nSpans][probe], [1, 1], axis=1)
        filter_comp = tf.expand_dims(dispF_comp, 0) * tf.expand_dims(phaseF_comp[probe], 0)
        u_split = [ tf.signal.ifft( tf.signal.fft( tf.squeeze(x) ) * filter_comp ) for x in u_split ]
        u_out.append( tf.stack(u_split, axis=1) )

    return u_out