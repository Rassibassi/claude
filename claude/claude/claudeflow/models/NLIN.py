'''
    All functions in this file are copied from a MATLAB version
    provided by Dar et al. [https://arxiv.org/abs/1310.6137]
    and translated into Python.

    Please cite the above appropriately if used.
'''

import tensorflow as tf
import claude.utils as cu

def tfConstants(dtype=tf.float64):
    c = cu.AttrDict()
    c.two = tf.constant(2,dtype)
    c.three = tf.constant(3,dtype)
    c.four = tf.constant(4,dtype)
    c.nine = tf.constant(9,dtype)
    c.twelve = tf.constant(12,dtype)    
    c.sixteen = tf.constant(16,dtype)
    c.eightyone = tf.constant(81,dtype)
    return c

def calcInterChannelNLIN(chi,kur,P0,nPol,dtype=tf.float64):
    c = tfConstants(dtype=dtype)
    
    NLIN_inter = chi[0,:]+(kur-c.two)*chi[1,:]
    if nPol == 2:
        NLIN_inter = c.sixteen/c.eightyone*(NLIN_inter+c.two*chi[0,:]/c.four+(kur-c.two)*chi[1,:]/c.four)
    NLIN_inter = P0**c.three*NLIN_inter
    return NLIN_inter

def calcInterChannelNLINAddTerms(X,kur,P0,nPol,dtype=tf.float64):
    c = tfConstants(dtype=dtype)

    NLIN_inter_addTerms = c.four*X[0,:]+c.four*(kur-c.two)*X[1,:]+c.two*X[2,:]+(kur-c.two)*X[3,:]
    if nPol == 2:
        NLIN_inter_addTerms = c.sixteen/c.eightyone*( NLIN_inter_addTerms+c.two*X[0,:]+(kur-c.two)*X[1,:]+X[2,:] ) # +0*(param.kur-2)*X24)
    NLIN_inter_addTerms = P0**c.three*NLIN_inter_addTerms
    return NLIN_inter_addTerms

def calcIntraChannelNLIN(X,kur,kur3,P0,nPol,dtype=tf.float64):
    c = tfConstants(dtype=dtype)

    NLIN_intra = c.two*X[1,:]+(kur-c.two)*(c.four*X[2,:]+X[3,:])+(kur3-c.nine*kur+c.twelve)*X[4,:]-(kur-c.two)**c.two*X[0,:]
    if nPol == 2:
        NLIN_intra = c.sixteen/c.eightyone*(NLIN_intra+X[1,:]+(kur-c.two)*X[2,:])
    NLIN_intra = P0**c.three*NLIN_intra
    return NLIN_intra

def calcInterChannelGN(chi,P0,nPol,dtype=tf.float64):
    c = tfConstants(dtype=dtype)

    GN_inter = chi[0,:]
    if nPol == 2:
        GN_inter = c.sixteen/c.eightyone*(GN_inter+c.two*chi[0,:]/c.four)
    GN_inter = P0**c.three*GN_inter
    return GN_inter

def calcIntraChannelGN(X,P0,nPol,dtype=tf.float64):
    c = tfConstants(dtype=dtype)

    GN_intra = c.two*X[1,:]
    if nPol == 2:
        GN_intra = c.sixteen/c.eightyone*(GN_intra+X[1,:])
    GN_intra = P0**c.three*GN_intra
    return GN_intra