# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#

"""
File: ML_Tensorflow/python/loss_functions.py

Created on: 13/09/22
Author: Andres Navarro
"""
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

def mse(targets, preds, mask=None):
    assert preds[0].get_shape() ==mask[0].get_shape()
    
    if tf.keras.backend.ndim(preds) == 3:
        if mask is not None:
            '''
            nrea= tf.constant(preds.get_shape().as_list()[1],tf.float32)
            mask_factor=nrea/tf.keras.backend.sum(mask,axis=1, keepdims=True)
            squarebias=mask_factor*tf.keras.backend.square(mask*(preds-targets))
            '''
            npoints=tf.cast(tf.shape(preds)[0]*tf.shape(preds)[1], tf.float32)
            mask_factor=npoints/tf.keras.backend.sum(mask)
            squarebias=mask_factor*tf.keras.backend.square(mask*(preds-targets))
        else:
            squarebias=tf.keras.backend.square(preds-targets)
        mse_val=tf.keras.backend.mean(squarebias)
            

    return mse_val

def msb(targets, preds, mask=None, caseweights=None):
    assert preds[0].get_shape() ==mask[0].get_shape()
    
    if tf.keras.backend.ndim(preds) == 3:
        if mask is not None:        
            #nrea= tf.constant(preds.get_shape().as_list()[1],tf.float32)
            nrea= tf.cast(tf.shape(preds)[1],tf.float32)
            masked_preds=preds*mask
            mask_factor=nrea/tf.keras.backend.sum(mask,axis=1, keepdims=True)
            means=mask_factor*tf.keras.backend.mean(masked_preds, axis=1, keepdims=True)
        else:
            means=tf.keras.backend.mean(preds, axis=1, keepdims=True)
            
        biases = means - targets

        if caseweights is not None:
            logger.info("Using case weights")
            num = tf.keras.backend.mean(caseweights*tf.keras.backend.square(biases))
            den = tf.keras.backend.mean(caseweights )
            msb_val=num/den
        else:
            logger.info("Not using case weights")
            msb_val=tf.keras.backend.mean(tf.keras.backend.square(biases))
    return msb_val

def mswb(targets, preds, point_preds, mask=None):
    if tf.keras.backend.ndim(preds) == 3:
        if mask is not None:
            masked_preds=preds*mask
            #mask factor cancel out for this case
            num = tf.keras.backend.mean(masked_preds*point_preds, axis=1, keepdims=True) 
            den = tf.keras.backend.mean(masked_preds , axis=1, keepdims=True) 
        else:
            num = tf.keras.backend.mean(preds*point_preds, axis=1, keepdims=True) 
            den = tf.keras.backend.mean(preds , axis=1, keepdims=True) 
        biases = num/den - targets
        mswb_val=tf.keras.backend.mean(tf.keras.backend.square(biases))
    return mswb_val

def mswb_lagrange1(targets, preds, point_preds, mask=None, lamb=1.0):
    ## constraining mean weights of all galaxies to avoid scale degeneracies.
    if tf.keras.backend.ndim(preds) == 3:
        if mask is not None:
            masked_preds=preds*mask
            #mask factor cancel out for this case
            num = tf.keras.backend.mean(masked_preds*point_preds, axis=1, keepdims=True) 
            den = tf.keras.backend.mean(masked_preds , axis=1, keepdims=True) 
        else:
            num = tf.keras.backend.mean(preds*point_preds, axis=1, keepdims=True) 
            den = tf.keras.backend.mean(preds , axis=1, keepdims=True) 
        biases = num/den - targets
        mswb_val=tf.keras.backend.mean(tf.keras.backend.square(biases))

        if mask is not None:
            #nrea= tf.constant(preds.get_shape().as_list()[1],tf.float32)
            nrea= tf.cast(tf.shape(preds)[1],tf.float32)
            mask_factor=nrea/tf.keras.backend.sum(mask,axis=1, keepdims=True)
            mean_preds_rea=mask_factor*tf.keras.backend.mean(masked_preds, axis=1, keepdims=True)
            mean_preds= tf.keras.backend.mean(mean_preds_rea)
        else:
            mean_preds = tf.keras.backend.mean(preds)
        lagrange_term=lamb*tf.keras.backend.square(mean_preds -0.5)
    return mswb_val+lagrange_term

def mswb_lagrange2(targets, preds, point_preds, mask=None, lamb=1.0):
    ## constraining mean weights of all galaxies to avoid scale degeneracies.
    mweight=1.5
    if tf.keras.backend.ndim(preds) == 3:
        if mask is not None:
            masked_preds=preds*mask
            #mask factor cancel out for this case
            num = tf.keras.backend.mean(masked_preds*point_preds, axis=1, keepdims=True) 
            den = tf.keras.backend.mean(masked_preds , axis=1, keepdims=True)

            #nrea= tf.constant(preds.get_shape().as_list()[1],tf.float32)
            nrea= tf.cast(tf.shape(preds)[1],tf.float32)
            mask_factor=nrea/tf.keras.backend.sum(mask,axis=1, keepdims=True)
            lagrange_term= lamb*tf.keras.backend.square(mask_factor*den-mweight)
        else:
            num = tf.keras.backend.mean(preds*point_preds, axis=1, keepdims=True) 
            den = tf.keras.backend.mean(preds , axis=1, keepdims=True)
            lagrange_term= lamb*tf.keras.backend.square(den-mweight)
        biases = num/den - targets
        mswb_val=tf.keras.backend.mean(tf.keras.backend.square(biases)+tf.keras.backend.square(lagrange_term))

    return mswb_val

# NEGATIVE LOG LIKELIHOOD
def nll(targets, pred_distribution , mask=None):
    import tensorflow as tf

    if tf.keras.backend.ndim(pred_distribution) == 2:
        targets=tf.reshape(targets, tf.shape(targets)[:2])
        if mask is not None:
            mask=tf.keras.backend.sum(mask, axis=2, keepdims=False)
            nrea=tf.cast(tf.shape(mask)[1], tf.float32)
            mask_factor=nrea/tf.keras.backend.sum(mask, axis=1, keepdims=True)
            nll=-pred_distribution.log_prob(targets)
            NLL=mask_factor*mask*nll            
        else:
            NLL=-pred_distribution.log_prob(targets)
       

    if tf.keras.backend.ndim(pred_distribution) == 3:
        if mask is not None:

            assert pred_distribution[0].get_shape() ==mask[0].get_shape()
        
            nrea=tf.cast(tf.shape(mask)[1], tf.float32)
            mask_factor=nrea/tf.keras.backend.sum(mask,axis=1, keepdims=True)
            nll=tf.reshape(-pred_distribution.log_prob(targets),tf.shape(mask))
            NLL=mask_factor*mask*nll
            
            '''
            npoints=tf.cast(tf.shape(mask)[0]*tf.shape(mask)[1], tf.float32)
            mask_factor=npoints/tf.keras.backend.sum(mask)
            nll=tf.reshape(-pred_distribution.log_prob(targets),tf.shape(mask))
            NLL=mask_factor*mask*nll
            '''
            
        else:
            NLL=-pred_distribution.log_prob(targets)

    val=tf.keras.backend.mean(NLL)

    return val
