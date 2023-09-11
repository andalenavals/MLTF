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

import logging
import tensorflow as tf

logger = logging.getLogger(__name__)
PRECISION=tf.float32 

def mse(targets, preds, mask=None, weights=None):
    r"""
    :Parameters:
        :targets: 3D array containing the true values to be predicted, :math:`p^{\mathrm{true}}`.
        :preds: 3D array with predictions from the neural network, :math:`\hat{p}`.
        :mask: 3D array acting as a mask for predictions before loss calculation. A value of 1 indicates to keep the prediction, while 0 indicates to ignore it. (Note: The mask's definition is opposite to that of masked arrays.)
        :weights: 3D array giving more or less importance to some cases and realizations, :math:`w`.

    :Returns:
        :math:`\frac{1}{\sum_{k=1}^{n_{\mathrm{case}}}\sum_{j=1}^{n_{\mathrm{rea}}} w_{j,k}}\sum_{k=1}^{n_{\mathrm{case}}}\sum_{j=1}^{n_{\mathrm{rea}}}w_{j,k}  \left( \hat{p}_{jk} - p^{\mathrm{true}}_k  \right)^2`
    """

    #assert preds[0].get_shape() ==mask[0].get_shape()     
    #assert tf.shape(preds[0])==tf.shape(mask[0])
    if tf.keras.backend.ndim(preds) == 3:
        if mask is not None:
            npoints=tf.cast(tf.shape(preds)[0]*tf.shape(preds)[1], PRECISION)
            mask_factor=npoints/tf.keras.backend.sum(mask)
            squarebias=mask_factor*tf.keras.backend.square(mask*(preds-targets))
        else:
            squarebias=tf.keras.backend.square(preds-targets)

        if weights is not None:
            logger.debug("Using case weights")
            num = tf.keras.backend.mean(weights*squarebias)
            den = tf.keras.backend.mean(weights )
            mse_val=num/den
        else:
            logger.debug("Not using case weights")
            mse_val=tf.keras.backend.mean(squarebias)

    return mse_val

def msb(targets, preds, mask=None, caseweights=None):
    r"""
    :Parameters:
        :targets: 3D array containing the true values to be predicted, :math:`p^{\mathrm{true}}`.
        :preds: 3D array with predictions from the neural network, :math:`\hat{p}`.
        :mask: 3D array acting as a mask for predictions before loss calculation. A value of 1 indicates to keep the prediction, while 0 indicates to ignore it. (Note: The mask's definition is opposite to that of masked arrays.)
        :caseweights: 3D array giving more or less importance to some cases, :math:`w`. 

    :Returns:
        :math:`\frac{1}{\sum_{k=1}^{n_{\mathrm{case}}} w_{k}}\sum_{k=1}^{n_{\mathrm{case}}} w_{k} \left[ \frac{1}{n_{\mathrm{rea}}}\sum_{j=1}^{n_{\mathrm{rea}}}  \hat{p}_{jk} - p^{\mathrm{true}}_k  \right]^2`
    """
    #assert preds[0].get_shape() ==mask[0].get_shape()
    if tf.keras.backend.ndim(preds) == 3:
        if mask is not None:        
            nrea= tf.cast(tf.shape(preds)[1],PRECISION)
            masked_preds=preds*mask
            mask_factor=nrea/tf.keras.backend.sum(mask,axis=1, keepdims=True)
            means=mask_factor*tf.keras.backend.mean(masked_preds, axis=1, keepdims=True)
        else:
            means=tf.keras.backend.mean(preds, axis=1, keepdims=True)
            
        biases = means - targets

        if caseweights is not None:
            logger.debug("Using case weights")
            num = tf.keras.backend.mean(caseweights*tf.keras.backend.square(biases))
            den = tf.keras.backend.mean(caseweights )
            msb_val=num/den
        else:
            logger.debug("Not using case weights")
            msb_val=tf.keras.backend.mean(tf.keras.backend.square(biases))
    return msb_val

def mswb(targets, wpreds, point_preds, mask=None):
    r"""
    :Parameters:
        :targets: 3D array containing the true values to be predicted, :math:`p^{\mathrm{true}}`.
        :wpreds: 3D array of weight predictions :math:`\hat{w}`.
        :point_preds: 3d array with estimates to calibrate with weights, :math:`p`.
        :mask: 3D array acting as a mask for predictions before loss calculation. A value of 1 indicates to keep the prediction, while 0 indicates to ignore it. (Note: The mask's definition is opposite to that of masked arrays.)

    :Returns:
        :math:`\frac{1}{n_{\mathrm{case}}} \sum_{k=1}^{n_{\mathrm{case}}}\left[ \frac{ \sum_{j=1}^{n_{\mathrm{rea}}}  p_{jk} \cdot \hat{w}_{jk}(p) }{\sum_{j=1}^{n_{\mathrm{rea}}} \hat{w}_{jk}(p)} - p^{\mathrm{true}}_k \right]^2`
    """

    if tf.keras.backend.ndim(wpreds) == 3:
        if mask is not None:
            masked_wpreds=wpreds*mask
            #mask factor cancel out for this case
            num = tf.keras.backend.mean(masked_wpreds*point_preds, axis=1, keepdims=True) 
            den = tf.keras.backend.mean(masked_wpreds , axis=1, keepdims=True) 
        else:
            num = tf.keras.backend.mean(wpreds*point_preds, axis=1, keepdims=True) 
            den = tf.keras.backend.mean(wpreds , axis=1, keepdims=True) 
        biases = num/den - targets
        mswb_val=tf.keras.backend.mean(tf.keras.backend.square(biases))
    return mswb_val

def mswb_lagrange1(targets, wpreds, point_preds, mask=None, lamb=1.0):
    r"""
    :Parameters:
        :targets: 3D array containing the true values to be predicted, :math:`p^{\mathrm{true}}`.
        :wpreds: 3D array of weight predictions :math:`\hat{w}`.
        :point_preds: 3d array with estimates to calibrate with weights, :math:`p`.
        :mask: 3D array acting as a mask for predictions before loss calculation. A value of 1 indicates to keep the prediction, while 0 indicates to ignore it. (Note: The mask's definition is opposite to that of masked arrays.)
        :lamb: float, lagrange multiplier.

    :Returns:
        :math:`\frac{1}{n_{\mathrm{case}}} \sum_{k=1}^{n_{\mathrm{case}}} \left[\frac{ \sum_{j=1}^{n_{\mathrm{rea}}}  p_{jk} \cdot \hat{w}_{jk}(p) }{\sum_{j=1}^{n_{\mathrm{rea}}} \hat{w}_{jk}(p)} - p^{\mathrm{true}}_k \right]^2 + \lambda \cdot \left( \frac{\sum_{k=1}^{n_{\mathrm{case}}}\sum_{j=1}^{n_{\mathrm{rea}}} \hat{w}_{jk}(p)}{n_{\mathrm{case}} \cdot n_{\mathrm{rea}}} - 0.5 \right)^2`
    """

    if tf.keras.backend.ndim(wpreds) == 3:
        if mask is not None:
            masked_wpreds=wpreds*mask
            #mask factor cancel out for this case
            num = tf.keras.backend.mean(masked_wpreds*point_preds, axis=1, keepdims=True) 
            den = tf.keras.backend.mean(masked_wpreds , axis=1, keepdims=True) 
        else:
            num = tf.keras.backend.mean(wpreds*point_preds, axis=1, keepdims=True) 
            den = tf.keras.backend.mean(wpreds , axis=1, keepdims=True) 
        biases = num/den - targets
        mswb_val=tf.keras.backend.mean(tf.keras.backend.square(biases))

        if mask is not None:
            nrea= tf.cast(tf.shape(wpreds)[1],PRECISION)
            mask_factor=nrea/tf.keras.backend.sum(mask,axis=1, keepdims=True)
            mean_preds_rea=mask_factor*tf.keras.backend.mean(masked_wpreds, axis=1, keepdims=True)
            mean_preds= tf.keras.backend.mean(mean_preds_rea)
        else:
            mean_preds = tf.keras.backend.mean(wpreds)
        lagrange_term=lamb*tf.keras.backend.square(mean_preds -0.5)
    return mswb_val+lagrange_term

def mswb_lagrange2(targets, wpreds, point_preds, mask=None, lamb=1.0):
    r"""
    :Parameters:
        :targets: 3D array containing the true values to be predicted, :math:`p^{\mathrm{true}}`.
        :wpreds: 3D array of weight predictions :math:`\hat{w}`.
        :point_preds: 3d array with estimates to calibrate with weights, :math:`p`.
        :mask: 3D array acting as a mask for predictions before loss calculation. A value of 1 indicates to keep the prediction, while 0 indicates to ignore it. (Note: The mask's definition is opposite to that of masked arrays.)
        :lamb: float, lagrange multiplier.

    :Returns:
        :math:`\frac{1}{n_{\mathrm{case}}} \sum_{k=1}^{n_{\mathrm{case}}} \left[\left(\frac{ \sum_{j=1}^{n_{\mathrm{rea}}}  p_{jk} \cdot \hat{w}_{jk}(p) }{\sum_{j=1}^{n_{\mathrm{rea}}} \hat{w}_{jk}(p)} - p^{\mathrm{true}}_k \right)^2 + \lambda \cdot \left( \frac{\sum_{j=1}^{n_{\mathrm{rea}}} w_{jk}(p)}{n_{\mathrm{rea}}} - 0.5\right)^2\right]`
    """

    mweight=0.5
    if tf.keras.backend.ndim(wpreds) == 3:
        if mask is not None:
            masked_wpreds=wpreds*mask
            #mask factor cancel out for this case
            num = tf.keras.backend.mean(masked_wpreds*point_preds, axis=1, keepdims=True) 
            den = tf.keras.backend.mean(masked_wpreds , axis=1, keepdims=True)

            nrea= tf.cast(tf.shape(wpreds)[1],PRECISION)
            mask_factor=nrea/tf.keras.backend.sum(mask,axis=1, keepdims=True)
            lagrange_term= lamb*tf.keras.backend.square(mask_factor*den-mweight)
        else:
            num = tf.keras.backend.mean(wpreds*point_preds, axis=1, keepdims=True) 
            den = tf.keras.backend.mean(wpreds , axis=1, keepdims=True)
            lagrange_term= lamb*tf.keras.backend.square(den-mweight)
        biases = num/den - targets
        mswb_val=tf.keras.backend.mean(tf.keras.backend.square(biases)+tf.keras.backend.square(lagrange_term))

    return mswb_val

def msmb(targets, mpreds, point_preds, mask=None):
    r"""
    :Parameters:
        :targets: 3D array containing the true values to be predicted, :math:`p^{\mathrm{true}}`.
        :mpreds: 3D array of corraction factor predictions :math:`\hat{m}`.
        :point_preds: 3d array with estimates to calibrate with weights, :math:`p`.
        :mask: 3D array acting as a mask for predictions before loss calculation. A value of 1 indicates to keep the prediction, while 0 indicates to ignore it. (Note: The mask's definition is opposite to that of masked arrays.)

    :Returns:
        :math:`\frac{1}{n_{\mathrm{case}}} \sum_{k=1}^{n_{\mathrm{case}}}\left[ \frac{1}{n_{\mathrm{rea}}} \sum_{j=1}^{n_{\mathrm{rea}}}  p_{jk} \cdot \left(1+\hat{m}_{jk}(p)\right) - p^{\mathrm{true}}_k \right]^2`
    """
    if tf.keras.backend.ndim(mpreds) == 3:
        if mask is not None:
            #nrea= tf.constant(preds.get_shape().as_list()[1],PRECISION)
            nrea= tf.cast(tf.shape(mpreds)[1],PRECISION)
            mask_factor=nrea/tf.keras.backend.sum(mask,axis=1, keepdims=True)
            masked_mpreds=(1+mpreds)*mask
            #masked_preds=(0.5+preds)*mask
            num = mask_factor*tf.keras.backend.mean(masked_mpreds*point_preds,axis=1,keepdims=True)
        else:
            num = tf.keras.backend.mean(( 1+mpreds)*point_preds, axis=1, keepdims=True)

        biases = num - targets
        msmb_val=tf.keras.backend.mean(tf.keras.backend.square(biases))
    return msmb_val

def mswcb(targets,  wpreds, mpreds, point_preds, mask=None):
    r"""
    :Parameters:
        :targets: 3D array containing the true values to be predicted, :math:`p^{\mathrm{true}}`.
        :wpreds: 3D array of corraction factor predictions :math:`\hat{w}`.
        :mpreds: 3D array of corraction factor predictions :math:`\hat{m}`.
        :point_preds: 3d array with estimates to calibrate with weights, :math:`p`.
        :mask: 3D array acting as a mask for predictions before loss calculation. A value of 1 indicates to keep the prediction, while 0 indicates to ignore it. (Note: The mask's definition is opposite to that of masked arrays.)

    :Returns:
        :math:`\frac{1}{n_{\mathrm{case}}} \sum_{k=1}^{n_{\mathrm{case}}} \left[ \frac{ \sum_{j=1}^{n_{\mathrm{rea}}}  p_{jk} \cdot (\hat{m}_{jk}(p) + 1) \cdot \hat{w}_{jk}(p) }{\sum_{j=1}^{n_{\mathrm{rea}}} \hat{w}_{jk}(p)} - p^{\mathrm{true}}_k \right]^2`
    """
    if tf.keras.backend.ndim(wpreds) == 3:
        if mask is not None:
            masked_w_preds=wpreds*mask
            masked_1pm_preds=(1+mpreds)*mask
            #mask factor cancel out for this case
            num = tf.keras.backend.mean((masked_1pm_preds)*masked_w_preds*point_preds, axis=1, keepdims=True)
            den = tf.keras.backend.mean(masked_w_preds , axis=1, keepdims=True) 
        else:
            num = tf.keras.backend.mean((1+mpreds)*wpreds*point_preds, axis=1, keepdims=True) 
            den = tf.keras.backend.mean(wpreds , axis=1, keepdims=True)
        
        biases = num/den - targets
        mswb_val=tf.keras.backend.mean(tf.keras.backend.square(biases))
    return mswb_val

# NEGATIVE LOG LIKELIHOOD
def nll(targets, pred_distribution , mask=None):
    r"""
    :Parameters:
        :targets: 3D array containing the true values to be predicted, :math:`p^{\mathrm{true}}`.
        :pred_distribution:  3D o 2D array of tfp.distributions.Distributions, :math:`P_{\theta}(X)`.
        :mask: 3D array acting as a mask for predictions before loss calculation. A value of 1 indicates to keep the prediction, while 0 indicates to ignore it. (Note: The mask's definition is opposite to that of masked arrays.)

    :Returns:
        :math:`-\frac{1}{n_{\mathrm{case}}n_{\mathrm{rea}}} \sum_{k=1}^{n_{case}} \sum_{j=1}^{n_{\mathrm{rea}}} \log{\left[P_{\theta}(X=p^{\mathrm{true}}_{j,k})\right]}`
    """
    if tf.keras.backend.ndim(pred_distribution) == 2:
        targets=tf.reshape(targets, tf.shape(targets)[:2])
        if mask is not None:
            mask=tf.keras.backend.sum(mask, axis=2, keepdims=False)
            nrea=tf.cast(tf.shape(mask)[1], PRECISION)
            mask_factor=nrea/tf.keras.backend.sum(mask, axis=1, keepdims=True)
            nll=-pred_distribution.log_prob(targets)
            NLL=mask_factor*mask*nll            
        else:
            NLL=-pred_distribution.log_prob(targets)
       

    if tf.keras.backend.ndim(pred_distribution) == 3:
        if mask is not None:
            #assert tf.shape(pred_distribution)[0]==tf.shape(mask)[0]
        
            nrea=tf.cast(tf.shape(mask)[1], PRECISION)
            mask_factor=nrea/tf.keras.backend.sum(mask,axis=1, keepdims=True)
            nll=tf.reshape(-pred_distribution.log_prob(targets),tf.shape(mask))
            NLL=mask_factor*mask*nll
            
            '''
            npoints=tf.cast(tf.shape(mask)[0]*tf.shape(mask)[1], PRECISION)
            mask_factor=npoints/tf.keras.backend.sum(mask)
            nll=tf.reshape(-pred_distribution.log_prob(targets),tf.shape(mask))
            NLL=mask_factor*mask*nll
            '''
            
        else:
            NLL=-pred_distribution.log_prob(targets)

    val=tf.keras.backend.mean(NLL)

    return val


def nll_normal(targets, pred_mu, pred_var, mask=None, weights=None):
    r"""
    :Parameters:
        :targets: 3D array containing the true values to be predicted, :math:`p^{\mathrm{true}}`.
        :pred_mu:  3D array with the predicted mean of the normal, :math:`\hat{\mu}`.
        :pred_var:  3D array with the predicted variance of the normal, :math:`\hat{\sigma^{2}}`.
        :mask: 3D array acting as a mask for predictions before loss calculation. A value of 1 indicates to keep the prediction, while 0 indicates to ignore it. (Note: The mask's definition is opposite to that of masked arrays.)

    :Returns:
        :math:`-\frac{1}{n_{\mathrm{case}}n_{\mathrm{rea}}} \sum_{k=1}^{n_{case}} \sum_{j=1}^{n_{\mathrm{rea}}} \log{\hat{\sigma}_{jk}}+0.5\left( \frac{ p^{\mathrm{true}}-\hat{\mu}_{jk}}{\hat{\sigma}_{jk}} \right)^{2}`
    """
    if tf.keras.backend.ndim(pred_mu) == 3:
        #this is just for stability
        pred_varu=tf.keras.backend.maximum(pred_var, 1.e-6)
        if mask is not None:
            npoints=tf.cast(tf.shape(pred_mu)[0]*tf.shape(pred_mu)[1], PRECISION)
            mask_factor=npoints/tf.keras.backend.sum(mask)
            squarebias=mask_factor*tf.keras.backend.square(mask*(pred_mu-targets))/pred_varu
            nll=mask_factor*mask*tf.keras.backend.log(pred_varu)+squarebias 
        else:
            squarebias=tf.keras.backend.square(pred_mu-targets)/pred_varu
            nll=tf.keras.backend.log(pred_varu)+squarebias

        if weights is not None:
            logger.debug("Using case weights")
            num = tf.keras.backend.mean(weights*nll)
            den = tf.keras.backend.mean(weights )
            rval=num/den
        else:
            logger.debug("Not using case weights")
            rval=tf.keras.backend.mean(nll)

        #rval=tf.keras.backend.maximum(rval, 1.e-6)
        #rval=tf.keras.backend.abs(rval)

    return rval
