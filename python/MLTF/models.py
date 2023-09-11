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
from . import layer
import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)
'''
Set of Sequential models for different loss functions. This module is particularly designed for regression problems. 
'''

def get_model(hidden_sizes=(5,5), activation='sigmoid', layer=layer.TfbilacLayer, dropout_prob=0.0, noutnodes=1):
    '''
    :Parameters:
       :hidden_sizes: tuple,list or array with the number of nodes for each layers
       :activation: activation function connecting inner layers
       :layer: an instance of tf.keras.layers.Layer
       :dropout_prob: float dropout probability
       :noutnodes: int, number of output nodes
    :Returns:
       instance of tf.keras.Sequential
    '''
    model = tf.keras.Sequential()

    for hidden_size in hidden_sizes:
        model.add(layer(hidden_size))
        if activation is not None:
            model.add(tf.keras.layers.Activation(activation))
        if dropout_prob>0.0:
            model.add(tf.keras.layers.Dropout(dropout_prob))
    # And the output layer, without activation:
    if noutnodes>0: model.add(layer(noutnodes))

    return model

    #print(model.summary(line_length=100))


def create_model(input_shape=None, hidden_sizes=(5,5), layer=layer.TfbilacLayer, activation='sigmoid', out_activation=None, loss_name='msb', use_mask=False, use_caseweights=False, lamb=None, dropout_prob=0.0,training=None, noutnodes=1, dtype='float32'):
    '''
    :Parameters:
       :input_shape: tuple, shape of the input. Should correspond to the shape of you input data ignoring the first dimension. (nreas,nfeats)
       :hidden_sizes: tuple,list or array with the number of nodes for each layers
       :layer: an instance of tf.keras.layers.Layer
       :activation: activation function connecting inner layers
       :out_activation: activation function for the output layer
       :loss_name: loss function
       :use_mask: bool, if True use include the mask Input. Avaliable for all loss functions
       :use_caseweights: bool, if True include the weights or caseweights in the Input. Avaliable for mse and msb loss functions.
       :lamb: float, lagrange multiplier. For mswb_lagrange1 and mswb_lagrange2, loss functions.
       :dropout_prob: float dropout probability
       :training: if True, activate the training behavious of the model (for example evalution during validation of the dropouts).
       :noutnodes: int, number of output nodes
       :dtype: dtype of the Inputs
    :Returns:
       instance of tf.keras.Model
    '''
    from . import loss_functions
    if dtype=='float32':
        loss_functions.PRECISION=tf.float32
    if dtype=='float16':
        loss_functions.PRECISION=tf.float16 
    #lamb: value for lagrange multiplier if loss function involves it.
    if input_shape is not None:
        nreas = input_shape[0]
        nfeas = input_shape[1]
    else:
        nreas=None
        nfeas=None
        input_shape=(nreas,nfeas)
    input_fea=tf.keras.Input(input_shape,dtype=dtype) #feat
    input_tar=tf.keras.Input((1, noutnodes),dtype=dtype) #targets
    inputs= [input_fea,input_tar]
    
    model = get_model(hidden_sizes=hidden_sizes, activation=activation,  layer=layer,dropout_prob=dropout_prob, noutnodes=noutnodes )

    shift=False; shiftval=1.0
    
    if out_activation is None:
        if (loss_name == 'mswb')| (loss_name == 'mswb_lagrange1')| (loss_name == 'mswb_lagrange2'):
            logger.info("Adding sigmoid activation to the last neuron since mswb was called")
            if shift:
                #sig=tf.keras.layers.Activation('sigmoid')
                #fin=tf.keras.layers.Lambda(lambda x:x+tf.constant(shiftval))#sig)
                model.add(tf.keras.layers.Activation('sigmoid'))
                model.add( tf.keras.layers.Lambda(lambda x:x+tf.constant(shiftval)) )
            else:
                model.add(tf.keras.layers.Activation('sigmoid')) #force positive weights for the mswb case
                    
    else:
        if (shift)&((loss_name == 'mswb')| (loss_name == 'mswb_lagrange1')| (loss_name == 'mswb_lagrange2')):
                model.add(tf.keras.layers.Activation(out_activation))
                model.add( tf.keras.layers.Lambda(lambda x:x+tf.constant(shiftval)) )
        else:
                model.add(tf.keras.layers.Activation(out_activation))

    x=model(inputs[0], training=training)

    if loss_name == 'mse':
        logger.info("Using %s"%(loss_name))
        if use_mask:
            input_mask = tf.keras.Input((nreas, noutnodes),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.mse( inputs[1], x, mask=inputs[2])
        else:
            loss_func=loss_functions.mse( inputs[1], x)
    
    if loss_name == 'msb':
        logger.info("Using %s"%(loss_name))
        if use_mask&(~use_caseweights):
            #input_mask = tf.keras.Input(input_shape,dtype=dtype)
            input_mask = tf.keras.Input((nreas, noutnodes),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.msb( inputs[1], x, mask=inputs[2])
        elif use_mask&use_caseweights:
            input_mask = tf.keras.Input((nreas,noutnodes),dtype=dtype); inputs.append(input_mask)
            input_caseweights = tf.keras.Input((1,1),dtype=dtype); inputs.append(input_caseweights)
            loss_func =loss_functions.msb( inputs[1], x, mask=inputs[2], caseweights=inputs[3])
        else:
            loss_func=loss_functions.msb( inputs[1], x)

    if loss_name == 'msmb':
        logger.info("Using %s"%(loss_name))

        input_pointpreds=tf.keras.Input((nreas, noutnodes),dtype=dtype)
        inputs.append(input_pointpreds)
        if use_mask:
            input_mask = tf.keras.Input((nreas, noutnodes),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.msmb( inputs[1], x, inputs[2], mask=inputs[3])
        else:
            loss_func =loss_functions.msmb( inputs[1], x, inputs[2])
            
    if loss_name == 'mswb':
        logger.info("Using %s"%(loss_name))

        input_pointpreds=tf.keras.Input((nreas, noutnodes),dtype=dtype)
        inputs.append(input_pointpreds)
        if use_mask & ~use_caseweights:
            #input_mask = tf.keras.Input(input_shape,dtype=dtype)
            input_mask = tf.keras.Input((nreas,noutnodes),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.mswb( inputs[1], x, inputs[2], mask=inputs[3])
        elif use_mask & use_caseweights:
            input_mask = tf.keras.Input((nreas,noutnodes),dtype=dtype); inputs.append(input_mask)
            input_caseweights = tf.keras.Input((1,1),dtype=dtype); inputs.append(input_caseweights)
            loss_func =loss_functions.msb( inputs[1], x, inputs[2], mask=inputs[3], caseweights=inputs[4])
        else:
            loss_func =loss_functions.mswb( inputs[1], x, inputs[2])

    if lamb is not None: logger.info("Using lambda %.3f"%(lamb))
    else: lamb=1.0
    if loss_name == 'mswb_lagrange1':
        logger.info("Using %s"%(loss_name))
        input_pointpreds=tf.keras.Input((nreas, noutnodes),dtype=dtype)
        inputs.append(input_pointpreds)
        if use_mask:
            input_mask = tf.keras.Input((nreas,noutnodes),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.mswb_lagrange1( inputs[1], x, inputs[2], mask=inputs[3], lamb=lamb)
        else:
            loss_func =loss_functions.mswb_lagrange1( inputs[1], x, inputs[2], lamb=lamb)
    if loss_name == 'mswb_lagrange2':
        logger.info("Using %s"%(loss_name))
        input_pointpreds=tf.keras.Input((nreas, noutnodes),dtype=dtype)
        inputs.append(input_pointpreds)
        if use_mask:
            input_mask = tf.keras.Input((nreas, noutnodes),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.mswb_lagrange2( inputs[1], x, inputs[2], mask=inputs[3], lamb=lamb)
        else:
            loss_func =loss_functions.mswb_lagrange2( inputs[1], x, inputs[2], lamb=lamb)

    # Adding loss function is problematic when trying multiprocessing

    logger.debug("Trying to add model to loss function")
    model=tf.keras.Model(inputs=inputs,outputs=[x])
    #model.compile(loss=loss_functions.msb,optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
    model.add_loss(loss_func)
    
    logger.debug("Loss function successfully added to the the model")
    
    return model


#two concatenated networks
def create_mw_model(input_shape_w=None, hidden_sizes_w=(5,5), layer_w=layer.TfbilacLayer, activation_w='sigmoid', out_activation_w=None, input_shape_m=None, hidden_sizes_m=(5,5), layer_m=layer.TfbilacLayer, activation_m='sigmoid', out_activation_m='sigmoid', loss_name='mswcb', use_mask=True, lamb=None, training=None, dropout_prob=0.0, noutnodes=1, dtype='float32' ):
    '''
    A model compoused of two concatenated Sequential models, one predicting :math:`\hat{w}` and the other predicting :math:`\hat{m}` for a `mswcb`.
    
    :Parameters:
       :input_shape_w: tuple, shape of the input for the network predicting :math:`\hat{w}`. Should correspond to the shape of you input data ignoring the first dimension. (nreas,nfeats)
       :hidden_sizes_w: tuple,list or array with the number of nodes for each layers
       :layer_w: an instance of tf.keras.layers.Layer
       :activation_w: activation function connecting inner layers
       :out_activation_w: activation function for the output layer
       :input_shape_m: tuple, shape of the input for the network predicting :math:`\hat{m}`. Should correspond to the shape of you input data ignoring the first dimension. (nreas,nfeats)
       :hidden_sizes_m: tuple,list or array with the number of nodes for each layers
       :layer_m: an instance of tf.keras.layers.Layer
       :activation_m: activation function connecting inner layers
       :out_activation_m: activation function for the output layer
       :loss_name: loss function
       :use_mask: bool, if True use include the mask Input. Avaliable for all loss functions
       :lamb: float, lagrange multiplier. For mswb_lagrange1 and mswb_lagrange2, loss functions.
       :training: if True, activate the training behavious of the model (for example evalution during validation of the dropouts).
       :dropout_prob: float dropout probability
       :noutnodes: int, number of output nodes
       :dtype: dtype of the Inputs
    :Returns:
       instance of tf.keras.Model
    '''
    tf.keras.backend.clear_session()
    from . import loss_functions
    if dtype=='float32':
        loss_functions.PRECISION=tf.float32
    if dtype=='float16':
        loss_functions.PRECISION=tf.float16 

    nreas_w = input_shape_w[0]; nreas_m = input_shape_m[0]
    nfeas_w = input_shape_w[1]; nreas_m = input_shape_m[0]
    assert nreas_w ==nreas_m
    
    input_fea_w=tf.keras.Input(input_shape_w,dtype=dtype) #feats for training the weights part
    input_fea_m=tf.keras.Input(input_shape_m,dtype=dtype) #feats for training the multiplicative bias
    input_tar=tf.keras.Input((1, noutnodes),dtype=dtype) #targets trug1
    input_pointpreds=tf.keras.Input((nreas_w, noutnodes),dtype=dtype) # point predicion by point NN

    inputs=[input_fea_w, input_fea_m,input_tar, input_pointpreds] 

    model_w=get_model(hidden_sizes=hidden_sizes_w, activation=activation_w, layer=layer_w, dropout_prob=dropout_prob, noutnodes=noutnodes )

    shift=False; shiftval=1.0
    if out_activation_w is None:
            if shift:
                logger.info("Setting sigmoid even though it was not declared, weights must be between 0 and 1")
                model_w.add(tf.keras.layers.Activation('sigmoid'))
                model_w.add( tf.keras.layers.Lambda(lambda x:x+tf.constant(shiftval)) )
            else:
                model_w.add(tf.keras.layers.Activation('sigmoid')) #force positive weights for the mswb case
    else:
        if shift:
            model_w.add(tf.keras.layers.Activation(out_activation_w))
            model_w.add( tf.keras.layers.Lambda(lambda x:x+tf.constant(shiftval)) )
        else:
            model_w.add(tf.keras.layers.Activation(out_activation_w))

    model_m=get_model(hidden_sizes=hidden_sizes_m, activation=activation_m, layer=layer_m, dropout_prob=dropout_prob, noutnodes=noutnodes)
    if out_activation_m is not None:
        model_m.add(tf.keras.layers.Activation(out_activation_m))
    
    x=model_w(inputs[0], training=training)
    y=model_m(inputs[1], training=training)

    if loss_name == 'mswcb':
        logger.info("Using %s"%(loss_name))
        if use_mask:
            input_mask = tf.keras.Input((nreas_w,noutnodes),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.mswcb( inputs[2], x, y, inputs[3], mask=inputs[4])
        else:
            loss_func =loss_functions.mswcb( inputs[2], x, y, inputs[3])

    if loss_name=='mswrb':
        logger.info("Using %s"%(loss_name))
        if use_mask:
            input_mask = tf.keras.Input((nreas_w,noutnodes),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.mswrb( inputs[2], x, y, inputs[3], mask=inputs[4])
        else:
            loss_func =loss_functions.mswrb( inputs[2], x, y, inputs[3])


    if lamb is not None: logger.info("Using lambda %.3f"%(lamb))
    else:
        lamb=1.0
    if loss_name == 'mswcb_lagrange1':
        logger.info("Using %s"%(loss_name))
        if use_mask:
            input_mask = tf.keras.Input((nreas_w,noutnodes),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.mswcb_lagrange1( inputs[2], x, y, inputs[3], mask=inputs[4], lamb=lamb)
        else:
            loss_func =loss_functions.mswcb_lagrange1( inputs[2], x, y, inputs[3], lamb=lamb)
    if loss_name == 'mswcb_lagrange2':
        logger.info("Using %s"%(loss_name))
        if use_mask:
            input_mask = tf.keras.Input((nreas_w,noutnodes),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.mswcb_lagrange2( inputs[2], x, y, inputs[3], mask=inputs[4], lamb=lamb)
        else:
            loss_func =loss_functions.mswcb_lagrange2( inputs[2], x, y, inputs[3], lamb=lamb)
    
    #concatenation would be needed if I wanted to build another loss function in terms of m and w
    # For instance, to use m as a feature of w, i.e, large m should have small weights
    #concat = tf.keras.layers.Concatenate()([x,y])
    
    full_model = tf.keras.Model(inputs=inputs, outputs=[x,y])
    full_model.add_loss(loss_func)
    return full_model


def create_musigma_model(input_shape=None, hidden_sizes=(5,5), layer=layer.TfbilacLayer, activation='sigmoid', out_activation=None, loss_name='nll_normal', use_mask=False, use_caseweights=False, dropout_prob=0.0,training=None,  dtype='float32' ):
    '''
    Model predicting 1D Independent Normal distributions for each case and realization.
    
    :Parameters:
        :input_shape: tuple, shape of the input. Should correspond to the shape of you input data ignoring the first dimension. (nreas,nfeats)
        :hidden_sizes: tuple,list or array with the number of nodes for each layers
        :layer: an instance of tf.keras.layers.Layer
        :activation: activation function connecting inner layers
        :out_activation: activation function for the output layer (this will affect only mu)
        :loss_name: loss function
        :loss_name: loss function
        :use_mask: bool, if True use include the mask Input. Avaliable for all loss functions
        :use_caseweights: bool, if True include the weights or caseweights in the Input. Avaliable for mse and msb loss functions.
        :dropout_prob: float dropout probability
        :training: if True, activate the training behavious of the model (for example evalution during validation of the dropouts).
        :dtype: dtype of the Inputs
    :Returns:
        instance of tf.keras.Model
    '''
    from . import loss_functions
    tf.keras.backend.clear_session()
    from . import loss_functions
    if dtype=='float32':
        loss_functions.PRECISION=tf.float32
    if dtype=='float16':
        loss_functions.PRECISION=tf.float16 
    
    if input_shape is not None:
        nreas = input_shape[0]
        nfeas = input_shape[1]
    else:
        nreas=None
        nfeas=None
        input_shape=(nreas,nfeas)
    input_fea=tf.keras.Input(input_shape,dtype=dtype) #feat
    input_tar=tf.keras.Input((1, 1),dtype=dtype) #targets
    inputs= [input_fea,input_tar]
    
    model = get_model(hidden_sizes=hidden_sizes, activation=activation, layer=layer, noutnodes=0)
    shift=False; shiftval=1.0

    #give the mean and the standard deviation
    distribution_params = tf.keras.layers.Dense(units=2)(model(inputs[0]))

    mean=tf.slice(distribution_params,[0,0,0],[-1,-1,1], name="mu")
    sigma=tf.slice(distribution_params,[0,0,1],[-1,-1,1], name="sigma")
    #std=tf.keras.layers.Activation("sigmoid")(sigma)
    std=tf.keras.layers.Activation("softplus")(sigma)
    #std=tf.keras.layers.Activation("relu")(sigma)

   
    if loss_name == 'nll_normal':
        logger.info("Using %s"%(loss_name))
        if use_mask:
            # 2 because one prediction is the mean and the other is std
            input_mask = tf.keras.Input((nreas,2),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.nll_normal( inputs[1], mean, std, mask=inputs[2])
        else:
            loss_func=loss_functions.nll_normal( inputs[1], mean, std)

    # Adding loss function is problematic when trying multiprocessing
    logger.debug("Trying to add model to loss function")
    model=tf.keras.Model(inputs=inputs,outputs=[mean,std])
    model.add_loss(loss_func)
    
    logger.debug("Loss function successfully added to the the model")
    #model.compile(loss=None, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics = [])
    return model



# PROBABILITY NETWORKS 
def create_probabilistic_model_independentnormal(input_shape=None, hidden_sizes=(5,5), layer=layer.TfbilacLayer, activation='sigmoid', out_activation=None, loss_name='nll', use_mask=False, use_caseweights=False, lamb=None, dtype='float32' ):
    '''
    Model predicting 1D Independent Normal distributions for each case and realization.
    
    :Parameters:
        :input_shape: tuple, shape of the input. Should correspond to the shape of you input data ignoring the first dimension. (nreas,nfeats)
        :hidden_sizes: tuple,list or array with the number of nodes for each layers
        :layer: an instance of tf.keras.layers.Layer
        :activation: activation function connecting inner layers
        :out_activation: activation function for the output layer
        :loss_name: loss function
        :use_mask: bool, if True use include the mask Input. Avaliable for all loss functions
        :use_caseweights: bool, if True include the weights or caseweights in the Input. Avaliable for mse and msb loss functions.
        :lamb: float, lagrange multiplier. For mswb_lagrange1 and mswb_lagrange2, loss functions.
        :dtype: dtype of the Inputs
    :Returns:
        instance of tf.keras.Model
    '''

    import tensorflow_probability as tfp
    tf.keras.backend.clear_session()
    from . import loss_functions
    if dtype=='float32':
        loss_functions.PRECISION=tf.float32
    if dtype=='float16':
        loss_functions.PRECISION=tf.float16 
    
    if input_shape is not None:
        nreas = input_shape[0]
        nfeas = input_shape[1]
    else:
        nreas=None
        nfeas=None
        input_shape=(nreas,nfeas)
    input_fea=tf.keras.Input(input_shape,dtype=dtype) #feat
    input_tar=tf.keras.Input((1, 1),dtype=dtype) #targets
    inputs= [input_fea,input_tar]
    
    model = get_model(hidden_sizes=hidden_sizes, activation=activation, layer=layer, noutnodes=0)
    shift=False; shiftval=1.0

    #give the mean and the standard deviation
    distribution_params = tf.keras.layers.Dense(units=2)(model(inputs[0]))

    mean=tf.slice(distribution_params,[0,0,0],[-1,-1,1], name="mu")
    sigma=tf.slice(distribution_params,[0,0,1],[-1,-1,1], name="sigma")
    #std=tf.keras.layers.Activation("sigmoid")(sigma)
    std=tf.keras.layers.Activation("softplus")(sigma)
    #std=tf.keras.layers.Activation("relu")(sigma)
    pars=tf.concat([mean,std],axis=2)
   
    
    x = tfp.layers.IndependentNormal(1)(pars)
    #x = tfp.layers.IndependentNormal(1)(distribution_params)
    
    if loss_name == 'nll':
        logger.info("Using %s"%(loss_name))
        if use_mask:
            input_mask = tf.keras.Input((nreas,1),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.nll( inputs[1], x, mask=inputs[2])
        else:
            loss_func=loss_functions.nll( inputs[1], x)

    # Adding loss function is problematic when trying multiprocessing
    logger.debug("Trying to add model to loss function")
    model=tf.keras.Model(inputs=inputs,outputs=[x])
    model.add_loss(loss_func)
    
    logger.debug("Loss function successfully added to the the model")
    #model.compile(loss=None, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics = [])
    return model


def create_probabilistic_model_mixturenormal(input_shape=None, hidden_sizes=(5,5), layer=layer.TfbilacLayer, activation='sigmoid', out_activation=None, loss_name='nll', ncomp=1, use_mask=False, use_caseweights=False, lamb=None, dtype='float32' ):
    '''
    Model predicting 1D MixtureNormal distributions for each case and realization.
    
    :Parameters:
        :input_shape: tuple, shape of the input. Should correspond to the shape of you input data ignoring the first dimension. (nreas,nfeats)
        :hidden_sizes: tuple,list or array with the number of nodes for each layers
        :layer: an instance of tf.keras.layers.Layer
        :activation: activation function connecting inner layers
        :out_activation: activation function for the output layer
        :ncomp: int, number of Normal distributions to combine
        :loss_name: loss function
        :use_mask: bool, if True use include the mask Input. Avaliable for all loss functions
        :use_caseweights: bool, if True include the weights or caseweights in the Input. Avaliable for mse and msb loss functions.
        :lamb: float, lagrange multiplier. For mswb_lagrange1 and mswb_lagrange2, loss functions.
        :dtype: dtype of the Inputs
    :Returns:
        instance of tf.keras.Model
    '''

    import tensorflow_probability as tfp
    tf.keras.backend.clear_session()
    from . import loss_functions
    if dtype=='float32':
        loss_functions.PRECISION=tf.float32
    if dtype=='float16':
        loss_functions.PRECISION=tf.float16
        
    if input_shape is not None:
        nreas = input_shape[0]
        nfeas = input_shape[1]
    else:
        nreas=None
        nfeas=None
        input_shape=(nreas,nfeas)
    input_fea=tf.keras.Input(input_shape,dtype=dtype) #feat
    input_tar=tf.keras.Input((1, 1),dtype=dtype) #targets
    inputs= [input_fea,input_tar]
    
    model = get_model(hidden_sizes=hidden_sizes, activation=activation, layer=layer, outnodes=0)
    shift=False; shiftval=1.0

    #give the mean and the standard deviation
    distribution_params = tf.keras.layers.Dense(units=ncomp*3)(model(inputs[0]))

    
    slices=[]
    for p in range(3):
        for c in [3*i for i in range(ncomp)]:
            sli=tf.slice(distribution_params,[0,0,c+p],[-1,-1,1], name="mu")
            if p==0: par=sli # mean
            if p==1: par=tf.keras.layers.Activation("softplus")(sli) #p
            if p==2: par=tf.keras.layers.Activation("softplus")(sli) #std
            slices.append(par)

    pars=tf.concat(slices,axis=2)
    

    #pars=distribution_params
    
    x = tfp.layers.MixtureNormal(ncomp)(pars)

    if loss_name == 'nll':
        logger.info("Using %s"%(loss_name))
        if use_mask:
            input_mask = tf.keras.Input((nreas,1),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.nll( inputs[1], x, mask=inputs[2])
        else:
            loss_func=loss_functions.nll( inputs[1], x)

    # Adding loss function is problematic when trying multiprocessing
    logger.debug("Trying to add model to loss function")
    model=(tf.keras.Model(inputs=inputs,outputs=[x]))
    model.add_loss(loss_func)
    
    logger.debug("Loss function successfully added to the the model")
    #model.compile(loss=None, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics = [])
    return model



# BAYESIAN NETWORK

# Make probabilist
'''
def random_gaussian_initializer(shape, dtype):
    n = int(shape / 2)
    loc_norm = tf.random_normal_initializer(mean=0., stddev=0.1)
    loc = tf.Variable(
        initial_value=loc_norm(shape=(n,), dtype=dtype)
    )
    scale_norm = tf.random_normal_initializer(mean=-3., stddev=0.1)
    scale = tf.Variable(
        initial_value=scale_norm(shape=(n,), dtype=dtype)
    )
    return tf.concat([loc, scale], 0)

def prior(kernel_size, bias_size, dtype=None):
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    """
    Scale mixture prior (See section 3.3 of the paper)
    """
    n = kernel_size + bias_size
    pi = 0.5
    return lambda t: tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[pi, 1. - pi]),
        components_distribution=tfd.Independent(
            tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=[[1e-0], [1e-6]]),
            reinterpreted_batch_ndims=1
        ))

def posterior(kernel_size, bias_size, dtype=None):
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype, initializer=lambda shape, dtype: random_gaussian_initializer(shape, dtype), trainable=True),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + 0.01 * tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1))
    ])
'''

'''
def prior(kernel_size, bias_size, dtype=None):
    import tensorflow_probability as tfp
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=tf.zeros(n), scale=1.0))
    ])
'''

def prior(kernel_size, bias_size, dtype=None):
    import tensorflow_probability as tfp
    n = kernel_size + bias_size
    prior_model = tf.keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n))
                #lambda t: tfp.distributions.Uniform( low=-1.0, high=1.0)
            )
        ]
    )
    return prior_model


def posterior(kernel_size, bias_size, dtype=None):
    import tensorflow_probability as tfp
    n = kernel_size + bias_size
    posterior_model = tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def get_bayesian_model(hidden_sizes=(5,5), activation='sigmoid', dropout_prob=0.0 , kl_weight=None, noutnodes=1):
    '''
    :Parameters:
        :hidden_sizes: tuple,list or array with the number of nodes for each layers
        :activation: activation function connecting inner layers
        :dropout_prob: float dropout probability
        :noutnodes: int, number of output nodes
    :Returns:
       instance of tf.keras.Sequential
    '''
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    model = tf.keras.Sequential()

    for hidden_size in hidden_sizes:
        #model.add(layer(hidden_size))
        lay=tfp.layers.DenseVariational(units=hidden_size, make_prior_fn= prior,make_posterior_fn=posterior, kl_weight=kl_weight)
        model.add(lay )
        if activation is not None:
            model.add(tf.keras.layers.Activation(activation))
        if dropout_prob>0.0:
            model.add(tf.keras.layers.Dropout(dropout_prob))

    # And the output layer, without activation:
    if noutnodes>0: 
        lay=tfp.layers.DenseVariational(units=noutnode, make_prior_fn= prior,make_posterior_fn=posterior )
        model.add(lay)
    
    return model

def create_bayesian_model(input_shape=None, hidden_sizes=(5,5),  activation='sigmoid', out_activation=None, loss_name='msb', use_mask=False,use_caseweights=False, lamb=None, kl_weight=None,  dropout_prob=0.0,training=None, noutnodes=1,  dtype='float32'):
    '''
    :Parameters:
        :input_shape: tuple, shape of the input. Should correspond to the shape of you input data ignoring the first dimension. (nreas,nfeats)
        :hidden_sizes: tuple,list or array with the number of nodes for each layers
        :activation: activation function connecting inner layers
        :out_activation: activation function for the output layer
        :loss_name: loss function
        :use_mask: bool, if True use include the mask Input. Avaliable for all loss functions
        :use_caseweights: bool, if True include the weights or caseweights in the Input. Avaliable for mse and msb loss functions.
        :lamb: float, lagrange multiplier. For mswb_lagrange1 and mswb_lagrange2, loss functions.
        :kl_weight: tensorflow_probability.layers.DenseVariational parameter
        :dropout_prob: float dropout probability
        :training: if True, activate the training behavious of the model (for example evalution during validation of the dropouts).
        :noutnodes: int, number of output nodes
        :dtype: dtype of the Inputs
    :Returns:
        instance of tf.keras.Model
    '''
    from . import loss_functions
    if dtype=='float32':
        loss_functions.PRECISION=tf.float32
    if dtype=='float16':
        loss_functions.PRECISION=tf.float16
        
    tf.keras.backend.clear_session()
    if input_shape is not None:
        nreas = input_shape[0]
        nfeas = input_shape[1]
    else:
        nreas=None
        nfeas=None
        input_shape=(nreas,nfeas)
    input_fea=tf.keras.Input(input_shape,dtype=dtype) #feat
    input_tar=tf.keras.Input((1, 1),dtype=dtype) #targets
    inputs= [input_fea,input_tar]
    
    model = get_bayesian_model(hidden_sizes=hidden_sizes, activation=activation, dropout_prob=dropout_prob,  kl_weight=kl_weight, noutnodes=noutnodes)
    shift=False; shiftval=1.0

    if out_activation is None:
        if (loss_name == 'mswb')| (loss_name == 'mswb_lagrange1')| (loss_name == 'mswb_lagrange2'):
            logger.info("Adding sigmoid activation to the last neuron since mswb was called")
            if shift:
                #sig=tf.keras.layers.Activation('sigmoid')
                #fin=tf.keras.layers.Lambda(lambda x:x+tf.constant(shiftval))#sig)
                model.add(tf.keras.layers.Activation('sigmoid'))
                model.add( tf.keras.layers.Lambda(lambda x:x+tf.constant(shiftval)) )
            else:
                model.add(tf.keras.layers.Activation('sigmoid')) #force positive weights for the mswb case
                    
    else:
        if (shift)&((loss_name == 'mswb')| (loss_name == 'mswb_lagrange1')| (loss_name == 'mswb_lagrange2')):
                model.add(tf.keras.layers.Activation(out_activation))
                model.add( tf.keras.layers.Lambda(lambda x:x+tf.constant(shiftval)) )
        else:
                model.add(tf.keras.layers.Activation(out_activation))

    x=model(inputs[0], training=training)

    
    #distribution_params = tf.keras.layers.Dense(units=2)(x)
    #x = tfp.layers.IndependentNormal(1)(distribution_params)

    if loss_name == 'mse':
        logger.info("Using %s"%(loss_name))
        if use_mask:
            input_mask = tf.keras.Input((nreas,noutnodes),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.mse( inputs[1], x, mask=inputs[2])
        else:
            loss_func=loss_functions.mse( inputs[1], x)
    

    if loss_name == 'msb':
        logger.info("Using %s"%(loss_name))
        if use_mask&(~use_caseweights):
            #input_mask = tf.keras.Input(input_shape,dtype=dtype)
            input_mask = tf.keras.Input((nreas,noutnodes),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.msb( inputs[1], x, mask=inputs[2])
        elif use_mask&use_caseweights:
            input_mask = tf.keras.Input((nreas,noutnodes),dtype=dtype); inputs.append(input_mask)
            input_caseweights = tf.keras.Input((1,noutnodes),dtype=dtype); inputs.append(input_caseweights)
            loss_func =loss_functions.msb( inputs[1], x, mask=inputs[2], caseweights=inputs[3])
        else:
            loss_func=loss_functions.msb( inputs[1], x)

    if loss_name == 'msmb':
        logger.info("Using %s"%(loss_name))

        input_pointpreds=tf.keras.Input((nreas, noutnodes),dtype=dtype)
        inputs.append(input_pointpreds)
        if use_mask:
            input_mask = tf.keras.Input((nreas,noutnodes),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.msmb( inputs[1], x, inputs[2], mask=inputs[3])
        else:
            loss_func =loss_functions.msmb( inputs[1], x, inputs[2])
            
    if loss_name == 'mswb':
        logger.info("Using %s"%(loss_name))

        input_pointpreds=tf.keras.Input((nreas, noutnodes),dtype=dtype)
        inputs.append(input_pointpreds)
        if use_mask & ~use_caseweights:
            #input_mask = tf.keras.Input(input_shape,dtype=dtype)
            input_mask = tf.keras.Input((nreas,noutnodes),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.mswb( inputs[1], x, inputs[2], mask=inputs[3])
        elif use_mask & use_caseweights:
            input_mask = tf.keras.Input((nreas,noutnodes),dtype=dtype); inputs.append(input_mask)
            input_caseweights = tf.keras.Input((1,noutnodes),dtype=dtype); inputs.append(input_caseweights)
            loss_func =loss_functions.msb( inputs[1], x, inputs[2], mask=inputs[3], caseweights=inputs[4])
        else:
            loss_func =loss_functions.mswb( inputs[1], x, inputs[2])

    if lamb is not None: logger.info("Using lambda %.3f"%(lamb))
    else: lamb=1.0
    if loss_name == 'mswb_lagrange1':
        logger.info("Using %s"%(loss_name))
        input_pointpreds=tf.keras.Input((nreas, noutnodes),dtype=dtype)
        inputs.append(input_pointpreds)
        if use_mask:
            input_mask = tf.keras.Input((nreas,noutnodes),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.mswb_lagrange1( inputs[1], x, inputs[2], mask=inputs[3], lamb=lamb)
        else:
            loss_func =loss_functions.mswb_lagrange1( inputs[1], x, inputs[2], lamb=lamb)
    if loss_name == 'mswb_lagrange2':
        logger.info("Using %s"%(loss_name))
        input_pointpreds=tf.keras.Input((nreas, noutnodes),dtype=dtype)
        inputs.append(input_pointpreds)
        if use_mask:
            input_mask = tf.keras.Input((nreas,noutnodes),dtype=dtype)
            inputs.append(input_mask)
            loss_func =loss_functions.mswb_lagrange2( inputs[1], x, inputs[2], mask=inputs[3], lamb=lamb)
        else:
            loss_func =loss_functions.mswb_lagrange2( inputs[1], x, inputs[2], lamb=lamb)


    # Adding loss function is problematic when trying multiprocessing
    logger.debug("Trying to add model to loss function")
    model=(tf.keras.Model(inputs=inputs,outputs=[x]))
    model.add_loss(loss_func)
    
    logger.debug("Loss function successfully added to the the model")
    #model.compile(loss=None, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics = [])
    return model
    
    return
    
