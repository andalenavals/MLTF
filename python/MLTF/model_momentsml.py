import logging

from . import layer as customlayer
from .layer import *
from . import loss_functions
from . import activations
from .activations import *


import numpy as np

logger = logging.getLogger(__name__)

def get_model(hidden_sizes=(5,5), activation='sigmoid', in_activation=None, layer=None, dropout_prob=0.0, l1=False, l2=False, lambda_value=0.01, batchnormalization=False, noutputs=1, out_activation=None, loss_name='msb'):
    import tensorflow as tf
    model = tf.keras.Sequential()

    if layer is None: layer=tf.keras.layers.Dense
    if in_activation is not None:
        logger.info("Adding input activations %s"%(in_activation) )
        if in_activation in activations.list:
            model.add(tf.keras.layers.Activation(eval(in_activation)))
        elif in_activation in customlayer.list:
            model.add(eval(in_activation)(1))
        else:
            model.add(tf.keras.layers.Activation(in_activation))
        
    for hidden_size in hidden_sizes:
        if l2:
            model.add(layer(hidden_size, kernel_regularizer=tf.keras.regularizers.l2(lambda_value)))
        elif l1:
            model.add(layer(hidden_size, kernel_regularizer=tf.keras.regularizers.l1(lambda_value)))
        else:
            model.add(layer(hidden_size))
        if activation in activations.list:
            model.add(tf.keras.layers.Activation(eval(activation)))
        elif activation in customlayer.list:
            model.add(eval(activation)(1))
        elif activation is not None:
            model.add(tf.keras.layers.Activation(activation))

        if batchnormalization:
            model.add(tf.keras.layers.BatchNormalization())
            
        #if dropout_prob>0.0:
        #    model.add(tf.keras.layers.Dropout(dropout_prob))
    # And the output layer, without activation:
    model.add(layer(noutputs))

    if out_activation is None:
        if loss_name in ['mswb','mswb_mudot', 'mswb_lagrange1', 'mswb_lagrange2', 'mswb_autodiff1', 'mswb_autodiff2']:
            logger.info("Adding sigmoid activation to the last neuron since mswb was called")
            model.add(tf.keras.layers.Activation('sigmoid'))
    else:
        model.add(tf.keras.layers.Activation(out_activation))

    #shift=False; shiftval=1.0
    #if shift: model.add( tf.keras.layers.Lambda(lambda x:x+tf.constant(shiftval)) )
       
    return model

    #print(model.summary(line_length=100))

def create_model(input_shape=None, hidden_sizes=(5,5), layer=None, activation='sigmoid', in_activation=None, out_activation=None, loss_name='msb', use_mask=False, use_caseweights=False, lamb=None, dropout_prob=0.0, l1=False, l2=False, lambda_value=0.01, batchnormalization=False, training=None, noutputs=1):
    import tensorflow as tf
    
    #lamb: value for lagrange multiplier if loss function involves it.
    tf.keras.backend.clear_session()
    
    if input_shape is not None:
        nreas = input_shape[0]
        nfeas = input_shape[1]
    else:
        nreas=None
        nfeas=None
        input_shape=(nreas,nfeas)
    input_fea=tf.keras.Input(input_shape,dtype='float32') #feat
    #input_tar=tf.keras.Input((None, noutputs),dtype='float32') #targets
    input_tar=tf.keras.Input((None, None), dtype='float32') #targets
    inputs= [input_fea,input_tar]

    model = get_model(hidden_sizes=hidden_sizes,
                      activation=activation, layer=layer, in_activation=in_activation,
                      dropout_prob=dropout_prob, l1=l1, l2=l2,
                      lambda_value=lambda_value, batchnormalization=batchnormalization,
                      noutputs=noutputs, out_activation=out_activation, loss_name=loss_name )
                
    x=model(inputs[0], training=training)

    loss_func=None
    if loss_name == 'mse':
        logger.info("Using %s"%(loss_name))
        if use_mask:
            input_mask = tf.keras.Input((nreas,noutputs),dtype='float32')
            inputs.append(input_mask)
            loss_func =loss_functions.mse( inputs[1], x, mask=inputs[2])
        else:
            loss_func=loss_functions.mse( inputs[1], x)
    
    if loss_name == 'msb':
        logger.info("Using %s"%(loss_name))
        if use_mask&(~use_caseweights):
            #input_mask = tf.keras.Input(input_shape,dtype='float32')
            input_mask = tf.keras.Input((nreas,noutputs),dtype='float32')
            inputs.append(input_mask)
            loss_func =loss_functions.msb( inputs[1], x, mask=inputs[2])
        elif use_mask&use_caseweights:
            input_mask = tf.keras.Input((nreas,noutputs),dtype='float32'); inputs.append(input_mask)
            input_caseweights = tf.keras.Input((1,1),dtype='float32'); inputs.append(input_caseweights)
            loss_func =loss_functions.msb( inputs[1], x, mask=inputs[2], caseweights=inputs[3])
        else:
            loss_func=loss_functions.msb( inputs[1], x)

    if loss_name == 'msmb':
        logger.info("Using %s"%(loss_name))

        #input_pointpreds=tf.keras.Input((nreas, noutputs),dtype='float32')
        input_pointpreds=tf.keras.Input((nreas, None),dtype='float32')
        inputs.append(input_pointpreds)
        if use_mask:
            #input_mask = tf.keras.Input((nreas,noutputs),dtype='float32')
            input_mask = tf.keras.Input((nreas, None),dtype='float32')
            inputs.append(input_mask)
            loss_func =loss_functions.msmb( inputs[1], x, inputs[2], mask=inputs[3])
        else:
            loss_func =loss_functions.msmb( inputs[1], x, inputs[2])
            
    if loss_name == 'mswb':
        logger.info("Using %s"%(loss_name))

        #input_pointpreds=tf.keras.Input((nreas,noutputs),dtype='float32')
        input_pointpreds=tf.keras.Input((nreas,None),dtype='float32')
        inputs.append(input_pointpreds)
        if use_mask & ~use_caseweights:
            #input_mask = tf.keras.Input(input_shape,dtype='float32')
            #input_mask = tf.keras.Input((nreas,noutputs),dtype='float32')
            input_mask = tf.keras.Input((nreas,None),dtype='float32')
            inputs.append(input_mask)
            loss_func =loss_functions.mswb( inputs[1], x, inputs[2], mask=inputs[3])
        elif use_mask & use_caseweights:
            #input_mask = tf.keras.Input((nreas,noutputs),dtype='float32'); inputs.append(input_mask)
            input_mask = tf.keras.Input((nreas,None),dtype='float32'); inputs.append(input_mask)
            input_caseweights = tf.keras.Input((1,1),dtype='float32'); inputs.append(input_caseweights)
            loss_func =loss_functions.mswb( inputs[1], x, inputs[2], mask=inputs[3], caseweights=inputs[4])
        else:
            loss_func =loss_functions.mswb( inputs[1], x, inputs[2])

    if lamb is not None: logger.info("Using lambda %.3f"%(lamb))
    if loss_name == 'mswb_lagrange1':
        logger.info("Using %s"%(loss_name))
        #input_pointpreds=tf.keras.Input((nreas, noutputs),dtype='float32')
        input_pointpreds=tf.keras.Input((nreas, None),dtype='float32')
        inputs.append(input_pointpreds)
        if use_mask:
            #input_mask = tf.keras.Input((nreas,noutputs),dtype='float32')
            input_mask = tf.keras.Input((nreas,None),dtype='float32')
            inputs.append(input_mask)
            loss_func =loss_functions.mswb_lagrange1( inputs[1], x, inputs[2], mask=inputs[3], lamb=lamb)
        else:
            loss_func =loss_functions.mswb_lagrange1( inputs[1], x, inputs[2], lamb=lamb)
    if loss_name == 'mswb_lagrange2':
        logger.info("Using %s"%(loss_name))
        #input_pointpreds=tf.keras.Input((nreas, noutputs),dtype='float32')
        input_pointpreds=tf.keras.Input((nreas, None),dtype='float32')
        inputs.append(input_pointpreds)
        if use_mask:
            #input_mask = tf.keras.Input((nreas,noutputs),dtype='float32')
            input_mask = tf.keras.Input((nreas,None),dtype='float32')
            inputs.append(input_mask)
            loss_func =loss_functions.mswb_lagrange2( inputs[1], x, inputs[2], mask=inputs[3], lamb=lamb)
        else:
            loss_func =loss_functions.mswb_lagrange2( inputs[1], x, inputs[2], lamb=lamb)
    if loss_name == 'mswb_autodiff1':
        logger.info("Using %s"%(loss_name))
        #input_pointpreds=tf.keras.Input((nreas, noutputs),dtype='float32')
        input_pointpreds=tf.keras.Input((nreas, None),dtype='float32')
        inputs.append(input_pointpreds)
        if use_mask:
            #input_mask = tf.keras.Input((nreas,noutputs),dtype='float32')
            input_mask = tf.keras.Input((nreas,None),dtype='float32')
            inputs.append(input_mask)
            loss_func =loss_functions.mswb_autodiff1( inputs[1], x, inputs[2], mask=inputs[3], lamb=lamb)
        else:
            loss_func =loss_functions.mswb_autodiff1( inputs[1], x, inputs[2], lamb=lamb)
    if loss_name == 'mswb_autodiff2':
        logger.info("Using %s"%(loss_name))
        #input_pointpreds=tf.keras.Input((nreas, noutputs),dtype='float32')
        input_pointpreds=tf.keras.Input((nreas, None),dtype='float32')
        inputs.append(input_pointpreds)
        if use_mask:
            #input_mask = tf.keras.Input((nreas,noutputs),dtype='float32')
            input_mask = tf.keras.Input((nreas,None),dtype='float32')
            inputs.append(input_mask)
            loss_func =loss_functions.mswb_autodiff2( inputs[1], x, inputs[2], mask=inputs[3], lamb=lamb)
        else:
            loss_func =loss_functions.mswb_autodiff2( inputs[1], x, inputs[2], lamb=lamb)


    # Adding loss function is problematic when trying multiprocessing
    model=(tf.keras.Model(inputs=inputs,outputs=[x]))
    logger.debug("Trying to add losss to the model")

    if loss_func is not None:
        model.add_loss(loss_func)
    #model.add_loss(lambda: loss_func)
    
    logger.debug("Loss function successfully added to the the model")
    logger.debug("Loss function successfully added to the the model")
    return model

#two concatenated networks

def create_2compind_model(input_shape=None, hidden_sizes=(5,5), layer=None, activation='sigmoid', in_activation=None, out_activation=None, loss_name='msb', use_mask=False, use_caseweights=False, lamb=None, dropout_prob=0.0, l1=False, l2=False, lambda_value=0.01, batchnormalization=False, training=None, noutputs=1):
    import tensorflow as tf
    tf.keras.backend.clear_session()

    if input_shape is not None:
        nreas = input_shape[0]
        nfeas = input_shape[1]
    else:
        nreas=None
        nfeas=None
        input_shape=(nreas,nfeas)
    input_fea=tf.keras.Input(input_shape,dtype='float32') #feat
    input_tar=tf.keras.Input((1, noutputs),dtype='float32') #targets
    inputs= [input_fea,input_tar]

    model1=get_model(hidden_sizes=hidden_sizes, activation=activation,
                     in_activation=in_activation , layer=layer,
                     dropout_prob=dropout_prob, l1=l1, l2=l2,
                     lambda_value=lambda_value, batchnormalization=batchnormalization,
                     noutputs=1)
    model2=get_model(hidden_sizes=hidden_sizes, activation=activation,
                     in_activation=in_activation, layer=layer,
                     dropout_prob=dropout_prob, l1=l1, l2=l2,
                     lambda_value=lambda_value,batchnormalization=batchnormalization,
                     noutputs=1)

    for mod in [model1, model2]:
        shift=False; shiftval=1.0
        if out_activation is None:
            if (loss_name == 'mswb')| (loss_name == 'mswb_lagrange1')| (loss_name == 'mswb_lagrange2'):
                if shift:
                    logger.info("Setting sigmoid even though it was not declared, weights must be between 0 and 1")
                    mod.add(tf.keras.layers.Activation('sigmoid'))
                    mod.add( tf.keras.layers.Lambda(lambda x:x+tf.constant(shiftval)) )
                else:
                    mod.add(tf.keras.layers.Activation('sigmoid')) #force positive weights for the mswb case
        else:
            if shift:
                mod.add(tf.keras.layers.Activation(out_activation))
                mod.add(tf.keras.layers.Lambda(lambda x:x+tf.constant(shiftval)) )
            else:
                mod.add(tf.keras.layers.Activation(out_activation))

    
    
    z=model1(inputs[0])
    y=model2(inputs[0])
    x=tf.keras.layers.Concatenate()([z,y])

    loss_func=None
    if loss_name == 'mse':
        logger.info("Using %s"%(loss_name))
        if use_mask:
            input_mask = tf.keras.Input((nreas,noutputs),dtype='float32')
            inputs.append(input_mask)
            loss_func =loss_functions.mse( inputs[1], x, mask=inputs[2])
        else:
            loss_func=loss_functions.mse( inputs[1], x)
    
    if loss_name == 'msb':
        logger.info("Using %s"%(loss_name))
        if use_mask&(~use_caseweights):
            #input_mask = tf.keras.Input(input_shape,dtype='float32')
            input_mask = tf.keras.Input((nreas,noutputs),dtype='float32')
            inputs.append(input_mask)
            loss_func =loss_functions.msb( inputs[1], x, mask=inputs[2])
        elif use_mask&use_caseweights:
            input_mask = tf.keras.Input((nreas,noutputs),dtype='float32'); inputs.append(input_mask)
            input_caseweights = tf.keras.Input((1,1),dtype='float32'); inputs.append(input_caseweights)
            loss_func =loss_functions.msb( inputs[1], x, mask=inputs[2], caseweights=inputs[3])
        else:
            loss_func=loss_functions.msb( inputs[1], x)

    if loss_name == 'msmb':
        logger.info("Using %s"%(loss_name))

        input_pointpreds=tf.keras.Input((nreas, noutputs),dtype='float32')
        inputs.append(input_pointpreds)
        if use_mask:
            input_mask = tf.keras.Input((nreas,noutputs),dtype='float32')
            inputs.append(input_mask)
            loss_func =loss_functions.msmb( inputs[1], x, inputs[2], mask=inputs[3])
        else:
            loss_func =loss_functions.msmb( inputs[1], x, inputs[2])
            
    if loss_name == 'mswb':
        logger.info("Using %s"%(loss_name))

        input_pointpreds=tf.keras.Input((nreas,noutputs),dtype='float32')
        inputs.append(input_pointpreds)
        if use_mask & ~use_caseweights:
            #input_mask = tf.keras.Input(input_shape,dtype='float32')
            input_mask = tf.keras.Input((nreas,noutputs),dtype='float32')
            inputs.append(input_mask)
            loss_func =loss_functions.mswb( inputs[1], x, inputs[2], mask=inputs[3])
        elif use_mask & use_caseweights:
            input_mask = tf.keras.Input((nreas,noutputs),dtype='float32'); inputs.append(input_mask)
            input_caseweights = tf.keras.Input((1,1),dtype='float32'); inputs.append(input_caseweights)
            loss_func =loss_functions.mswb( inputs[1], x, inputs[2], mask=inputs[3], caseweights=inputs[4])
        else:
            loss_func =loss_functions.mswb( inputs[1], x, inputs[2])
    

    # Adding loss function is problematic when trying multiprocessing
    model=(tf.keras.Model(inputs=inputs,outputs=[x]))
    logger.debug("Trying to add losss to the model")

    if loss_func is not None:
        model.add_loss(loss_func)
    
    logger.debug("Loss function successfully added to the the model")
    return model

#two concatenated networks

def create_mw_model(input_shape_w=None, hidden_sizes_w=(5,5), layer_w=None, activation_w='sigmoid', in_activation_w=None, out_activation_w=None, input_shape_m=None, hidden_sizes_m=(5,5), layer_m=None, activation_m='sigmoid', in_activation_m=None, out_activation_m='sigmoid', loss_name='mswcb', use_mask=True, lamb=None, dropout_prob=0.0, l1=False, l2=False, lambda_value=0.01, batchnormalization=False, noutputs=1 ):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    
    nreas_w = input_shape_w[0]; nreas_m = input_shape_m[0]
    nfeas_w = input_shape_w[1]; nreas_m = input_shape_m[0]
    assert nreas_w ==nreas_m
    
    input_fea_w=tf.keras.Input(input_shape_w,dtype='float32') #feats for training the weights part
    input_fea_m=tf.keras.Input(input_shape_m,dtype='float32') #feats for training the multiplicative bias
    input_tar=tf.keras.Input((1, noutputs),dtype='float32') #targets trug1
    input_pointpreds=tf.keras.Input((nreas_w, noutputs),dtype='float32') # point predicion by point NN

    inputs=[input_fea_w, input_fea_m,input_tar, input_pointpreds] 

    model_w=get_model(hidden_sizes=hidden_sizes_w,
                      activation=activation_w, in_activation=in_activation_w ,
                      layer=layer_w, dropout_prob=dropout_prob, l1=l1, l2=l2,
                      lambda_value=lambda_value, batchnormalization=batchnormalization,
                      noutputs=noutputs)

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

    model_m=get_model(hidden_sizes=hidden_sizes_m,
                      activation=activation_m, in_activation=in_activation_m,
                      layer=layer_m, dropout_prob=dropout_prob, l1=l1, l2=l2,
                      lambda_value=lambda_value, batchnormalization=batchnormalization,
                      noutputs=noutputs)
    if out_activation_m is not None:
        model_m.add(tf.keras.layers.Activation(out_activation_m))
    
    x=model_w(inputs[0])
    y=model_m(inputs[1])

    loss_func=None
    if loss_name == 'mswcb':
        logger.info("Using %s"%(loss_name))
        if use_mask:
            input_mask = tf.keras.Input((nreas_w,noutputs),dtype='float32')
            inputs.append(input_mask)
            loss_func =loss_functions.mswcb( inputs[2], x, y, inputs[3], mask=inputs[4])
        else:
            loss_func =loss_functions.mswcb( inputs[2], x, y, inputs[3])

    if loss_name=='mswrb':
        logger.info("Using %s"%(loss_name))
        if use_mask:
            input_mask = tf.keras.Input((nreas_w,noutputs),dtype='float32')
            inputs.append(input_mask)
            loss_func =loss_functions.mswrb( inputs[2], x, y, inputs[3], mask=inputs[4])
        else:
            loss_func =loss_functions.mswrb( inputs[2], x, y, inputs[3])


    if lamb is not None: logger.info("Using lambda %.3f"%(lamb))
    if loss_name == 'mswcb_lagrange1':
        logger.info("Using %s"%(loss_name))
        if use_mask:
            input_mask = tf.keras.Input((nreas_w, noutputs),dtype='float32')
            inputs.append(input_mask)
            loss_func =loss_functions.mswcb_lagrange1( inputs[2], x, y, inputs[3], mask=inputs[4], lamb=lamb)
        else:
            loss_func =loss_functions.mswcb_lagrange1( inputs[2], x, y, inputs[3], lamb=lamb)
    if loss_name == 'mswcb_lagrange2':
        logger.info("Using %s"%(loss_name))
        if use_mask:
            input_mask = tf.keras.Input((nreas_w, noutputs),dtype='float32')
            inputs.append(input_mask)
            loss_func =loss_functions.mswcb_lagrange2( inputs[2], x, y, inputs[3], mask=inputs[4], lamb=lamb)
        else:
            loss_func =loss_functions.mswcb_lagrange2( inputs[2], x, y, inputs[3], lamb=lamb)
    
    #concatenation would be needed if I wanted to build another loss function in terms of m and w
    # For instance, to use m as a feature of w, i.e, large m should have small weights
    #concat = tf.keras.layers.Concatenate()([x,y])
    
    full_model = (tf.keras.Model(inputs=inputs, outputs=[x,y]))

    if loss_func is not None:
        full_model.add_loss(loss_func)
    return full_model
