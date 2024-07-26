# function to load model base on args and weights.
import logging
logger = logging.getLogger(__name__)
import tensorflow as tf
if float(tf.__version__[:3]) < 2.0:
    tf.enable_eager_execution()
from configparser import ConfigParser
import numpy as np
    
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_model(h5file=None, modelkwargs=None, concatenatenn=False):
    # ncomp refers to the the number of outpur of the function getmodel
    # for training wm single connect use ncomp=1, fully connected use ncomp=2
    #import tensorflow as tf
 
    from . import model_momentsml as model

    logger.info("Model has %i outputs"%(modelkwargs["noutputs"]))
    
    if h5file is not None:
        pass
        #loaded_model=tf.keras.models.load(h5file)

    if (modelkwargs is not None):
        if modelkwargs["loss_name"] in [ 'mswcb', 'mswcb_mudot', 'mswcb_lagrange1', 'mswcb_lagrange2',  'mswrb', 'mswrb_lagrange1', 'mswrb_lagrange2']:
            logger.info("Loading a mw model")
            loaded_model=model.create_mw_model(**modelkwargs)
        else:
            if not concatenatenn:
                loaded_model=model.create_model(**modelkwargs)
            else:
                loaded_model=model.create_2compind_model(**modelkwargs)
    return loaded_model


def eval_model(model, model_weights,inputs, ncomp=1, modelkwargs=None):
    import numpy as np
    if (model_weights is not None) :
            status=model.load_weights(model_weights)
            status.expect_partial()
    
    if (isinstance(inputs, np.ma.MaskedArray))|(isinstance(inputs, np.ndarray)) :
        #if isinstance(inputs, np.ma.MaskedArray):
        #    inputs=inputs.data
        if float(tf.__version__[:3]) <2.0:
            preds=model(inputs)
        else:
            # re do this this would work only for trainnig w for training g only one extra tensor should be provided
            shapes=np.array(model.input_shape)
            shapes[:,0]=inputs.shape[0] # setting ncases
            shapes[shapes==None]=1 ##shear targets are always 1 realization, but model is set to None because when predicting
            inps=[inputs]+[tf.constant(0, shape=s) for s in shapes[1:]]
            preds=model(inps)
            #preds=model([inputs.data, tf.constant(0, shape=(inputs.shape[0], 1 , ncomp)), tf.constant(0, shape=inputs.shape)])
    elif isinstance(inputs, list):
        if float(tf.__version__[:3]) <2.0:
            preds=model(inputs)
        else:
            
            #shapes= np.array([list(l["config"]["batch_input_shape"]) for l in model.get_config()["layers"] if l["class_name"]=='InputLayer'])
            shapes=np.array(model.input_shape)
            shapes[:,0]=inputs[0].shape[0] # setting ncases
            shapes[shapes==None]=1 ##shear targets are always 1 realization, but model is set to None because when predicting
            inps=[inputs]+[tf.constant(0, shape=s) for s in shapes[2:]]# this is not general only for making prediction 2 input are needed (features m, features w)
            preds=model(inps)
            #preds=model(inputs+[tf.constant(0), tf.constant(0), tf.constant(0)])
    else:
        logger.info("Inputs is type %s"%(type(inputs)))
        raise
        
    if ncomp==1:
        if isinstance(inputs, np.ma.MaskedArray):
            feats_mask=np.all(~inputs.mask,axis=2,keepdims=True)
            preds=np.ma.masked_array(preds.numpy(),mask=~feats_mask)
        elif isinstance(inputs, list):
            if np.all( [isinstance(i, np.ma.MaskedArray) for i in inputs]):
                mask=False
                for inp in inputs:
                    mask|=np.any(inp.mask,axis=2,keepdims=True)
                preds=[np.ma.masked_array(p.numpy(),mask=mask) for p in preds]
    if ncomp==2:
        if isinstance(inputs, np.ma.MaskedArray):
            if (modelkwargs["loss_name"]=="msb_wg"):
                pred1, pred2=preds
                feats_mask=np.all(~inputs.mask,axis=2,keepdims=True)
                preds1=np.ma.masked_array(pred1,mask=~feats_mask)
                preds2=np.ma.masked_array(pred2,mask=~feats_mask)
                preds=[preds1,preds2]
            else:
                mask =np.all(~inputs.mask,axis=2,keepdims=True) 
                mask=np.append(mask,mask,axis=2)
                preds=np.ma.masked_array(preds.numpy(),mask=~mask)
        elif isinstance(inputs, list):
            if np.all( [isinstance(i, np.ma.MaskedArray) for i in inputs]):
                mask=False
                for inp in inputs:
                    mask|=np.any(inp.mask,axis=2,keepdims=True)
                mask=np.append(mask,mask,axis=2)
                preds=[np.ma.masked_array(p.numpy(),mask=mask) for p in preds]
    return preds
    

def getpreds_model_mem(q, inputs, h5file=None, modelkwargs=None, model_weights=None, combine="mean", concatenatenn=False):
    import numpy as np

    loaded_model=load_model(h5file=h5file, modelkwargs=modelkwargs, concatenatenn=concatenatenn)
    assert loaded_model is not None
    logger.info("Model loaded")

    if (type(model_weights) is list)|(isinstance(model_weights,np.ndarray)):
        auxlist=[]
        logger.info("Combining prediction of %i committe members using the %s"%(len(model_weights), combine))
        for mw in model_weights:
            p=eval_model(loaded_model, mw,inputs, ncomp=modelkwargs["noutputs"], modelkwargs=modelkwargs)
            auxlist.append(p)
        if combine=="mean":
            preds=np.nanmean(auxlist, axis=0)
        elif combine=="median":
            preds=np.nanmedian(auxlist, axis=0)
        if isinstance(p, list):
            if isinstance(p[0], np.ma.MaskedArray):
                preds=[np.ma.masked_array(r,mask=p[0].mask) for r in preds]
        else:
            if isinstance(p, np.ma.MaskedArray):
                preds=np.ma.masked_array(preds,mask=p.mask)
            
    else:
        preds=eval_model(loaded_model, model_weights,inputs, ncomp=modelkwargs["noutputs"], modelkwargs=modelkwargs)

    
    if q is not None:
        print("Putting prediction in Queue")
        q.put(preds)
    else:
        return preds


def getpreds_model(inputs, h5file=None, modelkwargs=None, model_weights=None, useprocess=False, combine="mean", concatenatenn=False):
    import multiprocessing
    
    logger.info("Getting predicion of model")
    #print(modelkwargs)

    if useprocess:
        q=multiprocessing.Queue()
        p=multiprocessing.Process(target=getpreds_model_mem, args=(q, inputs,h5file,modelkwargs,model_weights,combine,concatenatenn))
        p.start()
        #logger.info("Getting prediction in Queue")
        preds=q.get()
        p.join()
        #logger.info("Predicitons obtained")
        return preds
    else:
        preds=getpreds_model_mem(None, inputs, h5file,modelkwargs,model_weights,combine,concatenatenn)
        #logger.info("Predicitons obtained")
        return preds


def get3Ddata(catalog, colnames):
    """
    Function to extract a 3D numpy array from some columns of an astropy catalog.
    The point is to make all columns have the same 2D shape, even if they were 1D.

    The 3D output array has shape (ncase, nrea, nfeat).
    """

    ncase = len(catalog)

    if len(colnames) == 0:
        raise RuntimeError("No colnames to get data from!")

    # Check for exotic catalogs (do they even exist ?)
    for colname in colnames:
        if not catalog[colname].ndim in [1, 2]:
            raise RuntimeError("Can only work with 1D or 2D columns")

    # Let's check the depths of the 2D colums to see what size we need.
    nreas = list(set([catalog[colname].shape[1] for colname in colnames if catalog[colname].ndim == 2]))
    #logger.info("We have the following nreas different from one in there: {}".format(nreas))
    if len(nreas) > 1:
        raise RuntimeError("The columns have incompatible depths!")

    if len(nreas) == 0:
        nrea = 1
        logger.debug("For each column, only one realization is available.")

    else:
        nrea = nreas[0]
        logger.debug("Extracting data from {0} realizations...".format(nrea))

    # Legacy-code from MomentsML:
    #if "ngroup" in catalog.meta:
    #    if nrea != catalog.meta["ngroup"]:
    #        raise RuntimeError("Something very fishy: depth is not ngroup!")

    # And now we get the data:

    readycols = []
    for colname in colnames:

        col = np.ma.array(catalog[colname])

        if col.ndim == 2: # This is already (ncase, nrea)
            pass

        elif col.ndim == 1:
            # This column has only one realization, and we have to "duplicate" it nrea times...
            col = np.tile(col, (nrea, 1)).transpose() # (ncase, nrea)
            # Strangely, there is no np.ma.tile... Unit-test that masks are done correctly here!
            # In addition, let's check shapes here as well:
            assert col.shape == (ncase, nrea)
            assert col.mask.shape == (ncase, nrea)
            #print(col)

        else:
            raise RuntimeError("Weird column dimension")

        assert col.shape == (ncase, nrea)
        #readycols.append(np.ma.array(col))
        readycols.append(col)

    outarray = np.ma.dstack(readycols)

    #outarray = np.ma.array(readycols)
    #outarray = np.rollaxis(np.ma.array(readycols), 1)

    assert outarray.ndim == 3
    assert outarray.shape[0] == ncase
    assert outarray.shape[1] == nrea
    assert outarray.shape[2] == len(colnames)

    return outarray


def readconfig(configpath):
    """
    Reads in a config file
    """
    if type(configpath)==ConfigParser:
        return configpath
    
    config = ConfigParser(allow_no_value=True, inline_comment_prefixes=('#', ';'))
    #from localconfig import config
    
    if not os.path.exists(configpath):
        raise RuntimeError("Config file '{}' does not exist!".format(configpath))
    logger.debug("Reading config from '{}'...".format(configpath))
    config.read(configpath)
    
    return config

def get_modelkwargs(configfile=None, input_shape=None, use_mask=True, use_caseweights=False, dropout_prob=0.0, noutputs=1):
    from . import layer
    ## all this for getting features.shape only
    logger.info("Reading config file %s \n"%(configfile))
    conf= readconfig(configfile)

    hiden_sizes=list(eval(conf.get("net_model", "hidden_sizes")))
    activation=conf.get("net_model", "activation")
    loss_name=conf.get("net_model", "loss_name")
    try: in_activation=conf.get("net_model", "in_activation")
    except: in_activation=None
    try: lamb=eval(conf.get("net_model", "lambda"))
    except: lamb=None
    try: dropout_prob=eval(conf.get("net_model", 'dropout_prob'))
    except: dropout_prob=0.0
    try: l2=eval(conf.get("net_model", 'l2'))
    except: l2=False
    try: lambda_value=eval(conf.get("net_model", 'lambda_value'))
    except: lambda_value=0.0
    try: l1=eval(conf.get("net_model", 'l1'))
    except: l1=False
    try: batchnormalization=eval(conf.get("net_model", 'batchnormalization'))
    except: batchnormalization=False
    out_activation=conf.get("net_model", "out_activation")
    lay=eval(conf.get("net_model", "layer"))
    if (out_activation==""): out_activation=None
    if (activation==""): activation=None
    if (in_activation==""): in_activation=None
    if (lamb==""): lamb=None
    modelkwargs={"input_shape": input_shape,
                 "hidden_sizes":hiden_sizes,
                 "layer":lay,
                 "activation":activation,
                 "in_activation":in_activation,
                 "out_activation":out_activation,
                 "loss_name":loss_name,
                 "use_mask":use_mask,
                 "use_caseweights":use_caseweights,
                 "lamb": lamb,
                 "dropout_prob":dropout_prob,
                 "noutputs":noutputs,
                 "training":False,
                 "dropout_prob":dropout_prob,
                 "l2":l2,
                 "l1":l1,
                 "lambda_value": lambda_value,
                 "batchnormalization":batchnormalization
    }

    return modelkwargs


def get_modelkwargs_wm(configfile=None,input_shape_w=None,input_shape_m=None,  use_mask=True, noutputs=1):
    ## all this for getting features.shape only
    from . import layer
    logger.info("Reading config file %s \n"%(configfile))
    conf=readconfig(configfile)
    
    hiden_sizes_w=list(eval(conf.get("net_model", "hidden_sizes_w")))
    try: in_activation_w=conf.get("net_model", "in_activation_w")
    except: in_activation_w=None
    activation_w=conf.get("net_model", "activation_w")
    out_activation_w=conf.get("net_model", "out_activation_w")
    lay_w=eval(conf.get("net_model", "layer_w"))
    hiden_sizes_m=list(eval(conf.get("net_model", "hidden_sizes_m")))
    try: in_activation_m=conf.get("net_model", "in_activation_m")
    except: in_activation_m=None
    activation_m=conf.get("net_model", "activation_m")
    out_activation_m=conf.get("net_model", "out_activation_m")
    lay_m=eval(conf.get("net_model", "layer_m"))
    loss_name=conf.get("net_model", "loss_name")
    try: lamb=eval(conf.get("net_model", "lambda"))
    except: lamb=None
    
    if (out_activation_w==""): out_activation_w=None
    if (out_activation_m==""): out_activation_m=None
    if (in_activation_w==""): in_activation_w=None
    if (in_activation_m==""): in_activation_m=None
    if (activation_w==""): activation_w=None
    if (activation_m==""): activation_m=None
    if (lamb==""): lamb=None
        
    modelkwargs={"input_shape_w":input_shape_w,
                 "hidden_sizes_w":hiden_sizes_w,
                 "layer_w":lay_w,
                 "activation_w":activation_w,
                 "in_activation_w":in_activation_w,
                 "out_activation_w":out_activation_w,
                 "input_shape_m":input_shape_m,
                 "hidden_sizes_m":hiden_sizes_m,
                 "layer_m":lay_m,
                 "activation_m":activation_m,
                 "in_activation_m":in_activation_m,
                 "out_activation_m":out_activation_m,
                 "loss_name":loss_name,
                 "use_mask":use_mask,
                 "lamb": lamb,
                 "noutputs": noutputs,

    }
    logger.info("Modelkwargs obtained")
    return modelkwargs
  
