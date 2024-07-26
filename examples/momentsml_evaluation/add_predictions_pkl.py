import sys, os, glob
import MLTF
import numpy as np
import pickle
import copy



import logging
logger = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Main script for training and validating fiducial simulations')
 
    parser.add_argument('--cat',
                        default=None, 
                        help='Catalog')

    parser.add_argument('--rad_model_config',
                        default=None, 
                        help='Config file containing network properties of trurad model (apply smart size a first selection)')
    parser.add_argument('--rad_normer_feats',
                        default=None, 
                        help='normed used for the training of truerad feats')
    parser.add_argument('--rad_normer_targets',
                        default=None, 
                        help='normed used for the training of truerad targets')
    parser.add_argument('--rad_features',
                        default=[],
                        nargs='+',
                        help='Features for training mu')
    
    parser.add_argument('--g_features',
                        default=[],
                        nargs='+',
                        help='Features for training g')
    parser.add_argument('--g_model_config',
                        default=None, 
                        help='Config file containing network properties of g model (we need g model to train weights)')
    parser.add_argument('--g_normer',
                        default=None, 
                        help='normed used for the training of g')

    parser.add_argument('--ws_features',
                        default=[],
                        nargs='+',
                        help='Features for training ws')
    parser.add_argument('--ws_model_config',
                        default=None, 
                        help='Config file containing network properties of ws model')
    parser.add_argument('--ws_normer',
                        default=None, 
                        help='normed used for the training of ws')

    parser.add_argument('--w_features',
                        default=[],
                        nargs='+',
                        help='Features for training w')
    parser.add_argument('--m_features',
                        default=[],
                        nargs='+',
                        help='Features for training w')
    parser.add_argument('--wm_model_config',
                        default=None, 
                        help='Config file containing network properties of wm model')
    parser.add_argument('--w_normer',
                        default=None, 
                        help='normed used for the training of w')
    parser.add_argument('--m_normer',
                        default=None, 
                        help='normed used for the training of w')


    args = parser.parse_args()
    return args


def add_shear_preds(cat, shearfiles):
    [shear_model_config, shear_feats_normer_name, inputlabels]=shearfiles
    with open(shear_feats_normer_name, 'rb') as handle: shear_feats_normer = pickle.load(handle)
    features = MLTF.tools_momentsml.get3Ddata(cat, inputlabels )

    point_modelkwargs=MLTF.tools_momentsml.get_modelkwargs(configfile=shear_model_config, input_shape=features[0].shape, use_mask=True, noutputs=2)

    weights=[os.path.join(wfold, "weights.ckpt") for wfold in glob.glob("%s/**/"%(os.path.dirname(shear_model_config)))]
    shear_preds=MLTF.tools_momentsml.getpreds_model(shear_feats_normer(features).astype('float32'),
                                                    modelkwargs= point_modelkwargs, model_weights=weights,
                                                    useprocess=True, combine="mean", concatenatenn=True)
    for i,n in enumerate(["pred_s1","pred_s2"]):
        cat[n]=shear_preds[:,:,i]

def add_rad_preds(cat, radfiles):
    [rad_model_config, rad_feats_names, rad_targets_names,inputlabels]=radfiles
    with open(rad_feats_names, 'rb') as handle: rad_features_normer = pickle.load(handle)
    with open(rad_targets_names, 'rb') as handle: rad_targets_normer = pickle.load(handle)
    features = MLTF.tools_momentsml.get3Ddata(cat, inputlabels)
    point_modelkwargs=MLTF.tools_momentsml.get_modelkwargs(configfile=rad_model_config, input_shape=features[0].shape, use_mask=True, noutputs=1)
    
    weights=[os.path.join(wfold, "weights.ckpt") for wfold in glob.glob("%s/**/"%(os.path.dirname(rad_model_config)))]
    rad_preds=MLTF.tools_momentsml.getpreds_model(rad_features_normer(features).astype('float32'), modelkwargs= point_modelkwargs, model_weights=weights, useprocess=True, combine="mean")

    rad_preds=rad_targets_normer.denorm(rad_preds)
    cat["pre_rad"]=rad_preds[:,:, 0]

def add_ws_preds(cat, wsfiles):
    [ws_model_config, ws_feats_names, inputlabels]=wsfiles
    with open(ws_feats_names, 'rb') as handle: ws_features_normer = pickle.load(handle)

    features = MLTF.tools_momentsml.get3Ddata(cat, inputlabels)
    point_modelkwargs=MLTF.tools_momentsml.get_modelkwargs(configfile=ws_model_config, input_shape=features[0].shape, use_mask=True, noutputs=1)
    
    weights=[os.path.join(wfold, "weights.ckpt") for wfold in glob.glob("%s/**/"%(os.path.dirname(ws_model_config)))]
    ws_preds=MLTF.tools_momentsml.getpreds_model(ws_features_normer(features).astype('float32'), modelkwargs= point_modelkwargs, model_weights=weights, useprocess=True, combine="mean")
    
    cat["ws"]=ws_preds[:,:,0]

def add_wm_preds(cat, wmfiles):
    [wm_model_config, w_feats_normer_name, m_feats_normer_name, inputlabels_w, inputlabels_m, ]=wmfiles
    with open(w_feats_normer_name, 'rb') as handle: w_feats_normer = pickle.load(handle)
    with open(m_feats_normer_name, 'rb') as handle: m_feats_normer = pickle.load(handle)
    features_w = MLTF.tools_momentsml.get3Ddata(cat, inputlabels_w)
    features_m = MLTF.tools_momentsml.get3Ddata(cat, inputlabels_m)
    inputs=[w_feats_normer(features_w).astype('float32'), m_feats_normer(features_m).astype('float32')]
    modelkwargs=MLTF.tools_momentsml.get_modelkwargs_wm(configfile=wm_model_config, input_shape_w=features_w[0].shape, input_shape_m=features_m[0].shape, use_mask=True, noutputs=1)

    weights=[os.path.join(wfold, "weights.ckpt") for wfold in glob.glob("%s/**/"%(os.path.dirname(wm_model_config)))]
    w_preds, m_preds=MLTF.tools_momentsml.getpreds_model(inputs,
                                                    modelkwargs= modelkwargs, model_weights=weights,
                                                    useprocess=True, combine="mean")
    cat["w"]=w_preds[:,:,0]
    cat["m"]=m_preds[:,:,0]

       
  
def main():    
    args = parse_args()
    loggerformat='PID %(process)06d | %(asctime)s | %(levelname)s: %(name)s(%(funcName)s): %(message)s'
    #logging.basicConfig(format=loggerformat, level=logging.DEBUG)
    logging.basicConfig(format=loggerformat, level=logging.INFO)
    #logging.basicConfig(format=loggerformat, level=logging.NOTSET)
    

    catalog=args.cat
    logger.info("Reading %s"%(catalog))
    assert os.path.isfile(catalog)
    
    picklecat=catalog.replace(".fits",".pkl")
    assert os.path.isfile(picklecat)

    with open(picklecat, 'rb') as handle:
        cat = pickle.load(handle)

    wmfiles=[args.wm_model_config, args.w_normer,  args.m_normer,  args.w_features, args.m_features]
    add_wm_preds(cat, wmfiles)
    
    wsfiles=[args.ws_model_config, args.ws_normer, args.ws_features]
    add_ws_preds(cat, wsfiles)
            
    radfiles=[args.rad_model_config, args.rad_normer_feats, args.rad_normer_targets, args.rad_features]
    add_rad_preds(cat, radfiles)

    shearfiles=[args.g_model_config, args.g_normer,  args.g_features]
    add_shear_preds(cat, shearfiles)

    overwrite=True
    if overwrite:
        with open(picklecat, 'wb') as handle:
            pickle.dump(cat, handle, -1)
  
            
if __name__ == "__main__":
    main()
