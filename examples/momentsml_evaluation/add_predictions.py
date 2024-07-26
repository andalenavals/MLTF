import os, glob
import MLTF
import numpy as np
from astropy.io import fits
import pickle




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
                        help='Config file containing network properties of g model')
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


def get_shear_preds(cat, shearfiles):
    [shear_model_config, shear_feats_normer_name, inputlabels]=shearfiles
    with open(shear_feats_normer_name, 'rb') as handle: shear_feats_normer = pickle.load(handle)
    features=np.array([np.array([cat[f] for f in inputlabels]).T])
    point_modelkwargs=MLTF.tools_momentsml.get_modelkwargs(configfile=shear_model_config, input_shape=features[0].shape, use_mask=True, noutputs=2)

    weights=[os.path.join(wfold, "weights.ckpt") for wfold in glob.glob("%s/**/"%(os.path.dirname(shear_model_config)))]
    shear_preds=MLTF.tools_momentsml.getpreds_model(shear_feats_normer(features).astype('float32'),
                                                    modelkwargs= point_modelkwargs, model_weights=weights,
                                                    useprocess=True, combine="mean", concatenatenn=True)
    return shear_preds[0,:,:]

def get_rad_preds(cat, radfiles):
    [rad_model_config, rad_feats_names, rad_targets_names,inputlabels]=radfiles
    with open(rad_feats_names, 'rb') as handle: rad_features_normer = pickle.load(handle)
    with open(rad_targets_names, 'rb') as handle: rad_targets_normer = pickle.load(handle)

    features=np.array([np.array([cat[f] for f in inputlabels]).T])
    point_modelkwargs=MLTF.tools_momentsml.get_modelkwargs(configfile=rad_model_config, input_shape=features[0].shape, use_mask=True, noutputs=1)
    
    weights=[os.path.join(wfold, "weights.ckpt") for wfold in glob.glob("%s/**/"%(os.path.dirname(rad_model_config)))]
    rad_preds=MLTF.tools_momentsml.getpreds_model(rad_features_normer(features).astype('float32'), modelkwargs= point_modelkwargs, model_weights=weights, useprocess=True, combine="mean")

    rad_preds=rad_targets_normer.denorm(rad_preds)
    return rad_preds[0,:,0]

def get_ws_preds(cat, wsfiles):
    [ws_model_config, ws_feats_names, inputlabels]=wsfiles
    with open(ws_feats_names, 'rb') as handle: ws_features_normer = pickle.load(handle)

    features=np.array([np.array([cat[f] for f in inputlabels]).T])
    point_modelkwargs=MLTF.tools_momentsml.get_modelkwargs(configfile=ws_model_config, input_shape=features[0].shape, use_mask=True, noutputs=1)
    
    weights=[os.path.join(wfold, "weights.ckpt") for wfold in glob.glob("%s/**/"%(os.path.dirname(ws_model_config)))]
    ws_preds=MLTF.tools_momentsml.getpreds_model(ws_features_normer(features).astype('float32'), modelkwargs= point_modelkwargs, model_weights=weights, useprocess=True, combine="mean")
    
    return ws_preds[0,:,0]

def get_wm_preds(cat, wmfiles):
    [wm_model_config, w_feats_normer_name, m_feats_normer_name, inputlabels_w, inputlabels_m, ]=wmfiles
    with open(w_feats_normer_name, 'rb') as handle: w_feats_normer = pickle.load(handle)
    with open(m_feats_normer_name, 'rb') as handle: m_feats_normer = pickle.load(handle)
    features_w=np.array([np.array([cat[f] for f in inputlabels_w]).T])
    features_m=np.array([np.array([cat[f] for f in inputlabels_m]).T])
    inputs=[w_feats_normer(features_w).astype('float32'), m_feats_normer(features_m).astype('float32')]
    modelkwargs=MLTF.tools_momentsml.get_modelkwargs_wm(configfile=wm_model_config, input_shape_w=features_w[0].shape, input_shape_m=features_m[0].shape, use_mask=True, noutputs=1)

    weights=[os.path.join(wfold, "weights.ckpt") for wfold in glob.glob("%s/**/"%(os.path.dirname(wm_model_config)))]
    w_preds, m_preds=MLTF.tools_momentsml.getpreds_model(inputs,
                                                    modelkwargs= modelkwargs, model_weights=weights,
                                                    useprocess=True, combine="mean")
    return w_preds[0,:,0], m_preds[0,:,0]
   
       
  
def main():    
    args = parse_args()
    loggerformat='PID %(process)06d | %(asctime)s | %(levelname)s: %(name)s(%(funcName)s): %(message)s'
    #logging.basicConfig(format=loggerformat, level=logging.DEBUG)
    logging.basicConfig(format=loggerformat, level=logging.INFO)
    #logging.basicConfig(format=loggerformat, level=logging.NOTSET)
    


    assert os.path.isfile(args.cat)
    with fits.open(args.cat) as hdul:
        cat=hdul[1].data
        columns = hdul[1].columns

    wmfiles=[args.wm_model_config, args.w_normer,  args.m_normer,  args.w_features, args.m_features]
    w_preds,m_preds=get_wm_preds(cat, wmfiles)
    
    wsfiles=[args.ws_model_config, args.ws_normer, args.ws_features]
    ws_preds=get_ws_preds(cat, wsfiles)

    radfiles=[args.rad_model_config, args.rad_normer_feats, args.rad_normer_targets, args.rad_features]
    rad_preds=get_rad_preds(cat, radfiles)
    
    shearfiles=[args.g_model_config, args.g_normer,  args.g_features]
    shearpreds=get_shear_preds(cat, shearfiles)

    list_preds=[w_preds, m_preds, ws_preds, rad_preds, shearpreds[:,0], shearpreds[:,1]]
    list_names=["w", "m", "ws", "pred_rad", "pred_s1", "pred_s2"]
    
    for arr,colname in zip(list_preds,list_names):
        if colname not in columns.names:
            columns+=fits.Column(name=colname, format='D', array=arr)
        
    hdu = fits.BinTableHDU.from_columns(fits.ColDefs(columns))
    for arr,colname in zip(list_preds,list_names):
        hdu.data[colname] = arr
    hdu.writeto(args.cat, overwrite=True)
  
  
            
if __name__ == "__main__":
    main()
